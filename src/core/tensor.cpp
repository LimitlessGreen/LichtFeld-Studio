/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include "kernels/training_kernels.cuh"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <print>
#include <sstream>

namespace gs {

// Helper to check CUDA errors
#define CHECK_CUDA(x)                                                                          \
    do {                                                                                       \
        cudaError_t err = x;                                                                   \
        if (err != cudaSuccess) {                                                              \
            LOG_ERROR("CUDA error: {} at {}:{}", cudaGetErrorString(err), __FILE__, __LINE__); \
        }                                                                                      \
    } while (0)

    // ============= Static Members =============
    std::atomic<size_t> Tensor::next_id_{1};

    // ============= TensorShape Implementation =============
    std::string TensorShape::str() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < dims_.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << dims_[i];
        }
        oss << "]";
        return oss.str();
    }

    // ============= Tensor Constructors =============
    Tensor::Tensor(void* data, TensorShape shape, Device device, DataType dtype)
        : data_(data),
          shape_(shape),
          device_(device),
          dtype_(dtype),
          owns_memory_(false),
          id_(next_id_++) {
        LOG_TRACE("Created tensor #{} (view): shape={}, device={}, dtype={}",
                  id_, shape_.str(), device_name(device_), dtype_name(dtype_));
    }

    Tensor::Tensor(Tensor&& other) noexcept
        : data_(other.data_),
          shape_(std::move(other.shape_)),
          device_(other.device_),
          dtype_(other.dtype_),
          owns_memory_(other.owns_memory_),
          id_(other.id_) {

        // Clear the source
        other.data_ = nullptr;
        other.owns_memory_ = false;

        LOG_TRACE("Moved tensor #{} to tensor #{}", other.id_, id_);
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            // Free existing memory if we own it
            if (owns_memory_ && data_) {
                if (device_ == Device::CUDA) {
                    cudaFree(data_);
                } else {
                    std::free(data_);
                }
            }

            // Move data
            data_ = other.data_;
            shape_ = std::move(other.shape_);
            device_ = other.device_;
            dtype_ = other.dtype_;
            owns_memory_ = other.owns_memory_;
            id_ = other.id_;

            // Clear source
            other.data_ = nullptr;
            other.owns_memory_ = false;

            LOG_TRACE("Move-assigned tensor #{} to tensor #{}", other.id_, id_);
        }
        return *this;
    }

    Tensor::~Tensor() {
        if (owns_memory_ && data_) {
            LOG_TRACE("Destroying tensor #{} (freeing memory)", id_);
            if (device_ == Device::CUDA) {
                cudaFree(data_);
            } else {
                std::free(data_);
            }
        }
    }

    // ============= Factory Methods =============
    Tensor Tensor::empty(TensorShape shape, Device device, DataType dtype) {
        size_t bytes = shape.elements() * dtype_size(dtype);
        if (bytes == 0) {
            LOG_WARN("Creating empty tensor with 0 elements");
            return Tensor();
        }

        void* data = nullptr;
        if (device == Device::CUDA) {
            CHECK_CUDA(cudaMalloc(&data, bytes));
        } else {
            data = std::malloc(bytes);
            if (!data) {
                LOG_ERROR("Failed to allocate {} bytes on CPU", bytes);
                return Tensor();
            }
        }

        Tensor t(data, shape, device, dtype);
        t.owns_memory_ = true;

        LOG_DEBUG("Allocated tensor #{}: {} bytes, shape={}, device={}",
                  t.id_, bytes, shape.str(), device_name(device));

        return t;
    }

    Tensor Tensor::zeros(TensorShape shape, Device device, DataType dtype) {
        Tensor t = empty(shape, device, dtype);
        if (t.is_valid()) {
            t.zero_();
        }
        return t;
    }

    Tensor Tensor::ones(TensorShape shape, Device device, DataType dtype) {
        Tensor t = empty(shape, device, dtype);
        if (t.is_valid()) {
            t.fill_(1.0f);
        }
        return t;
    }

    Tensor Tensor::full(TensorShape shape, float value, Device device, DataType dtype) {
        Tensor t = empty(shape, device, dtype);
        if (t.is_valid()) {
            t.fill_(value);
        }
        return t;
    }

    // ============= View Operations =============
    Tensor Tensor::view(TensorShape new_shape) const {
        size_t new_elements = new_shape.elements();

        // Allow -1 in one dimension to infer size
        bool has_infer = false;
        size_t infer_dim = 0;
        size_t known_elements = 1;

        for (size_t i = 0; i < new_shape.rank(); ++i) {
            if (new_shape[i] == static_cast<size_t>(-1)) {
                if (has_infer) {
                    LOG_ERROR("Can only have one inferred dimension in view");
                    return Tensor();
                }
                has_infer = true;
                infer_dim = i;
            } else {
                known_elements *= new_shape[i];
            }
        }

        if (has_infer) {
            if (shape_.elements() % known_elements != 0) {
                LOG_ERROR("Cannot infer dimension: {} is not divisible by {}",
                          shape_.elements(), known_elements);
                return Tensor();
            }
            // Create mutable shape to modify
            std::vector<size_t> dims = new_shape.dims();
            dims[infer_dim] = shape_.elements() / known_elements;
            new_shape = TensorShape(dims);
            new_elements = new_shape.elements();
        }

        if (new_elements != shape_.elements()) {
            LOG_ERROR("View shape {} has {} elements, but tensor has {} elements",
                      new_shape.str(), new_elements, shape_.elements());
            return Tensor();
        }

        return Tensor(data_, new_shape, device_, dtype_);
    }

    Tensor Tensor::slice(size_t dim, size_t start, size_t end) const {
        if (dim >= shape_.rank()) {
            LOG_ERROR("Slice dimension {} out of range for rank {}", dim, shape_.rank());
            return Tensor();
        }

        if (start >= end || end > shape_[dim]) {
            LOG_ERROR("Invalid slice range [{}, {}) for dimension {} of size {}",
                      start, end, dim, shape_[dim]);
            return Tensor();
        }

        // Calculate new shape
        std::vector<size_t> new_dims = shape_.dims();
        new_dims[dim] = end - start;
        TensorShape new_shape(new_dims);

        // Calculate offset in elements
        size_t offset = 0;
        size_t stride = 1;
        for (size_t i = shape_.rank(); i > dim; --i) {
            stride *= shape_[i - 1];
        }
        offset = start * stride;

        // Calculate offset in bytes
        size_t byte_offset = offset * dtype_size(dtype_);
        void* new_data = static_cast<char*>(data_) + byte_offset;

        return Tensor(new_data, new_shape, device_, dtype_);
    }

    Tensor Tensor::squeeze(int dim) const {
        if (dim < 0)
            dim = shape_.rank() + dim;

        if (dim >= static_cast<int>(shape_.rank()) || shape_[dim] != 1) {
            // Return a view of the same tensor (no copy)
            return Tensor(data_, shape_, device_, dtype_);
        }

        std::vector<size_t> new_dims;
        for (size_t i = 0; i < shape_.rank(); ++i) {
            if (static_cast<int>(i) != dim) {
                new_dims.push_back(shape_[i]);
            }
        }

        return view(TensorShape(new_dims));
    }

    Tensor Tensor::unsqueeze(int dim) const {
        if (dim < 0)
            dim = shape_.rank() + dim + 1;

        std::vector<size_t> new_dims = shape_.dims();
        new_dims.insert(new_dims.begin() + dim, 1);

        return view(TensorShape(new_dims));
    }

    Tensor Tensor::permute(std::vector<int> dims) const {
        // For now, return clone() - implement proper permutation later
        LOG_WARN_ONCE("Permute not fully implemented - returning copy");
        return clone();
    }

    Tensor Tensor::transpose(int dim1, int dim2) const {
        std::vector<int> dims;
        for (size_t i = 0; i < shape_.rank(); ++i) {
            dims.push_back(static_cast<int>(i));
        }
        std::swap(dims[dim1], dims[dim2]);
        return permute(dims);
    }

    Tensor Tensor::flatten(int start_dim, int end_dim) const {
        if (start_dim < 0)
            start_dim = shape_.rank() + start_dim;
        if (end_dim < 0)
            end_dim = shape_.rank() + end_dim;

        std::vector<size_t> new_dims;

        // Keep dimensions before start_dim
        for (int i = 0; i < start_dim; ++i) {
            new_dims.push_back(shape_[i]);
        }

        // Flatten dimensions from start_dim to end_dim
        size_t flattened_size = 1;
        for (int i = start_dim; i <= end_dim; ++i) {
            flattened_size *= shape_[i];
        }
        new_dims.push_back(flattened_size);

        // Keep dimensions after end_dim
        for (size_t i = end_dim + 1; i < shape_.rank(); ++i) {
            new_dims.push_back(shape_[i]);
        }

        return view(TensorShape(new_dims));
    }

    // ============= Memory Operations =============
    Tensor Tensor::to(Device device) const {
        if (device_ == device) {
            return clone(); // Same device, just clone
        }

        Tensor result = empty(shape_, device, dtype_);
        if (!result.is_valid()) {
            return result;
        }

        size_t bytes = this->bytes();

        if (device_ == Device::CPU && device == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.data_, data_, bytes, cudaMemcpyHostToDevice));
        } else if (device_ == Device::CUDA && device == Device::CPU) {
            CHECK_CUDA(cudaMemcpy(result.data_, data_, bytes, cudaMemcpyDeviceToHost));
        }

        LOG_TRACE("Copied tensor #{} from {} to {} (new tensor #{})",
                  id_, device_name(device_), device_name(device), result.id_);

        return result;
    }

    Tensor Tensor::clone() const {
        Tensor result = empty(shape_, device_, dtype_);
        if (!result.is_valid()) {
            return result;
        }

        size_t bytes = this->bytes();

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.data_, data_, bytes, cudaMemcpyDeviceToDevice));
        } else {
            std::memcpy(result.data_, data_, bytes);
        }

        LOG_TRACE("Cloned tensor #{} to tensor #{}", id_, result.id_);

        return result;
    }

    void Tensor::copy_from(const Tensor& other) {
        if (shape_ != other.shape_) {
            LOG_ERROR("Cannot copy from tensor with different shape: {} vs {}",
                      shape_.str(), other.shape_.str());
            return;
        }

        if (dtype_ != other.dtype_) {
            LOG_ERROR("Cannot copy from tensor with different dtype: {} vs {}",
                      dtype_name(dtype_), dtype_name(other.dtype_));
            return;
        }

        size_t bytes = this->bytes();

        if (device_ == Device::CUDA && other.device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(data_, other.data_, bytes, cudaMemcpyDeviceToDevice));
        } else if (device_ == Device::CUDA && other.device_ == Device::CPU) {
            CHECK_CUDA(cudaMemcpy(data_, other.data_, bytes, cudaMemcpyHostToDevice));
        } else if (device_ == Device::CPU && other.device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(data_, other.data_, bytes, cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(data_, other.data_, bytes);
        }
    }

    void Tensor::fill_(float value) {
        if (!is_valid()) {
            LOG_ERROR("Cannot fill invalid tensor");
            return;
        }

        if (dtype_ != DataType::Float32) {
            LOG_ERROR("Fill only implemented for float32 tensors");
            return;
        }

        if (device_ == Device::CUDA) {
            gs::training::launch_fill_tensor(ptr<float>(), value, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] = value;
            }
        }
    }

    void Tensor::zero_() {
        if (!is_valid()) {
            LOG_ERROR("Cannot zero invalid tensor");
            return;
        }

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemset(data_, 0, bytes()));
        } else {
            std::memset(data_, 0, bytes());
        }
    }

    // ============= Debug Utilities =============
    std::string Tensor::str() const {
        std::ostringstream oss;
        oss << "Tensor(#" << id_ << ", ";

        if (!is_valid()) {
            oss << "invalid)";
            return oss.str();
        }

        oss << "shape=" << shape_.str()
            << ", dtype=" << dtype_name(dtype_)
            << ", device=" << device_name(device_);

        if (owns_memory_) {
            oss << ", owned";
        } else {
            oss << ", view";
        }

        oss << ")";
        return oss.str();
    }

    void Tensor::print(const std::string& name) const {
        std::string prefix = name.empty() ? "Tensor" : name;

        if (!is_valid()) {
            LOG_INFO("{}: {}", prefix, str());
            return;
        }

        // Get a few values for display
        auto values = debug_values(10);

        std::ostringstream oss;
        oss << str() << "\n  Values: [";

        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << std::fixed << std::setprecision(4) << values[i];
            if (i == 9 && numel() > 10) {
                oss << ", ... (" << numel() - 10 << " more)";
            }
        }
        oss << "]";

        LOG_INFO("{}: {}", prefix, oss.str());
    }

    void Tensor::print_formatted(const std::string& name, size_t max_per_dim) const {
        std::println("\n=== {} ===", name.empty() ? "Tensor" : name);
        std::println("{}", str());

        if (shape_.rank() == 1) {
            print_1d(max_per_dim);
        } else if (shape_.rank() == 2) {
            print_2d(max_per_dim);
        } else {
            std::println("  [Higher dimensional tensor - showing first slice]");
            auto first_slice = slice(0, 0, 1);
            first_slice.squeeze(0).print_2d(max_per_dim);
        }
    }

    void Tensor::print_1d(size_t max_elem) const {
        auto values = debug_values(std::min(max_elem, numel()));
        std::print("  [");

        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0)
                std::print(", ");
            std::print("{:8.4f}", values[i]);
        }

        if (numel() > max_elem) {
            std::print(", ... ({} more)", numel() - max_elem);
        }
        std::println("]");
    }

    void Tensor::print_2d(size_t max_per_dim) const {
        if (shape_.rank() != 2)
            return;

        size_t rows = std::min(max_per_dim, shape_[0]);
        size_t cols = std::min(max_per_dim, shape_[1]);

        // Get values
        auto values = debug_values(shape_[0] * shape_[1]);

        for (size_t i = 0; i < rows; ++i) {
            std::print("  [");
            for (size_t j = 0; j < cols; ++j) {
                if (j > 0)
                    std::print(", ");
                size_t idx = i * shape_[1] + j;
                std::print("{:8.4f}", values[idx]);
            }
            if (shape_[1] > cols) {
                std::print(", ... ({} more)", shape_[1] - cols);
            }
            std::print("]");

            if (i == rows - 1 && shape_[0] > rows) {
                std::print("  ... ({} more rows)", shape_[0] - rows);
            }
            std::println("");
        }
    }

    std::vector<float> Tensor::debug_values(size_t max_values) const {
        std::vector<float> values;

        if (!is_valid() || dtype_ != DataType::Float32) {
            return values;
        }

        size_t n = std::min(max_values, numel());
        values.resize(n);

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(values.data(), data_, n * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        } else {
            const float* data = ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                values[i] = data[i];
            }
        }

        return values;
    }

    std::vector<float> Tensor::to_vector() const {
        if (dtype_ != DataType::Float32) {
            LOG_ERROR("to_vector only supports float32");
            return {};
        }

        std::vector<float> result(numel());

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.data(), data_, bytes(), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(result.data(), data_, bytes());
        }

        return result;
    }

    void Tensor::dump_diagnostic(const std::string& filename) const {
        std::ofstream file(filename);
        std::print(file, "=== Tensor Diagnostic Dump ===\n");
        std::print(file, "Info: {}\n", str());
        std::print(file, "Memory address: {}\n", data_);
        std::print(file, "Bytes: {}\n", bytes());

        if (device_ == Device::CPU || numel() < 10000) {
            auto values = to_vector();
            std::print(file, "Values ({} total):\n", values.size());
            for (size_t i = 0; i < std::min(size_t(1000), values.size()); ++i) {
                std::print(file, "[{}]: {}\n", i, values[i]);
            }
        }

        file.close();
        LOG_INFO("Diagnostic dump saved to {}", filename);
    }

#undef CHECK_CUDA

    // ============= Error Classes =============
    TensorError::TensorError(const std::string& msg, const Tensor* t)
        : std::runtime_error(msg),
          tensor_info_(t ? t->str() : "N/A") {}

    // ============= Performance Monitoring =============
    TensorTimer::TensorTimer(std::string name)
        : start_(Clock::now()),
          name_(std::move(name)) {}

    TensorTimer::~TensorTimer() {
        auto duration = Clock::now() - start_;
        auto ms = std::chrono::duration<double, std::milli>(duration).count();
        LOG_TRACE("{} took {:.3f}ms", name_, ms);
    }

    // ============= Stream Output =============
    std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << t.str();
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
        os << shape.str();
        return os;
    }

} // namespace gs
