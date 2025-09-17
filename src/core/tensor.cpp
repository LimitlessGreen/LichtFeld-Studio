/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_broadcast.hpp"
#include "core/tensor_ops.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <print>

#define CHECK_CUDA(call)                              \
    do {                                              \
        cudaError_t error = call;                     \
        if (error != cudaSuccess) {                   \
            LOG_ERROR("CUDA error at {}:{} - {}: {}", \
                      __FILE__, __LINE__,             \
                      cudaGetErrorName(error),        \
                      cudaGetErrorString(error));     \
        }                                             \
    } while (0)

namespace gs {

    std::atomic<size_t> Tensor::next_id_{1};

    // ============= Constructors & Destructor =============
    Tensor::Tensor(void* data, TensorShape shape, Device device, DataType dtype)
        : data_(data),
          shape_(shape),
          device_(device),
          dtype_(dtype),
          owns_memory_(false),
          initialized_(true),
          id_(next_id_++) {

        if (profiling_enabled_) {
            LOG_DEBUG("Created tensor #{} (view): shape={}, device={}, dtype={}",
                      id_, shape_.str(), device_name(device_), dtype_name(dtype_));
        }
    }

    Tensor::Tensor(Tensor&& other) noexcept
        : data_(other.data_),
          shape_(other.shape_),
          device_(other.device_),
          dtype_(other.dtype_),
          owns_memory_(other.owns_memory_),
          initialized_(other.initialized_),
          id_(other.id_) {

        other.data_ = nullptr;
        other.owns_memory_ = false;
        other.initialized_ = false;
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            // Clean up existing resources
            if (owns_memory_ && data_) {
                if (device_ == Device::CUDA) {
                    cudaFree(data_);
                } else {
                    delete[] static_cast<char*>(data_);
                }
            }

            // Move from other
            data_ = other.data_;
            shape_ = other.shape_;
            device_ = other.device_;
            dtype_ = other.dtype_;
            owns_memory_ = other.owns_memory_;
            initialized_ = other.initialized_;
            id_ = other.id_;

            // Reset other
            other.data_ = nullptr;
            other.owns_memory_ = false;
            other.initialized_ = false;
        }
        return *this;
    }

    Tensor::~Tensor() {
        if (owns_memory_ && data_) {
            if (profiling_enabled_) {
                LOG_DEBUG("Destroying tensor #{}: shape={}, device={}",
                          id_, shape_.str(), device_name(device_));
            }

            if (device_ == Device::CUDA) {
                cudaFree(data_);
            } else {
                delete[] static_cast<char*>(data_);
            }
        }
    }

    // ============= Factory Methods =============
    Tensor Tensor::empty(TensorShape shape, Device device, DataType dtype) {
        if (!shape.is_initialized()) {
            LOG_ERROR("Cannot create tensor with uninitialized shape");
            return Tensor();
        }

        size_t n_bytes = shape.elements() * dtype_size(dtype);
        if (n_bytes == 0) {
            // Valid empty tensor with no memory allocation
            Tensor t;
            t.shape_ = shape;
            t.device_ = device;
            t.dtype_ = dtype;
            t.owns_memory_ = false;
            t.initialized_ = true;
            t.id_ = next_id_++;
            return t;
        }

        void* data = nullptr;
        if (device == Device::CUDA) {
            cudaError_t err = cudaMalloc(&data, n_bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("Failed to allocate {} bytes on CUDA: {}",
                          n_bytes, cudaGetErrorString(err));
                return Tensor();
            }
        } else {
            data = new char[n_bytes];
            if (!data) {
                LOG_ERROR("Failed to allocate {} bytes on CPU", n_bytes);
                return Tensor();
            }
        }

        Tensor t;
        t.data_ = data;
        t.shape_ = shape;
        t.device_ = device;
        t.dtype_ = dtype;
        t.owns_memory_ = true;
        t.initialized_ = true;
        t.id_ = next_id_++;

        if (profiling_enabled_) {
            LOG_DEBUG("Created tensor #{} (owned): shape={}, device={}, dtype={}, bytes={}",
                      t.id_, shape.str(), device_name(device), dtype_name(dtype), n_bytes);
        }

        return t;
    }

    Tensor Tensor::zeros(TensorShape shape, Device device, DataType dtype) {
        if (dtype != DataType::Float32) {
            LOG_ERROR("Currently only float32 is supported for zeros");
            return Tensor();
        }

        auto t = empty(shape, device, dtype);
        if (t.is_valid() && t.numel() > 0) {
            if (device == Device::CUDA) {
                CHECK_CUDA(cudaMemset(t.data_, 0, t.bytes()));
            } else {
                std::memset(t.data_, 0, t.bytes());
            }
        }

        return t;
    }

    Tensor Tensor::ones(TensorShape shape, Device device, DataType dtype) {
        return full(shape, 1.0f, device, dtype);
    }

    Tensor Tensor::full(TensorShape shape, float value, Device device, DataType dtype) {
        if (dtype != DataType::Float32) {
            LOG_ERROR("Currently only float32 is supported for full");
            return Tensor();
        }

        auto t = empty(shape, device, dtype);
        if (!t.is_valid() || t.numel() == 0) {
            return t;
        }

        if (device == Device::CUDA) {
            // Fill on GPU
            std::vector<float> temp(t.numel(), value);
            CHECK_CUDA(cudaMemcpy(t.data_, temp.data(), t.bytes(), cudaMemcpyHostToDevice));
        } else {
            float* data = t.ptr<float>();
            std::fill(data, data + t.numel(), value);
        }

        return t;
    }

    // ============= Memory Operations =============
    Tensor Tensor::clone() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto t = empty(shape_, device_, dtype_);
        if (numel() == 0) {
            return t;
        }

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(t.data_, data_, bytes(), cudaMemcpyDeviceToDevice));
        } else {
            std::memcpy(t.data_, data_, bytes());
        }

        return t;
    }

    Tensor Tensor::contiguous() const {
        // For now, tensors are always contiguous
        return clone();
    }

    Tensor Tensor::to(Device device) const {
        if (!is_valid()) {
            return Tensor();
        }

        if (device_ == device) {
            return clone();
        }

        auto t = empty(shape_, device, dtype_);
        if (numel() == 0) {
            return t;
        }

        if (device_ == Device::CPU && device == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(t.data_, data_, bytes(), cudaMemcpyHostToDevice));
        } else if (device_ == Device::CUDA && device == Device::CPU) {
            CHECK_CUDA(cudaMemcpy(t.data_, data_, bytes(), cudaMemcpyDeviceToHost));
        }

        return t;
    }

    Tensor Tensor::to(DataType dtype) const {
        if (!is_valid()) {
            return Tensor();
        }

        if (dtype_ == dtype) {
            return clone();
        }

        // Support Bool -> Float32 conversion
        if (dtype_ == DataType::Bool && dtype == DataType::Float32) {
            auto result = empty(shape_, device_, DataType::Float32);

            if (numel() == 0) {
                return result;
            }

            if (device_ == Device::CUDA) {
                // Launch a simple conversion kernel
                tensor_ops::launch_bool_to_float(ptr<unsigned char>(), result.ptr<float>(),
                                                numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const unsigned char* src = ptr<unsigned char>();
                float* dst = result.ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = src[i] ? 1.0f : 0.0f;
                }
            }

            return result;
        }

        // Support Float32 -> Bool conversion
        if (dtype_ == DataType::Float32 && dtype == DataType::Bool) {
            auto result = empty(shape_, device_, DataType::Bool);

            if (numel() == 0) {
                return result;
            }

            if (device_ == Device::CUDA) {
                // Launch a simple conversion kernel
                tensor_ops::launch_float_to_bool(ptr<float>(), result.ptr<unsigned char>(),
                                                numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const float* src = ptr<float>();
                unsigned char* dst = result.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = (src[i] != 0.0f) ? 1 : 0;
                }
            }

            return result;
        }

        LOG_ERROR("Type conversion from {} to {} not implemented",
                  dtype_name(dtype_), dtype_name(dtype));
        return Tensor();
    }

    // ============= In-place fill with zeros =============
    Tensor& Tensor::zero_() {
        if (!is_valid() || numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemset(data_, 0, bytes()));
        } else {
            std::memset(data_, 0, bytes());
        }

        return *this;
    }

    // ============= Shape Operations =============
    Tensor Tensor::reshape(TensorShape new_shape) const {
        return view(new_shape);
    }

    Tensor Tensor::view(TensorShape new_shape) const {
        if (!is_valid()) {
            return Tensor();
        }

        size_t new_elements = new_shape.elements();

        // Check for -1 dimension (infer)
        int infer_dim = -1;
        size_t known_elements = 1;
        for (size_t i = 0; i < new_shape.rank(); ++i) {
            if (new_shape[i] == static_cast<size_t>(-1)) {
                if (infer_dim != -1) {
                    LOG_ERROR("Can only infer one dimension");
                    return Tensor();
                }
                infer_dim = i;
            } else {
                known_elements *= new_shape[i];
            }
        }

        // Infer dimension if needed
        if (infer_dim != -1) {
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

        Tensor t(data_, new_shape, device_, dtype_);
        t.initialized_ = true;
        return t;
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

        // Calculate the stride for the dimension we're slicing
        for (size_t i = shape_.rank(); i > dim + 1; --i) {
            stride *= shape_[i - 1];
        }

        // Calculate the offset to the start of the slice
        offset = start * stride;

        // Calculate offset in bytes
        size_t byte_offset = offset * dtype_size(dtype_);
        void* new_data = static_cast<char*>(data_) + byte_offset;

        // Return a view into the sliced data
        Tensor t(new_data, new_shape, device_, dtype_);
        t.initialized_ = true;
        return t;
    }

    Tensor Tensor::squeeze(int dim) const {
        std::vector<size_t> new_dims;

        if (dim == -1) {
            // Squeeze all dimensions of size 1
            for (size_t i = 0; i < shape_.rank(); ++i) {
                if (shape_[i] != 1) {
                    new_dims.push_back(shape_[i]);
                }
            }
        } else {
            // Handle negative indexing
            if (dim < 0) {
                dim = shape_.rank() + dim;
            }

            // Check bounds
            if (dim < 0 || dim >= static_cast<int>(shape_.rank())) {
                Tensor t(data_, shape_, device_, dtype_);
                t.initialized_ = true;
                return t;
            }

            // Only squeeze if dimension is 1
            if (shape_[dim] != 1) {
                Tensor t(data_, shape_, device_, dtype_);
                t.initialized_ = true;
                return t;
            }

            // Build new shape without the squeezed dimension
            for (size_t i = 0; i < shape_.rank(); ++i) {
                if (static_cast<int>(i) != dim) {
                    new_dims.push_back(shape_[i]);
                }
            }
        }

        // Handle edge case where all dims are 1
        if (new_dims.empty()) {
            new_dims.push_back(1);
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
        if (!is_valid()) {
            return Tensor();
        }

        // Handle negative indices
        if (dim1 < 0)
            dim1 = shape_.rank() + dim1;
        if (dim2 < 0)
            dim2 = shape_.rank() + dim2;

        // Check bounds
        if (dim1 < 0 || dim1 >= static_cast<int>(shape_.rank()) ||
            dim2 < 0 || dim2 >= static_cast<int>(shape_.rank())) {
            LOG_ERROR("Transpose dimensions out of bounds");
            return Tensor();
        }

        // If same dimension, just return clone
        if (dim1 == dim2) {
            return clone();
        }

        // Create new shape with swapped dimensions
        std::vector<size_t> new_shape = shape_.dims();
        std::swap(new_shape[dim1], new_shape[dim2]);

        auto result = empty(TensorShape(new_shape), device_, dtype_);

        // For now, implement a general transpose by copying data
        // This is not optimal but works correctly
        size_t total_elements = numel();

        if (device_ == Device::CUDA) {
            // Copy to CPU, transpose, copy back
            std::vector<float> src_data(total_elements);
            std::vector<float> dst_data(total_elements);

            cudaMemcpy(src_data.data(), ptr<float>(), total_elements * sizeof(float),
                       cudaMemcpyDeviceToHost);

            // Calculate strides for source and destination
            std::vector<size_t> src_strides(shape_.rank());
            std::vector<size_t> dst_strides(shape_.rank());

            src_strides[shape_.rank() - 1] = 1;
            dst_strides[shape_.rank() - 1] = 1;

            for (int i = shape_.rank() - 2; i >= 0; --i) {
                src_strides[i] = src_strides[i + 1] * shape_[i + 1];
                dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];
            }

            // Perform the transpose
            std::vector<size_t> indices(shape_.rank(), 0);

            for (size_t i = 0; i < total_elements; ++i) {
                // Calculate source linear index
                size_t src_idx = 0;
                for (size_t d = 0; d < shape_.rank(); ++d) {
                    src_idx += indices[d] * src_strides[d];
                }

                // Calculate destination indices (with dims swapped)
                std::vector<size_t> dst_indices = indices;
                std::swap(dst_indices[dim1], dst_indices[dim2]);

                // Calculate destination linear index
                size_t dst_idx = 0;
                for (size_t d = 0; d < shape_.rank(); ++d) {
                    dst_idx += dst_indices[d] * dst_strides[d];
                }

                dst_data[dst_idx] = src_data[src_idx];

                // Increment indices
                for (int d = shape_.rank() - 1; d >= 0; --d) {
                    indices[d]++;
                    if (indices[d] < shape_[d]) {
                        break;
                    }
                    indices[d] = 0;
                }
            }

            cudaMemcpy(result.ptr<float>(), dst_data.data(), total_elements * sizeof(float),
                       cudaMemcpyHostToDevice);
        } else {
            // CPU implementation (similar logic)
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();

            // Calculate strides
            std::vector<size_t> src_strides(shape_.rank());
            std::vector<size_t> dst_strides(shape_.rank());

            src_strides[shape_.rank() - 1] = 1;
            dst_strides[shape_.rank() - 1] = 1;

            for (int i = shape_.rank() - 2; i >= 0; --i) {
                src_strides[i] = src_strides[i + 1] * shape_[i + 1];
                dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];
            }

            // Perform the transpose
            std::vector<size_t> indices(shape_.rank(), 0);

            for (size_t i = 0; i < total_elements; ++i) {
                size_t src_idx = 0;
                for (size_t d = 0; d < shape_.rank(); ++d) {
                    src_idx += indices[d] * src_strides[d];
                }

                std::vector<size_t> dst_indices = indices;
                std::swap(dst_indices[dim1], dst_indices[dim2]);

                size_t dst_idx = 0;
                for (size_t d = 0; d < shape_.rank(); ++d) {
                    dst_idx += dst_indices[d] * dst_strides[d];
                }

                dst[dst_idx] = src[src_idx];

                for (int d = shape_.rank() - 1; d >= 0; --d) {
                    indices[d]++;
                    if (indices[d] < shape_[d]) {
                        break;
                    }
                    indices[d] = 0;
                }
            }
        }

        return result;
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

    // ============= Broadcasting Methods Implementation =============
    Tensor Tensor::expand(const TensorShape& target_shape) const {
        if (!is_valid()) {
            return Tensor();
        }

        return BroadcastHelper::expand(*this, target_shape);
    }

    Tensor Tensor::broadcast_to(const TensorShape& target_shape) const {
        return expand(target_shape);
    }

    bool Tensor::can_broadcast_to(const TensorShape& target) const {
        return BroadcastHelper::can_broadcast(shape_, target);
    }

    TensorShape Tensor::broadcast_shape(const TensorShape& other) const {
        return BroadcastHelper::broadcast_shape(shape_, other);
    }

    // ============= TensorShape Implementation =============
    std::string TensorShape::str() const {
        if (!initialized_) {
            return "[uninitialized]";
        }
        if (dims_.empty()) {
            return "[]";
        }
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

    // ============= Tensor String Representation =============
    std::string Tensor::str() const {
        std::ostringstream oss;
        oss << "Tensor(";
        if (!initialized_) {
            oss << "uninitialized";
        } else {
            oss << "shape=" << shape_.str();
            oss << ", device=" << device_name(device_);
            oss << ", dtype=" << dtype_name(dtype_);
            if (owns_memory_) {
                oss << ", owned";
            } else {
                oss << ", view";
            }
        }
        oss << ")";
        return oss.str();
    }

    // ============= Debug Functions =============
    void Tensor::log_info(const std::string& name) const {
        const std::string& prefix = name.empty() ? "Tensor" : name;

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

    Tensor& Tensor::fill_(float value) {
        if (!is_valid() || numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            std::vector<float> temp(numel(), value);
            CHECK_CUDA(cudaMemcpy(data_, temp.data(), bytes(), cudaMemcpyHostToDevice));
        } else {
            float* data = ptr<float>();
            std::fill(data, data + numel(), value);
        }

        return *this;
    }

    Tensor& Tensor::copy_from(const Tensor& other) {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for copy_from");
            return *this;
        }

        if (shape_ != other.shape_) {
            LOG_ERROR("Shape mismatch in copy_from: {} vs {}", shape_.str(), other.shape_.str());
            return *this;
        }

        if (dtype_ != other.dtype_) {
            LOG_ERROR("Dtype mismatch in copy_from");
            return *this;
        }

        if (numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA && other.device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(data_, other.data_, bytes(), cudaMemcpyDeviceToDevice));
        } else if (device_ == Device::CUDA && other.device_ == Device::CPU) {
            CHECK_CUDA(cudaMemcpy(data_, other.data_, bytes(), cudaMemcpyHostToDevice));
        } else if (device_ == Device::CPU && other.device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(data_, other.data_, bytes(), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(data_, other.data_, bytes());
        }

        return *this;
    }

    std::expected<Tensor, std::string> Tensor::try_reshape(TensorShape shape) const {
        if (!is_valid()) {
            return std::unexpected("Tensor is not valid");
        }

        if (shape.elements() != numel()) {
            return std::unexpected("Shape mismatch: new shape has different number of elements");
        }

        return reshape(shape);
    }

    std::vector<Tensor> Tensor::split_batch(const Tensor& tensor, size_t batch_size) {
        std::vector<Tensor> batches;

        if (!tensor.is_valid() || tensor.shape().rank() == 0) {
            return batches;
        }

        size_t total_size = tensor.shape()[0];
        size_t num_batches = (total_size + batch_size - 1) / batch_size;

        for (size_t i = 0; i < num_batches; ++i) {
            size_t start = i * batch_size;
            size_t end = std::min(start + batch_size, total_size);
            batches.push_back(tensor.slice(0, start, end));
        }

        return batches;
    }

    std::vector<float> Tensor::debug_values(size_t max_values) const {
        std::vector<float> values;

        if (!is_valid() || dtype_ != DataType::Float32 || numel() == 0) {
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
        if (dtype_ != DataType::Float32 || !is_valid()) {
            LOG_ERROR("to_vector only supports valid float32 tensors");
            return {};
        }

        if (numel() == 0) {
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
          tensor_info_(t ? t->str() : "") {}

    // ============= TensorBuilder Implementation =============
    TensorBuilder& TensorBuilder::with_shape(TensorShape shape) {
        shape_ = shape;
        return *this;
    }

    TensorBuilder& TensorBuilder::on_device(Device device) {
        device_ = device;
        return *this;
    }

    TensorBuilder& TensorBuilder::with_dtype(DataType dtype) {
        dtype_ = dtype;
        return *this;
    }

    TensorBuilder& TensorBuilder::filled_with(float value) {
        fill_value_ = value;
        return *this;
    }

    TensorBuilder& TensorBuilder::ensure_finite() {
        check_finite_ = true;
        return *this;
    }

    Tensor TensorBuilder::build() {
        Tensor t;
        if (fill_value_.has_value()) {
            t = Tensor::full(shape_, *fill_value_, device_, dtype_);
        } else {
            t = Tensor::empty(shape_, device_, dtype_);
        }

        if (check_finite_) {
            t.assert_finite();
        }

        return t;
    }

} // namespace gs