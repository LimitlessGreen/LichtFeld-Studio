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
      data_owner_(nullptr),  // Non-owning view
      shape_(shape),
      device_(device),
      dtype_(dtype),
      initialized_(true),
      id_(next_id_++) {

        if (profiling_enabled_) {
            LOG_DEBUG("Created tensor #{} (view): shape={}, device={}, dtype={}",
                      id_, shape_.str(), device_name(device_), dtype_name(dtype_));
        }
    }

    Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      data_owner_(std::move(other.data_owner_)),
      shape_(other.shape_),
      device_(other.device_),
      dtype_(other.dtype_),
      initialized_(other.initialized_),
      is_view_(other.is_view_),
      id_(other.id_) {

        other.data_ = nullptr;
        other.initialized_ = false;
        other.is_view_ = false;
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            // Resources are automatically cleaned up by shared_ptr
            data_ = other.data_;
            data_owner_ = std::move(other.data_owner_);
            shape_ = other.shape_;
            device_ = other.device_;
            dtype_ = other.dtype_;
            initialized_ = other.initialized_;
            is_view_ = other.is_view_;
            id_ = other.id_;

            // Reset other
            other.data_ = nullptr;
            other.initialized_ = false;
            other.is_view_ = false;
        }
        return *this;
    }

    Tensor::~Tensor() {
        if (data_owner_ && profiling_enabled_) {
            LOG_DEBUG("Destroying tensor #{}: shape={}, device={}, refcount={}",
                      id_, shape_.str(), device_name(device_), data_owner_.use_count());
        }
        // shared_ptr automatically handles deallocation
    }

    Tensor Tensor::clone() const {
        if (!is_valid()) {
            return Tensor();
        }

        Tensor t = empty(shape_, device_, dtype_);
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

        // Support Float32 -> Int32 conversion
        if (dtype_ == DataType::Float32 && dtype == DataType::Int32) {
            auto result = empty(shape_, device_, DataType::Int32);

            if (numel() == 0) {
                return result;
            }

            if (device_ == Device::CUDA) {
                tensor_ops::launch_float_to_int(ptr<float>(), result.ptr<int>(),
                                               numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const float* src = ptr<float>();
                int* dst = result.ptr<int>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = static_cast<int>(src[i]);
                }
            }

            return result;
        }

        // Support Int32 -> Float32 conversion
        if (dtype_ == DataType::Int32 && dtype == DataType::Float32) {
            auto result = empty(shape_, device_, DataType::Float32);

            if (numel() == 0) {
                return result;
            }

            if (device_ == Device::CUDA) {
                tensor_ops::launch_int_to_float(ptr<int>(), result.ptr<float>(),
                                               numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const int* src = ptr<int>();
                float* dst = result.ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = static_cast<float>(src[i]);
                }
            }

            return result;
        }

        LOG_ERROR("Type conversion from {} to {} not implemented",
                  dtype_name(dtype_), dtype_name(dtype));
        return Tensor();
    }

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

    Tensor Tensor::cat(const Tensor& other, int dim) const {
        std::vector<Tensor> tensors;
        tensors.push_back(clone());
        tensors.push_back(other.clone());
        return tensor::cat(std::move(tensors), dim);
    }

    Tensor Tensor::broadcast_to(const TensorShape& target_shape) const {
        return gs::broadcast_to(*this, target_shape);
    }

    bool Tensor::can_broadcast_to(const TensorShape& target) const {
        auto result = broadcast::shape(shape_.dims(), target.dims());
        return !result.empty() && result == target.dims();
    }

    TensorShape Tensor::broadcast_shape(const TensorShape& other) const {
        auto result = broadcast::shape(shape_.dims(), other.dims());
        return result.empty() ? TensorShape() : TensorShape(result);
    }

    // ============= Special operations =============
    Tensor Tensor::normalize(int dim, float eps) const {
        if (dim == -1) {
            // Normalize entire tensor
            auto m = mean();
            auto s = std().add(eps);
            return sub(m).div(s);
        }
        // Per-dimension would use reduce with axes
        std::vector<int> axes = {dim};
        auto m = mean(axes, true);
        auto s = std(axes, true).add(eps);
        return sub(m).div(s);
    }

    Tensor Tensor::logit(float eps) const {
        if (!is_valid()) {
            return Tensor();
        }

        // logit(x) = log(x / (1 - x))
        // But we need to clamp x to [eps, 1-eps] first
        auto x_clamped = clamp(eps, 1.0f - eps);
        auto one_minus_x = full(shape_, 1.0f, device_, dtype_).sub(x_clamped);
        return x_clamped.div(one_minus_x).log();
    }

    // ============= NEW: Bitwise NOT =============
    Tensor Tensor::operator~() const {
        if (dtype_ != DataType::Bool) {
            LOG_ERROR("Bitwise NOT only works on boolean tensors");
            return Tensor();
        }

        auto result = empty(shape_, device_, DataType::Bool);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_unary_op(data_, result.data_, numel(),
                                       UnaryOp::LogicalNot, dtype_, nullptr);
            cudaDeviceSynchronize();
        } else {
            const unsigned char* src = ptr<unsigned char>();
            unsigned char* dst = result.ptr<unsigned char>();
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = !src[i];
            }
        }

        return result;
    }

    // ============= NEW: Bitwise OR =============
    Tensor Tensor::operator|(const Tensor& other) const {
        if (dtype_ != DataType::Bool || other.dtype() != DataType::Bool) {
            LOG_ERROR("Bitwise OR only works on boolean tensors");
            return Tensor();
        }

        return binary_op_impl(other, BinaryOp::BitwiseOr);
    }

    Tensor& Tensor::clamp_(float min_val, float max_val) {
        if (!is_valid() || dtype_ != DataType::Float32) {
            return *this;
        }

        if (numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_clamp_scalar(ptr<float>(), min_val, max_val, numel(), nullptr);
            cudaDeviceSynchronize();
        } else {
            float* data = ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] = std::clamp(data[i], min_val, max_val);
            }
        }

        return *this;
    }

    Tensor& Tensor::clamp_min_(float min) {
        return clamp_(min, std::numeric_limits<float>::max());
    }

    Tensor& Tensor::clamp_max_(float max) {
        return clamp_(std::numeric_limits<float>::lowest(), max);
    }
    // ============= Cumulative sum =============
    Tensor Tensor::cumsum(int dim) const {
        if (!is_valid()) {
            LOG_ERROR("cumsum on invalid tensor");
            return Tensor();
        }

        dim = resolve_dim(dim);
        if (dim < 0 || dim >= static_cast<int>(shape_.rank())) {
            LOG_ERROR("Invalid dimension for cumsum: {}", dim);
            return Tensor();
        }

        auto result = clone();

        if (device_ == Device::CUDA) {
            // Use CUDA kernel for cumulative sum
            tensor_ops::launch_cumsum(result.raw_ptr(), shape_.dims().data(),
                                     shape_.rank(), dim, dtype_, nullptr);
            cudaDeviceSynchronize();
        } else {
            // CPU implementation
            if (dtype_ == DataType::Float32) {
                float* data = result.ptr<float>();

                // Calculate stride for the dimension
                size_t stride = 1;
                for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                    stride *= shape_[i];
                }

                size_t outer_size = 1;
                for (size_t i = 0; i < static_cast<size_t>(dim); ++i) {
                    outer_size *= shape_[i];
                }

                size_t dim_size = shape_[dim];

                // Cumulative sum along dimension
                for (size_t o = 0; o < outer_size; ++o) {
                    for (size_t s = 0; s < stride; ++s) {
                        size_t base = o * dim_size * stride + s;
                        for (size_t d = 1; d < dim_size; ++d) {
                            data[base + d * stride] += data[base + (d-1) * stride];
                        }
                    }
                }
            } else if (dtype_ == DataType::Int32) {
                int* data = result.ptr<int>();

                size_t stride = 1;
                for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                    stride *= shape_[i];
                }

                size_t outer_size = 1;
                for (size_t i = 0; i < static_cast<size_t>(dim); ++i) {
                    outer_size *= shape_[i];
                }

                size_t dim_size = shape_[dim];

                for (size_t o = 0; o < outer_size; ++o) {
                    for (size_t s = 0; s < stride; ++s) {
                        size_t base = o * dim_size * stride + s;
                        for (size_t d = 1; d < dim_size; ++d) {
                            data[base + d * stride] += data[base + (d-1) * stride];
                        }
                    }
                }
            }
        }

        return result;
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
            if (data_owner_) {
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
            first_slice.squeeze().print_2d(max_per_dim);
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

    std::optional<Tensor> Tensor::try_reshape(TensorShape shape) const {
        if (!is_valid()) {
            return std::nullopt;
        }

        if (shape.elements() != numel()) {
            return std::nullopt;
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

    float Tensor::item() const {
        if (!is_valid() || numel() != 1) {
            LOG_ERROR("item() requires a valid single-element tensor");
            return 0.0f;
        }

        if (dtype_ != DataType::Float32) {
            LOG_ERROR("item() only supports float32 tensors");
            return 0.0f;
        }

        float value = 0.0f;
        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(&value, data_, sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            value = *static_cast<const float*>(data_);
        }

        return value;
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

    // ============= Validation & Assertions =============
    Tensor& Tensor::assert_shape(TensorShape expected, const std::string& msg) {
        if (shape_ != expected) {
            std::string error_msg = msg.empty() ?
                "Shape assertion failed: expected " + expected.str() + " but got " + shape_.str() :
                msg;
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    Tensor& Tensor::assert_device(Device expected) {
        if (device_ != expected) {
            std::string error_msg = "Device assertion failed: expected " +
                std::string(device_name(expected)) + " but got " +
                std::string(device_name(device_));
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    Tensor& Tensor::assert_dtype(DataType expected) {
        if (dtype_ != expected) {
            std::string error_msg = "DataType assertion failed: expected " +
                std::string(dtype_name(expected)) + " but got " +
                std::string(dtype_name(dtype_));
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    Tensor& Tensor::assert_finite() {
        if (has_nan() || has_inf()) {
            std::string error_msg = "Tensor contains NaN or Inf values";
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    // ============= Comparison Utilities =============
    bool Tensor::has_nan() const {
        if (!is_valid() || numel() == 0) {
            return false;
        }

        auto values = to_vector();
        return std::any_of(values.begin(), values.end(),
                          [](float x) { return std::isnan(x); });
    }

    bool Tensor::has_inf() const {
        if (!is_valid() || numel() == 0) {
            return false;
        }

        auto values = to_vector();
        return std::any_of(values.begin(), values.end(),
                          [](float x) { return std::isinf(x); });
    }

bool Tensor::all_close(const Tensor& other, float rtol, float atol) const {
    if (!is_valid() || !other.is_valid()) {
        return false;
    }

    // Check shape and dtype match (but not device - we handle that below)
    if (shape_ != other.shape_ || dtype_ != other.dtype_) {
        return false;
    }

    // Handle empty tensors
    if (numel() == 0) {
        return true;  // Two empty tensors with same shape are considered close
    }

    // We need to get the data to CPU for comparison
    const float* a_data = nullptr;
    const float* b_data = nullptr;

    // Create temporary CPU tensors if needed
    Tensor a_temp, b_temp;

    if (device_ == Device::CUDA) {
        a_temp = to(Device::CPU);
        a_data = a_temp.ptr<float>();
    } else {
        a_data = ptr<float>();
    }

    if (other.device() == Device::CUDA) {
        b_temp = other.to(Device::CPU);
        b_data = b_temp.ptr<float>();
    } else {
        b_data = other.ptr<float>();
    }

    // Check if pointers are valid
    if (!a_data || !b_data) {
        return false;
    }

    // Compare values with tolerance
    for (size_t i = 0; i < numel(); ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        float tol = atol + rtol * std::abs(b_data[i]);
        if (diff > tol) {
            return false;
        }
    }

    return true;
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