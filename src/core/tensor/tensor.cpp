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
          data_owner_(nullptr),  // Non-owning
          shape_(shape),
          device_(device),
          dtype_(dtype),
          is_view_(true),  // This is a view
          id_(next_id_++) {

        if (profiling_enabled_) {
            LOG_DEBUG("Created tensor #{} (non-owning view): shape={}, device={}, dtype={}",
                      id_, shape_.str(), device_name(device_), dtype_name(dtype_));
        }
    }

    // ============= Copy Constructor - SHALLOW COPY (LibTorch behavior) =============
    Tensor::Tensor(const Tensor& other)
        : data_(other.data_),                    // Share the pointer
          data_owner_(other.data_owner_),        // Share ownership via shared_ptr!
          shape_(other.shape_),
          device_(other.device_),
          dtype_(other.dtype_),
          is_view_(other.is_view_),
          id_(next_id_++) {

        if (profiling_enabled_) {
            LOG_DEBUG("Shallow copy: tensor #{} from #{}: shape={}, device={}, dtype={}, refcount={}",
                      id_, other.id_, shape_.str(), device_name(device_), dtype_name(dtype_),
                      data_owner_ ? data_owner_.use_count() : 0);
        }
    }

    // ============= Copy Assignment - SHALLOW COPY (LibTorch behavior) =============
    Tensor& Tensor::operator=(const Tensor& other) {
        if (this == &other) {
            return *this;
        }

        // Shallow copy - share data via shared_ptr
        data_ = other.data_;
        data_owner_ = other.data_owner_;  // shared_ptr handles refcounting automatically
        shape_ = other.shape_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        is_view_ = other.is_view_;
        id_ = next_id_++;

        if (profiling_enabled_) {
            LOG_DEBUG("Shallow assign: tensor #{} from #{}: shape={}, device={}, dtype={}, refcount={}",
                      id_, other.id_, shape_.str(), device_name(device_), dtype_name(dtype_),
                      data_owner_ ? data_owner_.use_count() : 0);
        }

        return *this;
    }

    // ============= Move Constructor =============
    Tensor::Tensor(Tensor&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          data_owner_(std::move(other.data_owner_)),
          shape_(std::move(other.shape_)),
          device_(other.device_),
          dtype_(other.dtype_),
          is_view_(std::exchange(other.is_view_, false)),
          id_(other.id_) {

        if (profiling_enabled_) {
            LOG_DEBUG("Move constructed: tensor #{} (moved-from is now invalid)", id_);
        }
    }

    // ============= Move Assignment =============
    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            data_ = std::exchange(other.data_, nullptr);
            data_owner_ = std::move(other.data_owner_);
            shape_ = std::move(other.shape_);
            device_ = other.device_;
            dtype_ = other.dtype_;
            is_view_ = std::exchange(other.is_view_, false);
            id_ = other.id_;

            if (profiling_enabled_) {
                LOG_DEBUG("Move assigned: tensor #{} (moved-from is now invalid)", id_);
            }
        }
        return *this;
    }

    // ============= Destructor =============
    Tensor::~Tensor() {
        if (data_owner_ && profiling_enabled_) {
            LOG_DEBUG("Destroying tensor #{}: shape={}, device={}, refcount={}",
                      id_, shape_.str(), device_name(device_), data_owner_.use_count());
        }
        // shared_ptr automatically handles memory cleanup when refcount reaches 0
    }

    // ============= Deep Copy (explicit) =============
    Tensor Tensor::clone() const {
        if (!is_valid()) {
            LOG_ERROR("Cannot clone invalid tensor");
            return Tensor();
        }

        if (numel() == 0) {
            // Return empty tensor with same shape and properties
            return empty(shape_, device_, dtype_);
        }

        // Create new tensor with same properties
        auto result = empty(shape_, device_, dtype_);

        // Copy data
        size_t bytes = this->bytes();

        if (device_ == Device::CUDA) {
            cudaError_t err = cudaMemcpy(result.data_, data_, bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                LOG_ERROR("CUDA memcpy failed in clone(): {}", cudaGetErrorString(err));
                return Tensor();
            }
        } else {
            std::memcpy(result.data_, data_, bytes);
        }

        if (profiling_enabled_) {
            LOG_DEBUG("Deep clone: tensor #{} from #{}: copied {} bytes",
                      result.id_, id_, bytes);
        }

        return result;
    }

    // ============= Contiguous (only copies if view) =============
    Tensor Tensor::contiguous() const {
        if (!is_view_) {
            // Already contiguous and not a view - return shallow copy
            return *this;
        }
        // Is a view - need to make a contiguous deep copy
        return clone();
    }

    // ============= Device Transfer =============
    Tensor Tensor::to(Device device) const {
        if (!is_valid()) {
            LOG_ERROR("Cannot transfer invalid tensor to device");
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

    template<typename SrcT, typename DstT>
  static Tensor convert_dtype_impl(const Tensor& src, DataType dst_dtype,
                                    const std::function<DstT(SrcT)>& converter) {
        auto result = Tensor::empty(src.shape(), src.device(), dst_dtype);
        if (src.numel() == 0) return result;

        if (src.device() == Device::CUDA) {
            auto cpu_src = src.to(Device::CPU);
            const SrcT* src_data = cpu_src.ptr<SrcT>();
            std::vector<DstT> temp(src.numel());
            for (size_t i = 0; i < src.numel(); ++i) {
                temp[i] = converter(src_data[i]);
            }
            cudaMemcpy(result.raw_ptr(), temp.data(),
                       src.numel() * sizeof(DstT), cudaMemcpyHostToDevice);
        } else {
            const SrcT* src_data = src.ptr<SrcT>();
            DstT* dst_data = result.ptr<DstT>();
            for (size_t i = 0; i < src.numel(); ++i) {
                dst_data[i] = converter(src_data[i]);
            }
        }
        return result;
    }


    // ============= Type Conversion =============
    Tensor Tensor::to(DataType dtype) const {
        if (!is_valid()) {
            LOG_ERROR("Cannot convert invalid tensor to different dtype");
            return Tensor();
        }

        if (dtype_ == dtype) {
            return clone();
        }

        // ========== KEPT AS-IS: Bool <-> Float32 (use CUDA kernels) ==========
        if (dtype_ == DataType::Bool && dtype == DataType::Float32) {
            auto result = empty(shape_, device_, DataType::Float32);
            if (numel() == 0) return result;

            if (device_ == Device::CUDA) {
                tensor_ops::launch_bool_to_float(ptr<unsigned char>(), result.ptr<float>(), numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const unsigned char* src = ptr<unsigned char>();
                float* dst = result.ptr<float>();
                for (size_t i = 0; i < numel(); ++i) dst[i] = src[i] ? 1.0f : 0.0f;
            }
            return result;
        }

        if (dtype_ == DataType::Float32 && dtype == DataType::Bool) {
            auto result = empty(shape_, device_, DataType::Bool);
            if (numel() == 0) return result;

            if (device_ == Device::CUDA) {
                tensor_ops::launch_float_to_bool(ptr<float>(), result.ptr<unsigned char>(), numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const float* src = ptr<float>();
                unsigned char* dst = result.ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) dst[i] = (src[i] != 0.0f) ? 1 : 0;
            }
            return result;
        }

        // ========== KEPT AS-IS: Float32 <-> Int32 (use CUDA kernels) ==========
        if (dtype_ == DataType::Float32 && dtype == DataType::Int32) {
            auto result = empty(shape_, device_, DataType::Int32);
            if (numel() == 0) return result;

            if (device_ == Device::CUDA) {
                tensor_ops::launch_float_to_int(ptr<float>(), result.ptr<int>(), numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const float* src = ptr<float>();
                int* dst = result.ptr<int>();
                for (size_t i = 0; i < numel(); ++i) dst[i] = static_cast<int>(src[i]);
            }
            return result;
        }

        if (dtype_ == DataType::Int32 && dtype == DataType::Float32) {
            auto result = empty(shape_, device_, DataType::Float32);
            if (numel() == 0) return result;

            if (device_ == Device::CUDA) {
                tensor_ops::launch_int_to_float(ptr<int>(), result.ptr<float>(), numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const int* src = ptr<int>();
                float* dst = result.ptr<float>();
                for (size_t i = 0; i < numel(); ++i) dst[i] = static_cast<float>(src[i]);
            }
            return result;
        }

        // ========== COMPACT: UInt8 Conversions (8 conversions in ~40 lines) ==========
        if (dtype_ == DataType::Float32 && dtype == DataType::UInt8) {
            return convert_dtype_impl<float, uint8_t>(*this, DataType::UInt8,
                [](float v) { return static_cast<uint8_t>(std::round(std::clamp(v, 0.0f, 255.0f))); });
        }

        if (dtype_ == DataType::UInt8 && dtype == DataType::Float32) {
            return convert_dtype_impl<uint8_t, float>(*this, DataType::Float32,
                [](uint8_t v) { return static_cast<float>(v); });
        }

        if (dtype_ == DataType::Int32 && dtype == DataType::UInt8) {
            return convert_dtype_impl<int, uint8_t>(*this, DataType::UInt8,
                [](int v) { return static_cast<uint8_t>(std::clamp(v, 0, 255)); });
        }

        if (dtype_ == DataType::UInt8 && dtype == DataType::Int32) {
            return convert_dtype_impl<uint8_t, int>(*this, DataType::Int32,
                [](uint8_t v) { return static_cast<int>(v); });
        }

        if (dtype_ == DataType::Bool && dtype == DataType::UInt8) {
            return convert_dtype_impl<unsigned char, uint8_t>(*this, DataType::UInt8,
                [](unsigned char v) { return v ? 1 : 0; });
        }

        if (dtype_ == DataType::UInt8 && dtype == DataType::Bool) {
            return convert_dtype_impl<uint8_t, unsigned char>(*this, DataType::Bool,
                [](uint8_t v) { return v != 0 ? 1 : 0; });
        }

        if (dtype_ == DataType::Int64 && dtype == DataType::UInt8) {
            return convert_dtype_impl<int64_t, uint8_t>(*this, DataType::UInt8,
                [](int64_t v) { return static_cast<uint8_t>(std::clamp(v, static_cast<int64_t>(0), static_cast<int64_t>(255))); });
        }

        if (dtype_ == DataType::UInt8 && dtype == DataType::Int64) {
            return convert_dtype_impl<uint8_t, int64_t>(*this, DataType::Int64,
                [](uint8_t v) { return static_cast<int64_t>(v); });
        }

        // ========== COMPACT: Int64 Conversions (4 conversions in ~20 lines) ==========
        if (dtype_ == DataType::Int64 && dtype == DataType::Float32) {
            return convert_dtype_impl<int64_t, float>(*this, DataType::Float32,
                [](int64_t v) { return static_cast<float>(v); });
        }

        if (dtype_ == DataType::Float32 && dtype == DataType::Int64) {
            return convert_dtype_impl<float, int64_t>(*this, DataType::Int64,
                [](float v) { return static_cast<int64_t>(v); });
        }

        if (dtype_ == DataType::Int32 && dtype == DataType::Int64) {
            return convert_dtype_impl<int, int64_t>(*this, DataType::Int64,
                [](int v) { return static_cast<int64_t>(v); });
        }

        if (dtype_ == DataType::Int64 && dtype == DataType::Int32) {
            return convert_dtype_impl<int64_t, int>(*this, DataType::Int32,
                [](int64_t v) { return static_cast<int>(v); });
        }

        LOG_ERROR("Type conversion from {} to {} not implemented",
                  dtype_name(dtype_), dtype_name(dtype));
        return Tensor();
    }


    // ============= In-place Operations =============

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

    // ============= Shape Operations =============

    Tensor Tensor::cat(const Tensor& other, int dim) const {
        std::vector<Tensor> tensors;
        tensors.push_back(clone());
        tensors.push_back(other.clone());
        return Tensor::cat(tensors, dim);
    }

    // ============= Broadcasting =============

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
        if (!is_valid()) {
            LOG_ERROR("Cannot normalize invalid tensor");
            return Tensor();
        }

        if (dim == -1) {
            auto m = mean();
            auto s = std({}, false, false).add(eps);
            return sub(m).div(s);
        }
        std::vector<int> axes = {dim};
        auto m = mean(axes, true);
        auto s = std(axes, true, false).add(eps);
        return sub(m).div(s);
    }

    Tensor Tensor::logit(float eps) const {
        if (!is_valid()) {
            LOG_ERROR("Cannot compute logit of invalid tensor");
            return Tensor();
        }

        auto x_clamped = clamp(eps, 1.0f - eps);
        auto one_minus_x = full(shape_, 1.0f, device_, dtype_).sub(x_clamped);
        return x_clamped.div(one_minus_x).log();
    }

    // ============= Bitwise Operations =============

    Tensor Tensor::operator~() const {
        if (!is_valid()) {
            LOG_ERROR("Bitwise NOT on invalid tensor");
            return Tensor();
        }

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

    Tensor Tensor::operator|(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Bitwise OR on invalid tensor");
            return Tensor();
        }

        if (dtype_ != DataType::Bool || other.dtype() != DataType::Bool) {
            LOG_ERROR("Bitwise OR only works on boolean tensors");
            return Tensor();
        }

        return binary_op_impl(other, BinaryOp::BitwiseOr);
    }

    // ============= Clamp Operations =============

    Tensor& Tensor::clamp_(float min_val, float max_val) {
        if (!is_valid() || numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            if (dtype_ == DataType::Float32) {
                tensor_ops::launch_clamp_scalar(ptr<float>(), min_val, max_val, numel(), 0);
                cudaDeviceSynchronize();
            } else if (dtype_ == DataType::Int32) {
                tensor_ops::launch_clamp_scalar_int(ptr<int>(),
                                                    static_cast<int>(min_val),
                                                    static_cast<int>(max_val),
                                                    numel(), 0);
                cudaDeviceSynchronize();
            }
        } else {
            if (dtype_ == DataType::Float32) {
                float* data = ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    if (!std::isnan(data[i])) {
                        data[i] = std::clamp(data[i], min_val, max_val);
                    }
                }
            } else if (dtype_ == DataType::Int32) {
                int* data = ptr<int>();
                int min_int = static_cast<int>(min_val);
                int max_int = static_cast<int>(max_val);
                for (size_t i = 0; i < numel(); ++i) {
                    data[i] = std::clamp(data[i], min_int, max_int);
                }
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
            tensor_ops::launch_cumsum(result.raw_ptr(), shape_.dims().data(),
                                      shape_.rank(), dim, dtype_, nullptr);
            cudaDeviceSynchronize();
        } else {
            if (dtype_ == DataType::Float32) {
                float* data = result.ptr<float>();

                auto strides = shape_.strides();
                size_t dim_stride = strides[dim];
                size_t dim_size = shape_[dim];
                size_t total = numel();

                for (size_t idx = 0; idx < total; ++idx) {
                    size_t coord_along_dim = (idx / dim_stride) % dim_size;

                    if (coord_along_dim == 0)
                        continue;

                    data[idx] += data[idx - dim_stride];
                }
            } else if (dtype_ == DataType::Int32) {
                int* data = result.ptr<int>();

                auto strides = shape_.strides();
                size_t dim_stride = strides[dim];
                size_t dim_size = shape_[dim];
                size_t total = numel();

                for (size_t idx = 0; idx < total; ++idx) {
                    size_t coord_along_dim = (idx / dim_stride) % dim_size;
                    if (coord_along_dim == 0)
                        continue;
                    data[idx] += data[idx - dim_stride];
                }
            }
        }

        return result;
    }

    // ============= TensorShape Implementation =============

    std::string TensorShape::str() const {
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
        if (!is_valid()) {
            oss << "invalid";
        } else {
            oss << "shape=" << shape_.str();
            oss << ", device=" << device_name(device_);
            oss << ", dtype=" << dtype_name(dtype_);
            if (data_owner_) {
                oss << ", owned, refcount=" << data_owner_.use_count();
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

        if (!is_valid()) {
            std::println("  (invalid tensor)");
            return;
        }

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
        if (!is_valid()) return;

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
        if (!is_valid() || shape_.rank() != 2)
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

    // ============= Utility Functions =============

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
        if (!is_valid()) {
            LOG_ERROR("to_vector on invalid tensor");
            return {};
        }

        if (numel() == 0) {
            return {};
        }

        // Handle Bool dtype by converting to float
        if (dtype_ == DataType::Bool) {
            auto float_tensor = to(DataType::Float32);
            return float_tensor.to_vector();
        }

        // Handle Int64 dtype by converting to float
        if (dtype_ == DataType::Int64) {
            auto float_tensor = to(DataType::Float32);
            return float_tensor.to_vector();
        }

        // Handle Int32 dtype by converting to float
        if (dtype_ == DataType::Int32) {
            auto float_tensor = to(DataType::Float32);
            return float_tensor.to_vector();
        }

        // Handle UInt8 dtype by converting to float (NEW)
        if (dtype_ == DataType::UInt8) {
            auto float_tensor = to(DataType::Float32);
            return float_tensor.to_vector();
        }

        if (dtype_ != DataType::Float32) {
            LOG_ERROR("to_vector only supports float32, int32, int64, uint8 and bool tensors, got {}",
                      dtype_name(dtype_));
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


    std::vector<int64_t> Tensor::to_vector_int64() const {
        LOG_DEBUG("to_vector_int64() called");
        LOG_DEBUG("  dtype: {}", dtype_name(dtype_));
        LOG_DEBUG("  device: {}", device_name(device_));
        LOG_DEBUG("  numel: {}", numel());
        LOG_DEBUG("  is_valid: {}", is_valid());

        if (!is_valid()) {
            LOG_ERROR("to_vector_int64() on invalid tensor");
            return {};
        }

        if (dtype_ != DataType::Int64) {
            LOG_ERROR("to_vector_int64() requires Int64 tensor, got {}", dtype_name(dtype_));
            return {};
        }

        if (numel() == 0) {
            LOG_DEBUG("Empty tensor, returning empty vector");
            return {};
        }

        LOG_DEBUG("Creating result vector of size {}", numel());
        std::vector<int64_t> result(numel());

        if (device_ == Device::CUDA) {
            LOG_DEBUG("Copying from CUDA to CPU, bytes: {}", bytes());
            CHECK_CUDA(cudaMemcpy(result.data(), data_, bytes(), cudaMemcpyDeviceToHost));
            LOG_DEBUG("CUDA copy complete");
        } else {
            LOG_DEBUG("Copying from CPU memory, bytes: {}", bytes());
            std::memcpy(result.data(), data_, bytes());
            LOG_DEBUG("CPU copy complete");
        }

        LOG_DEBUG("to_vector_int64() complete, returning {} elements", result.size());
        return result;
    }

    std::vector<int> Tensor::to_vector_int() const {
        if (!is_valid()) {
            LOG_ERROR("to_vector_int on invalid tensor");
            return {};
        }

        if (numel() == 0) {
            return {};
        }

        // Handle Bool dtype by converting to int
        if (dtype_ == DataType::Bool) {
            auto int_tensor = to(DataType::Int32);
            return int_tensor.to_vector_int();
        }

        if (dtype_ != DataType::Int32) {
            LOG_ERROR("to_vector_int only supports int32 and bool tensors, got {}",
                      dtype_name(dtype_));
            return {};
        }

        std::vector<int> result(numel());

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.data(), data_, bytes(), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(result.data(), data_, bytes());
        }

        return result;
    }

    std::vector<bool> Tensor::to_vector_bool() const {
        if (!is_valid()) {
            LOG_ERROR("to_vector_bool only supports valid bool tensors");
            return {};
        }

        // Support both Bool and UInt8 dtypes (UInt8 can be used as byte array)
        if (dtype_ != DataType::Bool && dtype_ != DataType::UInt8) {
            LOG_ERROR("to_vector_bool only supports bool and uint8 tensors, got {}",
                      dtype_name(dtype_));
            return {};
        }

        if (numel() == 0) {
            return {};
        }

        std::vector<bool> result(numel());

        if (device_ == Device::CUDA) {
            std::vector<unsigned char> temp(numel());
            CHECK_CUDA(cudaMemcpy(temp.data(), data_, bytes(), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < numel(); ++i) {
                result[i] = temp[i] != 0;
            }
        } else {
            const unsigned char* data = ptr<unsigned char>();
            for (size_t i = 0; i < numel(); ++i) {
                result[i] = data[i] != 0;
            }
        }

        return result;
    }

    std::vector<uint8_t> Tensor::to_vector_uint8() const {
        if (!is_valid()) {
            LOG_ERROR("to_vector_uint8 on invalid tensor");
            return {};
        }

        if (numel() == 0) {
            return {};
        }

        // Handle UInt8 dtype directly
        if (dtype_ == DataType::UInt8) {
            std::vector<uint8_t> result(numel());

            if (device_ == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(result.data(), data_, bytes(), cudaMemcpyDeviceToHost));
            } else {
                std::memcpy(result.data(), data_, bytes());
            }

            return result;
        }

        // Handle Bool dtype (convert to uint8)
        if (dtype_ == DataType::Bool) {
            std::vector<uint8_t> result(numel());

            if (device_ == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(result.data(), data_, bytes(), cudaMemcpyDeviceToHost));
            } else {
                const unsigned char* src = ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    result[i] = src[i];
                }
            }

            return result;
        }

        // For other types, log error
        LOG_ERROR("to_vector_uint8 only supports uint8 and bool tensors directly, got {}. "
                  "Convert to UInt8 first using .to(DataType::UInt8)",
                  dtype_name(dtype_));
        return {};
    }

    void Tensor::dump_diagnostic(const std::string& filename) const {
        std::ofstream file(filename);
        std::print(file, "=== Tensor Diagnostic Dump ===\n");
        std::print(file, "Info: {}\n", str());
        std::print(file, "Memory address: {}\n", data_);

        if (is_valid()) {
            std::print(file, "Bytes: {}\n", bytes());

            if (device_ == Device::CPU || numel() < 10000) {
                auto values = to_vector();
                std::print(file, "Values ({} total):\n", values.size());
                for (size_t i = 0; i < std::min(size_t(1000), values.size()); ++i) {
                    std::print(file, "[{}]: {}\n", i, values[i]);
                }
            }
        } else {
            std::print(file, "Tensor is invalid\n");
        }

        file.close();
        LOG_INFO("Diagnostic dump saved to {}", filename);
    }

    // ============= Validation & Assertions =============

    Tensor& Tensor::assert_shape(TensorShape expected, const std::string& msg) {
        if (!is_valid()) {
            std::string error_msg = "Cannot assert shape on invalid tensor";
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }

        if (shape_ != expected) {
            std::string error_msg = msg.empty() ? "Shape assertion failed: expected " + expected.str() + " but got " + shape_.str() : msg;
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }
        return *this;
    }

    Tensor& Tensor::assert_device(Device expected) {
        if (!is_valid()) {
            std::string error_msg = "Cannot assert device on invalid tensor";
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }

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
        if (!is_valid()) {
            std::string error_msg = "Cannot assert dtype on invalid tensor";
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }

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
        if (!is_valid()) {
            std::string error_msg = "Cannot assert finite on invalid tensor";
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg, this);
        }

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

        if (shape_ != other.shape_ || dtype_ != other.dtype_) {
            return false;
        }

        if (numel() == 0) {
            return true;
        }

        const float* a_data = nullptr;
        const float* b_data = nullptr;

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

        if (!a_data || !b_data) {
            return false;
        }

        for (size_t i = 0; i < numel(); ++i) {
            float diff = std::abs(a_data[i] - b_data[i]);
            float tol = atol + rtol * std::abs(b_data[i]);
            if (diff > tol) {
                return false;
            }
        }

        return true;
    }

    // ============= Error Classes =============

    TensorError::TensorError(const std::string& msg, const Tensor* t)
        : std::runtime_error(msg),
          tensor_info_(t ? t->str() : "") {}

#undef CHECK_CUDA

} // namespace gs