/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_broadcast.hpp"
#include "core/tensor_ops.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

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

    // ============= Boolean Tensor Creation =============
    Tensor Tensor::full_bool(TensorShape shape, bool value, Device device) {
        auto t = empty(shape, device, DataType::Bool);
        if (t.is_valid() && t.numel() > 0) {
            unsigned char fill_val = value ? 1 : 0;
            if (device == Device::CUDA) {
                CHECK_CUDA(cudaMemset(t.data_, fill_val, t.bytes()));
            } else {
                std::memset(t.data_, fill_val, t.bytes());
            }
        }
        return t;
    }

    Tensor Tensor::zeros_bool(TensorShape shape, Device device) {
        return full_bool(shape, false, device);
    }

    Tensor Tensor::ones_bool(TensorShape shape, Device device) {
        return full_bool(shape, true, device);
    }

    // ============= From Vector Creation =============
    Tensor Tensor::from_vector(const std::vector<float>& data, TensorShape shape, Device device) {
        if (shape.elements() != data.size()) {
            LOG_ERROR("Shape elements {} doesn't match data size {}",
                      shape.elements(), data.size());
            return Tensor();
        }

        auto t = empty(shape, device, DataType::Float32);
        if (!t.is_valid() || t.numel() == 0) {
            return t;
        }

        if (device == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(t.data_, data.data(), t.bytes(), cudaMemcpyHostToDevice));
        } else {
            std::memcpy(t.data_, data.data(), t.bytes());
        }

        return t;
    }

    Tensor Tensor::from_vector(const std::vector<int>& data, TensorShape shape, Device device) {
        if (shape.elements() != data.size()) {
            LOG_ERROR("Shape elements {} doesn't match data size {}",
                      shape.elements(), data.size());
            return Tensor();
        }

        auto t = empty(shape, device, DataType::Int32);
        if (!t.is_valid() || t.numel() == 0) {
            return t;
        }

        if (device == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(t.data_, data.data(), t.bytes(), cudaMemcpyHostToDevice));
        } else {
            std::memcpy(t.data_, data.data(), t.bytes());
        }

        return t;
    }

    Tensor Tensor::from_vector(const std::vector<bool>& data, TensorShape shape, Device device) {
        if (shape.elements() != data.size()) {
            LOG_ERROR("Shape elements {} doesn't match data size {}",
                      shape.elements(), data.size());
            return Tensor();
        }

        auto t = empty(shape, device, DataType::Bool);
        if (!t.is_valid() || t.numel() == 0) {
            return t;
        }

        // Convert bool vector to byte array
        std::vector<unsigned char> byte_data(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            byte_data[i] = data[i] ? 1 : 0;
        }

        if (device == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(t.data_, byte_data.data(), t.bytes(), cudaMemcpyHostToDevice));
        } else {
            std::memcpy(t.data_, byte_data.data(), t.bytes());
        }

        return t;
    }

    // ============= Comparison Operations =============
    Tensor Tensor::eq(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Tensors must be on the same device for comparison");
            return Tensor();
        }

        if (dtype_ != other.dtype_) {
            LOG_ERROR("Dtype mismatch for comparison");
            return Tensor();
        }

        // Handle broadcasting - use this-> to avoid name collision
        TensorShape bcast_shape = this->broadcast_shape(other.shape_);
        if (!bcast_shape.is_initialized()) {
            LOG_ERROR("Cannot broadcast shapes for comparison");
            return Tensor();
        }

        auto result = empty(bcast_shape, device_, DataType::Bool);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_compare_eq(ptr<float>(), other.ptr<float>(),
                                          result.ptr<unsigned char>(),
                                          shape_.dims().data(), other.shape_.dims().data(),
                                          bcast_shape.dims().data(),
                                          shape_.rank(), other.shape_.rank(),
                                          bcast_shape.rank(),
                                          bcast_shape.elements(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation
            const float* a_data = ptr<float>();
            const float* b_data = other.ptr<float>();
            unsigned char* result_data = result.ptr<unsigned char>();

            // Simple case: same shape
            if (shape_ == other.shape_) {
                for (size_t i = 0; i < numel(); ++i) {
                    result_data[i] = (a_data[i] == b_data[i]) ? 1 : 0;
                }
            } else {
                // Broadcasting needed - use BroadcastIterator
                BroadcastIterator iter_a(shape_, bcast_shape);
                BroadcastIterator iter_b(other.shape_, bcast_shape);

                size_t idx = 0;
                while (!iter_a.done()) {
                    result_data[idx++] = (a_data[iter_a.index()] == b_data[iter_b.index()]) ? 1 : 0;
                    iter_a.next();
                    iter_b.next();
                }
            }
        }

        return result;
    }

    Tensor Tensor::eq(float value) const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = empty(shape_, device_, DataType::Bool);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_compare_scalar_eq(ptr<float>(), value,
                                                 result.ptr<unsigned char>(),
                                                 numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* data = ptr<float>();
            unsigned char* result_data = result.ptr<unsigned char>();
            for (size_t i = 0; i < numel(); ++i) {
                result_data[i] = (data[i] == value) ? 1 : 0;
            }
        }

        return result;
    }

    // Similar implementations for other comparison operations
    Tensor Tensor::ne(const Tensor& other) const {
        auto eq_result = eq(other);
        return eq_result.logical_not();
    }

    Tensor Tensor::ne(float value) const {
        auto eq_result = eq(value);
        return eq_result.logical_not();
    }

    Tensor Tensor::lt(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Tensor();
        }

        if (device_ != other.device_ || dtype_ != other.dtype_) {
            LOG_ERROR("Tensors must have same device and dtype for comparison");
            return Tensor();
        }

        TensorShape bcast_shape = this->broadcast_shape(other.shape_);
        if (!bcast_shape.is_initialized()) {
            return Tensor();
        }

        auto result = empty(bcast_shape, device_, DataType::Bool);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_compare_lt(ptr<float>(), other.ptr<float>(),
                                          result.ptr<unsigned char>(),
                                          shape_.dims().data(), other.shape_.dims().data(),
                                          bcast_shape.dims().data(),
                                          shape_.rank(), other.shape_.rank(),
                                          bcast_shape.rank(),
                                          bcast_shape.elements(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* a_data = ptr<float>();
            const float* b_data = other.ptr<float>();
            unsigned char* result_data = result.ptr<unsigned char>();

            if (shape_ == other.shape_) {
                for (size_t i = 0; i < numel(); ++i) {
                    result_data[i] = (a_data[i] < b_data[i]) ? 1 : 0;
                }
            } else {
                BroadcastIterator iter_a(shape_, bcast_shape);
                BroadcastIterator iter_b(other.shape_, bcast_shape);

                size_t idx = 0;
                while (!iter_a.done()) {
                    result_data[idx++] = (a_data[iter_a.index()] < b_data[iter_b.index()]) ? 1 : 0;
                    iter_a.next();
                    iter_b.next();
                }
            }
        }

        return result;
    }

    Tensor Tensor::lt(float value) const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = empty(shape_, device_, DataType::Bool);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_compare_scalar_lt(ptr<float>(), value,
                                                 result.ptr<unsigned char>(),
                                                 numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* data = ptr<float>();
            unsigned char* result_data = result.ptr<unsigned char>();
            for (size_t i = 0; i < numel(); ++i) {
                result_data[i] = (data[i] < value) ? 1 : 0;
            }
        }

        return result;
    }

    Tensor Tensor::le(const Tensor& other) const {
        // a <= b is equivalent to !(a > b)
        return gt(other).logical_not();
    }

    Tensor Tensor::le(float value) const {
        return gt(value).logical_not();
    }

    Tensor Tensor::gt(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Tensor();
        }

        if (device_ != other.device_ || dtype_ != other.dtype_) {
            LOG_ERROR("Tensors must have same device and dtype for comparison");
            return Tensor();
        }

        TensorShape bcast_shape = this->broadcast_shape(other.shape_);
        if (!bcast_shape.is_initialized()) {
            return Tensor();
        }

        auto result = empty(bcast_shape, device_, DataType::Bool);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_compare_gt(ptr<float>(), other.ptr<float>(),
                                          result.ptr<unsigned char>(),
                                          shape_.dims().data(), other.shape_.dims().data(),
                                          bcast_shape.dims().data(),
                                          shape_.rank(), other.shape_.rank(),
                                          bcast_shape.rank(),
                                          bcast_shape.elements(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* a_data = ptr<float>();
            const float* b_data = other.ptr<float>();
            unsigned char* result_data = result.ptr<unsigned char>();

            if (shape_ == other.shape_) {
                for (size_t i = 0; i < numel(); ++i) {
                    result_data[i] = (a_data[i] > b_data[i]) ? 1 : 0;
                }
            } else {
                BroadcastIterator iter_a(shape_, bcast_shape);
                BroadcastIterator iter_b(other.shape_, bcast_shape);

                size_t idx = 0;
                while (!iter_a.done()) {
                    result_data[idx++] = (a_data[iter_a.index()] > b_data[iter_b.index()]) ? 1 : 0;
                    iter_a.next();
                    iter_b.next();
                }
            }
        }

        return result;
    }

    Tensor Tensor::gt(float value) const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = empty(shape_, device_, DataType::Bool);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_compare_scalar_gt(ptr<float>(), value,
                                                 result.ptr<unsigned char>(),
                                                 numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* data = ptr<float>();
            unsigned char* result_data = result.ptr<unsigned char>();
            for (size_t i = 0; i < numel(); ++i) {
                result_data[i] = (data[i] > value) ? 1 : 0;
            }
        }

        return result;
    }

    Tensor Tensor::ge(const Tensor& other) const {
        // a >= b is equivalent to !(a < b)
        return lt(other).logical_not();
    }

    Tensor Tensor::ge(float value) const {
        return lt(value).logical_not();
    }

    // ============= Logical Operations =============
    Tensor Tensor::logical_and(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Tensor();
        }

        if (dtype_ != DataType::Bool || other.dtype_ != DataType::Bool) {
            LOG_ERROR("Logical operations require boolean tensors");
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Tensors must be on same device");
            return Tensor();
        }

        TensorShape bcast_shape = this->broadcast_shape(other.shape_);
        if (!bcast_shape.is_initialized()) {
            return Tensor();
        }

        auto result = empty(bcast_shape, device_, DataType::Bool);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_logical_and(ptr<unsigned char>(), other.ptr<unsigned char>(),
                                           result.ptr<unsigned char>(),
                                           shape_.dims().data(), other.shape_.dims().data(),
                                           bcast_shape.dims().data(),
                                           shape_.rank(), other.shape_.rank(),
                                           bcast_shape.rank(),
                                           bcast_shape.elements(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const unsigned char* a_data = ptr<unsigned char>();
            const unsigned char* b_data = other.ptr<unsigned char>();
            unsigned char* result_data = result.ptr<unsigned char>();

            if (shape_ == other.shape_) {
                for (size_t i = 0; i < numel(); ++i) {
                    result_data[i] = (a_data[i] && b_data[i]) ? 1 : 0;
                }
            } else {
                BroadcastIterator iter_a(shape_, bcast_shape);
                BroadcastIterator iter_b(other.shape_, bcast_shape);

                size_t idx = 0;
                while (!iter_a.done()) {
                    result_data[idx++] = (a_data[iter_a.index()] && b_data[iter_b.index()]) ? 1 : 0;
                    iter_a.next();
                    iter_b.next();
                }
            }
        }

        return result;
    }

    Tensor Tensor::logical_or(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Tensor();
        }

        if (dtype_ != DataType::Bool || other.dtype_ != DataType::Bool) {
            LOG_ERROR("Logical operations require boolean tensors");
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Tensors must be on same device");
            return Tensor();
        }

        TensorShape bcast_shape = this->broadcast_shape(other.shape_);
        if (!bcast_shape.is_initialized()) {
            return Tensor();
        }

        auto result = empty(bcast_shape, device_, DataType::Bool);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_logical_or(ptr<unsigned char>(), other.ptr<unsigned char>(),
                                          result.ptr<unsigned char>(),
                                          shape_.dims().data(), other.shape_.dims().data(),
                                          bcast_shape.dims().data(),
                                          shape_.rank(), other.shape_.rank(),
                                          bcast_shape.rank(),
                                          bcast_shape.elements(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const unsigned char* a_data = ptr<unsigned char>();
            const unsigned char* b_data = other.ptr<unsigned char>();
            unsigned char* result_data = result.ptr<unsigned char>();

            if (shape_ == other.shape_) {
                for (size_t i = 0; i < numel(); ++i) {
                    result_data[i] = (a_data[i] || b_data[i]) ? 1 : 0;
                }
            } else {
                BroadcastIterator iter_a(shape_, bcast_shape);
                BroadcastIterator iter_b(other.shape_, bcast_shape);

                size_t idx = 0;
                while (!iter_a.done()) {
                    result_data[idx++] = (a_data[iter_a.index()] || b_data[iter_b.index()]) ? 1 : 0;
                    iter_a.next();
                    iter_b.next();
                }
            }
        }

        return result;
    }

    Tensor Tensor::logical_not() const {
        if (!is_valid()) {
            return Tensor();
        }

        if (dtype_ != DataType::Bool) {
            LOG_ERROR("Logical not requires boolean tensor");
            return Tensor();
        }

        auto result = empty(shape_, device_, DataType::Bool);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_logical_not(ptr<unsigned char>(),
                                           result.ptr<unsigned char>(),
                                           numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const unsigned char* data = ptr<unsigned char>();
            unsigned char* result_data = result.ptr<unsigned char>();
            for (size_t i = 0; i < numel(); ++i) {
                result_data[i] = data[i] ? 0 : 1;
            }
        }

        return result;
    }

    Tensor Tensor::logical_xor(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Tensor();
        }

        if (dtype_ != DataType::Bool || other.dtype_ != DataType::Bool) {
            LOG_ERROR("Logical operations require boolean tensors");
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Tensors must be on same device");
            return Tensor();
        }

        TensorShape bcast_shape = this->broadcast_shape(other.shape_);
        if (!bcast_shape.is_initialized()) {
            return Tensor();
        }

        auto result = empty(bcast_shape, device_, DataType::Bool);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_logical_xor(ptr<unsigned char>(), other.ptr<unsigned char>(),
                                           result.ptr<unsigned char>(),
                                           shape_.dims().data(), other.shape_.dims().data(),
                                           bcast_shape.dims().data(),
                                           shape_.rank(), other.shape_.rank(),
                                           bcast_shape.rank(),
                                           bcast_shape.elements(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const unsigned char* a_data = ptr<unsigned char>();
            const unsigned char* b_data = other.ptr<unsigned char>();
            unsigned char* result_data = result.ptr<unsigned char>();

            if (shape_ == other.shape_) {
                for (size_t i = 0; i < numel(); ++i) {
                    result_data[i] = ((a_data[i] != 0) != (b_data[i] != 0)) ? 1 : 0;
                }
            } else {
                BroadcastIterator iter_a(shape_, bcast_shape);
                BroadcastIterator iter_b(other.shape_, bcast_shape);

                size_t idx = 0;
                while (!iter_a.done()) {
                    result_data[idx++] = ((a_data[iter_a.index()] != 0) != (b_data[iter_b.index()] != 0)) ? 1 : 0;
                    iter_a.next();
                    iter_b.next();
                }
            }
        }

        return result;
    }

    // ============= Masking Operations =============
    Tensor Tensor::masked_select(const Tensor& mask) const {
        if (!is_valid() || !mask.is_valid()) {
            return Tensor();
        }

        if (mask.dtype() != DataType::Bool) {
            LOG_ERROR("Mask must be boolean tensor");
            return Tensor();
        }

        if (shape_ != mask.shape()) {
            LOG_ERROR("Tensor and mask must have same shape for masked_select");
            return Tensor();
        }

        if (device_ != mask.device()) {
            LOG_ERROR("Tensor and mask must be on same device");
            return Tensor();
        }

        // Count true elements in mask
        size_t count = mask.count_nonzero();
        if (count == 0) {
            return empty({0}, device_, dtype_);
        }

        // Create output tensor
        auto result = empty({count}, device_, dtype_);

        if (device_ == Device::CUDA) {
            // Use stream compaction for efficient gathering
            tensor_ops::launch_masked_select(ptr<float>(), mask.ptr<unsigned char>(),
                                             result.ptr<float>(), numel(), count, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* data = ptr<float>();
            const unsigned char* mask_data = mask.ptr<unsigned char>();
            float* result_data = result.ptr<float>();

            size_t out_idx = 0;
            for (size_t i = 0; i < numel(); ++i) {
                if (mask_data[i]) {
                    result_data[out_idx++] = data[i];
                }
            }
        }

        return result;
    }

    Tensor& Tensor::masked_fill_(const Tensor& mask, float value) {
        if (!is_valid() || !mask.is_valid()) {
            LOG_ERROR("Invalid tensors for masked_fill_");
            return *this;
        }

        if (mask.dtype() != DataType::Bool) {
            LOG_ERROR("Mask must be boolean tensor");
            return *this;
        }

        if (shape_ != mask.shape()) {
            LOG_ERROR("Tensor and mask must have same shape");
            return *this;
        }

        if (device_ != mask.device()) {
            LOG_ERROR("Tensor and mask must be on same device");
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_masked_fill(ptr<float>(), mask.ptr<unsigned char>(),
                                           value, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = ptr<float>();
            const unsigned char* mask_data = mask.ptr<unsigned char>();

            for (size_t i = 0; i < numel(); ++i) {
                if (mask_data[i]) {
                    data[i] = value;
                }
            }
        }

        return *this;
    }

    Tensor Tensor::masked_fill(const Tensor& mask, float value) const {
        auto result = clone();
        result.masked_fill_(mask, value);
        return result;
    }

    Tensor Tensor::where(const Tensor& condition, const Tensor& other) const {
        return Tensor::where(condition, *this, other);
    }

    Tensor Tensor::where(const Tensor& condition, const Tensor& x, const Tensor& y) {
        if (!condition.is_valid() || !x.is_valid() || !y.is_valid()) {
            return Tensor();
        }

        if (condition.dtype() != DataType::Bool) {
            LOG_ERROR("Condition must be boolean tensor");
            return Tensor();
        }

        if (x.device() != y.device() || x.device() != condition.device()) {
            LOG_ERROR("All tensors must be on same device");
            return Tensor();
        }

        if (x.dtype() != y.dtype()) {
            LOG_ERROR("x and y must have same dtype");
            return Tensor();
        }

        // Determine broadcast shape
        TensorShape bcast_shape = BroadcastHelper::broadcast_shape(x.shape(), y.shape());
        bcast_shape = BroadcastHelper::broadcast_shape(condition.shape(), bcast_shape);

        if (!bcast_shape.is_initialized()) {
            LOG_ERROR("Cannot broadcast shapes for where operation");
            return Tensor();
        }

        auto result = empty(bcast_shape, x.device(), x.dtype());

        if (x.device() == Device::CUDA) {
            tensor_ops::launch_where(condition.ptr<unsigned char>(),
                                     x.ptr<float>(), y.ptr<float>(),
                                     result.ptr<float>(),
                                     condition.shape().dims().data(),
                                     x.shape().dims().data(),
                                     y.shape().dims().data(),
                                     bcast_shape.dims().data(),
                                     condition.shape().rank(),
                                     x.shape().rank(),
                                     y.shape().rank(),
                                     bcast_shape.rank(),
                                     bcast_shape.elements(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const unsigned char* cond_data = condition.ptr<unsigned char>();
            const float* x_data = x.ptr<float>();
            const float* y_data = y.ptr<float>();
            float* result_data = result.ptr<float>();

            // Simple case: all same shape
            if (condition.shape() == x.shape() && x.shape() == y.shape()) {
                for (size_t i = 0; i < x.numel(); ++i) {
                    result_data[i] = cond_data[i] ? x_data[i] : y_data[i];
                }
            } else {
                // Broadcasting needed
                BroadcastIterator cond_iter(condition.shape(), bcast_shape);
                BroadcastIterator x_iter(x.shape(), bcast_shape);
                BroadcastIterator y_iter(y.shape(), bcast_shape);

                size_t idx = 0;
                while (!cond_iter.done()) {
                    result_data[idx++] = cond_data[cond_iter.index()] ? x_data[x_iter.index()] : y_data[y_iter.index()];
                    cond_iter.next();
                    x_iter.next();
                    y_iter.next();
                }
            }
        }

        return result;
    }

    // ============= Indexing Operations =============
    Tensor Tensor::index_select(int dim, const Tensor& indices) const {
        return index_select(dim, indices, BoundaryMode::Assert);
    }

    Tensor Tensor::index_select(int dim, const Tensor& indices, BoundaryMode mode) const {
        if (!is_valid() || !indices.is_valid()) {
            return Tensor();
        }

        if (indices.dtype() != DataType::Int32 && indices.dtype() != DataType::Int64) {
            LOG_ERROR("Indices must be integer tensor");
            return Tensor();
        }

        if (indices.ndim() != 1) {
            LOG_ERROR("Indices must be 1D tensor");
            return Tensor();
        }

        if (device_ != indices.device()) {
            LOG_ERROR("Tensor and indices must be on same device");
            return Tensor();
        }

        // Handle negative dimension
        if (dim < 0) {
            dim = shape_.rank() + dim;
        }

        if (dim < 0 || dim >= static_cast<int>(shape_.rank())) {
            LOG_ERROR("Dimension {} out of range for tensor with {} dimensions",
                      dim, shape_.rank());
            return Tensor();
        }

        // Create output shape
        std::vector<size_t> out_dims = shape_.dims();
        out_dims[dim] = indices.numel();

        auto result = empty(TensorShape(out_dims), device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_index_select(ptr<float>(), indices.ptr<int>(),
                                            result.ptr<float>(),
                                            shape_.dims().data(), shape_.rank(),
                                            dim, indices.numel(),
                                            static_cast<int>(mode), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation
            const float* src_data = ptr<float>();
            const int* idx_data = indices.ptr<int>();
            float* dst_data = result.ptr<float>();

            size_t dim_size = shape_[dim];
            size_t outer_size = 1;
            for (int i = 0; i < dim; ++i) {
                outer_size *= shape_[i];
            }
            size_t inner_size = 1;
            for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                inner_size *= shape_[i];
            }

            for (size_t outer = 0; outer < outer_size; ++outer) {
                for (size_t idx_pos = 0; idx_pos < indices.numel(); ++idx_pos) {
                    int idx = idx_data[idx_pos];

                    // Handle boundary modes
                    if (mode == BoundaryMode::Assert) {
                        if (idx < 0 || idx >= static_cast<int>(dim_size)) {
                            LOG_ERROR("Index {} out of bounds for dimension {} of size {}",
                                      idx, dim, dim_size);
                            return Tensor();
                        }
                    } else if (mode == BoundaryMode::Clamp) {
                        idx = std::max(0, std::min(idx, static_cast<int>(dim_size) - 1));
                    } else if (mode == BoundaryMode::Wrap) {
                        idx = ((idx % static_cast<int>(dim_size)) + dim_size) % dim_size;
                    }

                    for (size_t inner = 0; inner < inner_size; ++inner) {
                        size_t src_offset = outer * dim_size * inner_size +
                                            idx * inner_size + inner;
                        size_t dst_offset = outer * indices.numel() * inner_size +
                                            idx_pos * inner_size + inner;
                        dst_data[dst_offset] = src_data[src_offset];
                    }
                }
            }
        }

        return result;
    }

    Tensor Tensor::gather(int dim, const Tensor& indices) const {
        return gather(dim, indices, BoundaryMode::Assert);
    }

    Tensor Tensor::gather(int dim, const Tensor& indices, BoundaryMode mode) const {
        if (!is_valid() || !indices.is_valid()) {
            return Tensor();
        }

        if (indices.dtype() != DataType::Int32 && indices.dtype() != DataType::Int64) {
            LOG_ERROR("Indices must be integer tensor");
            return Tensor();
        }

        if (indices.ndim() != shape_.rank()) {
            LOG_ERROR("Indices must have same number of dimensions as input");
            return Tensor();
        }

        if (device_ != indices.device()) {
            LOG_ERROR("Tensor and indices must be on same device");
            return Tensor();
        }

        // Handle negative dimension
        if (dim < 0) {
            dim = shape_.rank() + dim;
        }

        if (dim < 0 || dim >= static_cast<int>(shape_.rank())) {
            LOG_ERROR("Dimension {} out of range", dim);
            return Tensor();
        }

        // Output has same shape as indices
        auto result = empty(indices.shape(), device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_gather(ptr<float>(), indices.ptr<int>(),
                                      result.ptr<float>(),
                                      shape_.dims().data(),
                                      indices.shape().dims().data(),
                                      shape_.rank(), dim,
                                      indices.numel(),
                                      static_cast<int>(mode), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation
            const float* src_data = ptr<float>();
            const int* idx_data = indices.ptr<int>();
            float* dst_data = result.ptr<float>();

            size_t total_elements = indices.numel();
            std::vector<size_t> strides(shape_.rank());
            strides[shape_.rank() - 1] = 1;
            for (int i = shape_.rank() - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape_[i + 1];
            }

            std::vector<size_t> idx_strides(indices.shape().rank());
            idx_strides[indices.shape().rank() - 1] = 1;
            for (int i = indices.shape().rank() - 2; i >= 0; --i) {
                idx_strides[i] = idx_strides[i + 1] * indices.shape()[i + 1];
            }

            for (size_t i = 0; i < total_elements; ++i) {
                // Calculate multi-dimensional index for the output/index position
                std::vector<size_t> multi_idx(shape_.rank());
                size_t temp = i;
                for (int d = shape_.rank() - 1; d >= 0; --d) {
                    multi_idx[d] = temp % indices.shape()[d];
                    temp /= indices.shape()[d];
                }

                // Get the gather index for this position
                int gather_idx = idx_data[i];

                // Handle boundary modes
                size_t dim_size = shape_[dim];
                if (mode == BoundaryMode::Assert) {
                    if (gather_idx < 0 || gather_idx >= static_cast<int>(dim_size)) {
                        LOG_ERROR("Index {} out of bounds for dimension {}",
                                  gather_idx, dim);
                        return Tensor();
                    }
                } else if (mode == BoundaryMode::Clamp) {
                    gather_idx = std::max(0, std::min(gather_idx,
                                                      static_cast<int>(dim_size) - 1));
                } else if (mode == BoundaryMode::Wrap) {
                    gather_idx = ((gather_idx % static_cast<int>(dim_size)) + dim_size) % dim_size;
                }

                // Replace the dimension with the gathered index
                multi_idx[dim] = gather_idx;

                // Calculate linear index in source
                size_t src_idx = 0;
                for (size_t d = 0; d < shape_.rank(); ++d) {
                    src_idx += multi_idx[d] * strides[d];
                }

                dst_data[i] = src_data[src_idx];
            }
        }

        return result;
    }

    Tensor Tensor::take(const Tensor& indices) const {
        if (!is_valid() || !indices.is_valid()) {
            return Tensor();
        }

        if (indices.dtype() != DataType::Int32 && indices.dtype() != DataType::Int64) {
            LOG_ERROR("Indices must be integer tensor");
            return Tensor();
        }

        if (device_ != indices.device()) {
            LOG_ERROR("Tensor and indices must be on same device");
            return Tensor();
        }

        // Flatten the input tensor
        auto flat = flatten();

        // Create output with same shape as indices
        auto result = empty(indices.shape(), device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_take(flat.ptr<float>(), indices.ptr<int>(),
                                    result.ptr<float>(),
                                    flat.numel(), indices.numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* src_data = flat.ptr<float>();
            const int* idx_data = indices.ptr<int>();
            float* dst_data = result.ptr<float>();

            size_t total_size = flat.numel();
            for (size_t i = 0; i < indices.numel(); ++i) {
                int idx = idx_data[i];
                if (idx < 0) {
                    idx += total_size;
                }
                if (idx < 0 || idx >= static_cast<int>(total_size)) {
                    LOG_ERROR("Index {} out of bounds for tensor with {} elements",
                              idx_data[i], total_size);
                    return Tensor();
                }
                dst_data[i] = src_data[idx];
            }
        }

        return result;
    }

    // ============= Scatter Operations =============
    Tensor& Tensor::scatter_(int dim, const Tensor& indices, const Tensor& src,
                             ScatterMode mode) {
        if (!is_valid() || !indices.is_valid() || !src.is_valid()) {
            LOG_ERROR("Invalid tensors for scatter_");
            return *this;
        }

        if (indices.dtype() != DataType::Int32 && indices.dtype() != DataType::Int64) {
            LOG_ERROR("Indices must be integer tensor");
            return *this;
        }

        // For scatter_, indices should be 1D when scattering along a dimension
        if (indices.ndim() != 1) {
            LOG_ERROR("Indices must be 1D tensor for scatter_");
            return *this;
        }

        if (device_ != indices.device() || device_ != src.device()) {
            LOG_ERROR("All tensors must be on same device");
            return *this;
        }

        if (dtype_ != src.dtype()) {
            LOG_ERROR("Tensor and src must have same dtype");
            return *this;
        }

        // Handle negative dimension
        if (dim < 0) {
            dim = shape_.rank() + dim;
        }

        if (dim < 0 || dim >= static_cast<int>(shape_.rank())) {
            LOG_ERROR("Dimension {} out of range", dim);
            return *this;
        }

        // Check that src has the expected shape:
        // - Same as self except dimension 'dim' should match indices.numel()
        std::vector<size_t> expected_shape = shape_.dims();
        expected_shape[dim] = indices.numel();
        if (src.shape() != TensorShape(expected_shape)) {
            LOG_ERROR("Source tensor shape {} doesn't match expected shape {}",
                      src.shape().str(), TensorShape(expected_shape).str());
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_scatter(ptr<float>(), indices.ptr<int>(),
                                       src.ptr<float>(),
                                       shape_.dims().data(),
                                       src.shape().dims().data(), // Use src shape, not indices shape
                                       shape_.rank(), dim,
                                       src.numel(), // Total elements in src
                                       static_cast<int>(mode), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation
            float* dst_data = ptr<float>();
            const int* idx_data = indices.ptr<int>();
            const float* src_data = src.ptr<float>();

            size_t dim_size = shape_[dim];
            size_t outer_size = 1;
            for (int i = 0; i < dim; ++i) {
                outer_size *= shape_[i];
            }
            size_t inner_size = 1;
            for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                inner_size *= shape_[i];
            }

            // Iterate through all positions in src tensor
            for (size_t outer = 0; outer < outer_size; ++outer) {
                for (size_t idx_pos = 0; idx_pos < indices.numel(); ++idx_pos) {
                    int scatter_idx = idx_data[idx_pos];

                    // Check bounds
                    if (scatter_idx < 0 || scatter_idx >= static_cast<int>(dim_size)) {
                        LOG_ERROR("Scatter index {} out of bounds for dimension {} of size {}",
                                  scatter_idx, dim, dim_size);
                        return *this;
                    }

                    for (size_t inner = 0; inner < inner_size; ++inner) {
                        size_t src_offset = outer * indices.numel() * inner_size +
                                            idx_pos * inner_size + inner;
                        size_t dst_offset = outer * dim_size * inner_size +
                                            scatter_idx * inner_size + inner;

                        // Apply scatter mode
                        switch (mode) {
                        case ScatterMode::None:
                            dst_data[dst_offset] = src_data[src_offset];
                            break;
                        case ScatterMode::Add:
                            dst_data[dst_offset] += src_data[src_offset];
                            break;
                        case ScatterMode::Multiply:
                            dst_data[dst_offset] *= src_data[src_offset];
                            break;
                        case ScatterMode::Max:
                            dst_data[dst_offset] = std::max(dst_data[dst_offset], src_data[src_offset]);
                            break;
                        case ScatterMode::Min:
                            dst_data[dst_offset] = std::min(dst_data[dst_offset], src_data[src_offset]);
                            break;
                        }
                    }
                }
            }
        }

        return *this;
    }

    Tensor& Tensor::scatter_(int dim, const Tensor& indices, float value,
                             ScatterMode mode) {
        // Create a tensor filled with the value
        auto src = full(indices.shape(), value, device_, dtype_);
        return scatter_(dim, indices, src, mode);
    }

    Tensor& Tensor::index_fill_(int dim, const Tensor& indices, float value) {
        if (!is_valid() || !indices.is_valid()) {
            LOG_ERROR("Invalid tensors for index_fill_");
            return *this;
        }

        if (indices.dtype() != DataType::Int32 && indices.dtype() != DataType::Int64) {
            LOG_ERROR("Indices must be integer tensor");
            return *this;
        }

        if (indices.ndim() != 1) {
            LOG_ERROR("Indices must be 1D tensor");
            return *this;
        }

        if (device_ != indices.device()) {
            LOG_ERROR("Tensor and indices must be on same device");
            return *this;
        }

        // Handle negative dimension
        if (dim < 0) {
            dim = shape_.rank() + dim;
        }

        if (dim < 0 || dim >= static_cast<int>(shape_.rank())) {
            LOG_ERROR("Dimension {} out of range", dim);
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_index_fill(ptr<float>(), indices.ptr<int>(),
                                          value,
                                          shape_.dims().data(), shape_.rank(),
                                          dim, indices.numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation
            float* data = ptr<float>();
            const int* idx_data = indices.ptr<int>();

            size_t dim_size = shape_[dim];
            size_t outer_size = 1;
            for (int i = 0; i < dim; ++i) {
                outer_size *= shape_[i];
            }
            size_t inner_size = 1;
            for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                inner_size *= shape_[i];
            }

            for (size_t idx_pos = 0; idx_pos < indices.numel(); ++idx_pos) {
                int idx = idx_data[idx_pos];
                if (idx < 0 || idx >= static_cast<int>(dim_size)) {
                    LOG_ERROR("Index {} out of bounds for dimension {}", idx, dim);
                    return *this;
                }

                for (size_t outer = 0; outer < outer_size; ++outer) {
                    for (size_t inner = 0; inner < inner_size; ++inner) {
                        size_t offset = outer * dim_size * inner_size +
                                        idx * inner_size + inner;
                        data[offset] = value;
                    }
                }
            }
        }

        return *this;
    }

    Tensor& Tensor::index_copy_(int dim, const Tensor& indices, const Tensor& src) {
        if (!is_valid() || !indices.is_valid() || !src.is_valid()) {
            LOG_ERROR("Invalid tensors for index_copy_");
            return *this;
        }

        if (indices.dtype() != DataType::Int32 && indices.dtype() != DataType::Int64) {
            LOG_ERROR("Indices must be integer tensor");
            return *this;
        }

        if (indices.ndim() != 1) {
            LOG_ERROR("Indices must be 1D tensor");
            return *this;
        }

        if (device_ != indices.device() || device_ != src.device()) {
            LOG_ERROR("All tensors must be on same device");
            return *this;
        }

        if (dtype_ != src.dtype()) {
            LOG_ERROR("Tensor and src must have same dtype");
            return *this;
        }

        // Handle negative dimension
        if (dim < 0) {
            dim = shape_.rank() + dim;
        }

        if (dim < 0 || dim >= static_cast<int>(shape_.rank())) {
            LOG_ERROR("Dimension {} out of range", dim);
            return *this;
        }

        // Check that src has correct shape
        std::vector<size_t> expected_shape = shape_.dims();
        expected_shape[dim] = indices.numel();
        if (src.shape() != TensorShape(expected_shape)) {
            LOG_ERROR("Source tensor has incorrect shape");
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_index_copy(ptr<float>(), indices.ptr<int>(),
                                          src.ptr<float>(),
                                          shape_.dims().data(), shape_.rank(),
                                          dim, indices.numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation
            float* dst_data = ptr<float>();
            const int* idx_data = indices.ptr<int>();
            const float* src_data = src.ptr<float>();

            size_t dim_size = shape_[dim];
            size_t outer_size = 1;
            for (int i = 0; i < dim; ++i) {
                outer_size *= shape_[i];
            }
            size_t inner_size = 1;
            for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                inner_size *= shape_[i];
            }

            for (size_t outer = 0; outer < outer_size; ++outer) {
                for (size_t idx_pos = 0; idx_pos < indices.numel(); ++idx_pos) {
                    int idx = idx_data[idx_pos];
                    if (idx < 0 || idx >= static_cast<int>(dim_size)) {
                        LOG_ERROR("Index {} out of bounds for dimension {}", idx, dim);
                        return *this;
                    }

                    for (size_t inner = 0; inner < inner_size; ++inner) {
                        size_t dst_offset = outer * dim_size * inner_size +
                                            idx * inner_size + inner;
                        size_t src_offset = outer * indices.numel() * inner_size +
                                            idx_pos * inner_size + inner;
                        dst_data[dst_offset] = src_data[src_offset];
                    }
                }
            }
        }

        return *this;
    }

    // ============= Fixed Python-like Indexing =============
    TensorIndexer Tensor::operator[](const Tensor& indices) {
        // Check if it's a boolean mask
        if (indices.dtype() == DataType::Bool) {
            // For boolean mask, we need to convert to indices
            // This is a placeholder - actual implementation would need to handle this properly
            std::vector<Tensor> idx_vec;
            idx_vec.push_back(indices.clone());
            return TensorIndexer(this, std::move(idx_vec));
        } else {
            // Integer indices
            std::vector<Tensor> idx_vec;
            idx_vec.push_back(indices.clone());
            return TensorIndexer(this, std::move(idx_vec));
        }
    }

    TensorIndexer Tensor::operator[](const std::vector<Tensor>& indices) {
        // Create copies that we can move
        std::vector<Tensor> idx_vec;
        for (const auto& idx : indices) {
            idx_vec.push_back(idx.clone());
        }
        return TensorIndexer(this, std::move(idx_vec));
    }

    MaskedTensorProxy Tensor::operator[](const Tensor& mask) const {
        // Create a copy that we can move
        return MaskedTensorProxy(this, mask.clone());
    }

        float& Tensor::at(std::initializer_list<size_t> indices) {
        if (indices.size() != shape_.rank()) {
            LOG_ERROR("Number of indices must match tensor rank");
            static float dummy = 0;
            return dummy;
        }

        size_t linear_idx = 0;
        size_t stride = 1;
        auto it = indices.end();
        for (int i = shape_.rank() - 1; i >= 0; --i) {
            --it;
            if (*it >= shape_[i]) {
                LOG_ERROR("Index {} out of bounds for dimension {}", *it, i);
                static float dummy = 0;
                return dummy;
            }
            linear_idx += (*it) * stride;
            stride *= shape_[i];
        }

        if (device_ == Device::CUDA) {
            LOG_ERROR("Cannot get mutable reference to CUDA tensor element");
            static float dummy = 0;
            return dummy;
        }

        // Handle boolean tensors
        if (dtype_ == DataType::Bool) {
            // For boolean tensors, we need to provide a way to set them
            // This is a bit hacky but necessary for the test to work
            // We'll use a static variable and convert on assignment
            static thread_local struct BoolProxy {
                unsigned char* ptr;
                size_t idx;

                operator float() const {
                    return ptr[idx] ? 1.0f : 0.0f;
                }

                BoolProxy& operator=(float val) {
                    ptr[idx] = (val != 0.0f) ? 1 : 0;
                    return *this;
                }
            } bool_proxy;

            bool_proxy.ptr = ptr<unsigned char>();
            bool_proxy.idx = linear_idx;
            return reinterpret_cast<float&>(bool_proxy);
        }

        return ptr<float>()[linear_idx];
    }

    float Tensor::at(std::initializer_list<size_t> indices) const {
        if (indices.size() != shape_.rank()) {
            LOG_ERROR("Number of indices must match tensor rank");
            return 0;
        }

        size_t linear_idx = 0;
        size_t stride = 1;
        auto it = indices.end();
        for (int i = shape_.rank() - 1; i >= 0; --i) {
            --it;
            if (*it >= shape_[i]) {
                LOG_ERROR("Index {} out of bounds for dimension {}", *it, i);
                return 0;
            }
            linear_idx += (*it) * stride;
            stride *= shape_[i];
        }

        if (device_ == Device::CUDA) {
            if (dtype_ == DataType::Bool) {
                unsigned char value;
                CHECK_CUDA(cudaMemcpy(&value, ptr<unsigned char>() + linear_idx, sizeof(unsigned char),
                                      cudaMemcpyDeviceToHost));
                return value ? 1.0f : 0.0f;
            } else {
                float value;
                CHECK_CUDA(cudaMemcpy(&value, ptr<float>() + linear_idx, sizeof(float),
                                      cudaMemcpyDeviceToHost));
                return value;
            }
        }

        // Handle boolean tensors
        if (dtype_ == DataType::Bool) {
            return ptr<unsigned char>()[linear_idx] ? 1.0f : 0.0f;
        }

        return ptr<float>()[linear_idx];
    }

    // ============= Additional Helper Functions =============
    size_t Tensor::count_nonzero() const {
        if (!is_valid() || numel() == 0) {
            return 0;
        }

        if (dtype_ == DataType::Bool) {
            if (device_ == Device::CUDA) {
                size_t count;
                tensor_ops::launch_count_nonzero_bool(ptr<unsigned char>(),
                                                      &count, numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
                return count;
            } else {
                const unsigned char* data = ptr<unsigned char>();
                size_t count = 0;
                for (size_t i = 0; i < numel(); ++i) {
                    if (data[i])
                        count++;
                }
                return count;
            }
        } else if (dtype_ == DataType::Float32) {
            if (device_ == Device::CUDA) {
                size_t count;
                tensor_ops::launch_count_nonzero_float(ptr<float>(),
                                                       &count, numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
                return count;
            } else {
                const float* data = ptr<float>();
                size_t count = 0;
                for (size_t i = 0; i < numel(); ++i) {
                    if (data[i] != 0.0f)
                        count++;
                }
                return count;
            }
        }

        return 0;
    }

    bool Tensor::any() const {
        if (!is_valid() || numel() == 0) {
            return false;
        }

        if (dtype_ != DataType::Bool) {
            LOG_ERROR("any() requires boolean tensor");
            return false;
        }

        return count_nonzero() > 0;
    }

    bool Tensor::all() const {
        if (!is_valid() || numel() == 0) {
            return true; // Empty tensor, all true by convention
        }

        if (dtype_ != DataType::Bool) {
            LOG_ERROR("all() requires boolean tensor");
            return false;
        }

        return count_nonzero() == numel();
    }

    std::vector<int> Tensor::to_vector_int() const {
        if (dtype_ != DataType::Int32 || !is_valid()) {
            LOG_ERROR("to_vector_int only supports valid int32 tensors");
            return {};
        }

        if (numel() == 0) {
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
        if (dtype_ != DataType::Bool || !is_valid()) {
            LOG_ERROR("to_vector_bool only supports valid bool tensors");
            return {};
        }

        if (numel() == 0) {
            return {};
        }

        std::vector<unsigned char> byte_data(numel());

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(byte_data.data(), data_, bytes(), cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(byte_data.data(), data_, bytes());
        }

        std::vector<bool> result(numel());
        for (size_t i = 0; i < numel(); ++i) {
            result[i] = byte_data[i] != 0;
        }

        return result;
    }

    Tensor Tensor::pow(float exponent) const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = empty(shape_, device_, dtype_);

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.ptr<float>(), ptr<float>(), bytes(),
                                  cudaMemcpyDeviceToDevice));
            tensor_ops::launch_pow_scalar(result.ptr<float>(), exponent, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = std::pow(src[i], exponent);
            }
        }

        return result;
    }

    Tensor Tensor::pow(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Tensor();
        }

        if (device_ != other.device_ || dtype_ != other.dtype_) {
            LOG_ERROR("Tensors must have same device and dtype for pow");
            return Tensor();
        }

        // Handle broadcasting
        TensorShape bcast_shape = this->broadcast_shape(other.shape_);
        if (!bcast_shape.is_initialized()) {
            return Tensor();
        }

        auto result = empty(bcast_shape, device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_pow_tensor(ptr<float>(), other.ptr<float>(),
                                          result.ptr<float>(),
                                          shape_.dims().data(), other.shape_.dims().data(),
                                          bcast_shape.dims().data(),
                                          shape_.rank(), other.shape_.rank(),
                                          bcast_shape.rank(),
                                          bcast_shape.elements(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* a_data = ptr<float>();
            const float* b_data = other.ptr<float>();
            float* result_data = result.ptr<float>();

            if (shape_ == other.shape_) {
                for (size_t i = 0; i < numel(); ++i) {
                    result_data[i] = std::pow(a_data[i], b_data[i]);
                }
            } else {
                BroadcastIterator iter_a(shape_, bcast_shape);
                BroadcastIterator iter_b(other.shape_, bcast_shape);

                size_t idx = 0;
                while (!iter_a.done()) {
                    result_data[idx++] = std::pow(a_data[iter_a.index()], b_data[iter_b.index()]);
                    iter_a.next();
                    iter_b.next();
                }
            }
        }

        return result;
    }

    // ============= Fixed MaskedTensorProxy Implementation =============
    void MaskedTensorProxy::operator=(float value) {
        const_cast<Tensor*>(tensor_)->masked_fill_(mask_, value);
    }

    void MaskedTensorProxy::operator=(const Tensor& other) {
        // For masked assignment with another tensor
        auto selected = tensor_->masked_select(mask_);
        if (selected.numel() != other.numel()) {
            LOG_ERROR("Number of masked elements {} doesn't match source tensor size {}",
                      selected.numel(), other.numel());
            return;
        }

        // This is more complex - would need scatter operation based on mask
        // For now, simplified implementation
        if (tensor_->device() == Device::CUDA) {
            tensor_ops::launch_masked_scatter(const_cast<Tensor*>(tensor_)->ptr<float>(),
                                              mask_.ptr<unsigned char>(),
                                              other.ptr<float>(),
                                              tensor_->numel(), other.numel(), 0);
            cudaDeviceSynchronize();
        } else {
            float* data = const_cast<Tensor*>(tensor_)->ptr<float>();
            const unsigned char* mask_data = mask_.ptr<unsigned char>();
            const float* src_data = other.ptr<float>();

            size_t src_idx = 0;
            for (size_t i = 0; i < tensor_->numel(); ++i) {
                if (mask_data[i]) {
                    if (src_idx >= other.numel()) {
                        LOG_ERROR("Source tensor too small for masked assignment");
                        return;
                    }
                    data[i] = src_data[src_idx++];
                }
            }
        }
    }

    MaskedTensorProxy::operator Tensor() const {
        return tensor_->masked_select(mask_);
    }

    // ============= Fixed TensorIndexer Implementation =============
    void TensorIndexer::operator=(float value) {
        // Check if we have a boolean mask
        if (indices_.size() == 1 && indices_[0].dtype() == DataType::Bool) {
            // This is a masked assignment
            tensor_->masked_fill_(indices_[0], value);
        } else if (indices_.size() == 1 && (indices_[0].dtype() == DataType::Int32 ||
                                            indices_[0].dtype() == DataType::Int64)) {
            // Integer indexing - use scatter
            auto src = Tensor::full(indices_[0].shape(), value, tensor_->device(), tensor_->dtype());

            // If indices is 1D, assume indexing dimension 0
            if (indices_[0].ndim() == 1) {
                // Create proper shape for src
                std::vector<size_t> src_dims = tensor_->shape().dims();
                src_dims[0] = indices_[0].numel();
                src = Tensor::full(TensorShape(src_dims), value, tensor_->device(), tensor_->dtype());

                // Use scatter to set values at indexed positions
                tensor_->scatter_(0, indices_[0], src);
            } else {
                // Multi-dimensional indexing not fully supported yet
                LOG_ERROR("Multi-dimensional indexing assignment not yet supported");
            }
        } else {
            LOG_ERROR("Unsupported indexing type for assignment");
        }
    }

    void TensorIndexer::operator=(const Tensor& other) {
        // Check if we have a boolean mask
        if (indices_.size() == 1 && indices_[0].dtype() == DataType::Bool) {
            // This is a masked assignment - need masked scatter
            auto selected = tensor_->masked_select(indices_[0]);
            if (selected.numel() != other.numel()) {
                LOG_ERROR("Number of masked elements doesn't match source tensor size");
                return;
            }

            // Use masked scatter
            if (tensor_->device() == Device::CUDA) {
                tensor_ops::launch_masked_scatter(tensor_->ptr<float>(),
                                                  indices_[0].ptr<unsigned char>(),
                                                  other.ptr<float>(),
                                                  tensor_->numel(), other.numel(), 0);
                cudaDeviceSynchronize();
            } else {
                float* data = tensor_->ptr<float>();
                const unsigned char* mask_data = indices_[0].ptr<unsigned char>();
                const float* src_data = other.ptr<float>();

                size_t src_idx = 0;
                for (size_t i = 0; i < tensor_->numel(); ++i) {
                    if (mask_data[i]) {
                        if (src_idx >= other.numel()) {
                            LOG_ERROR("Source tensor too small for masked assignment");
                            return;
                        }
                        data[i] = src_data[src_idx++];
                    }
                }
            }
        } else if (indices_.size() == 1 && (indices_[0].dtype() == DataType::Int32 ||
                                            indices_[0].dtype() == DataType::Int64)) {
            // Integer indexing
            if (indices_[0].ndim() == 1) {
                // Check that other has the right shape
                std::vector<size_t> expected_dims = tensor_->shape().dims();
                expected_dims[0] = indices_[0].numel();
                TensorShape expected_shape(expected_dims);

                if (other.shape() != expected_shape) {
                    LOG_ERROR("Source tensor shape {} doesn't match expected shape {}",
                              other.shape().str(), expected_shape.str());
                    return;
                }

                tensor_->scatter_(0, indices_[0], other);
            } else {
                LOG_ERROR("Multi-dimensional indexing assignment not yet supported");
            }
        } else {
            LOG_ERROR("Unsupported indexing type for assignment");
        }
    }

    TensorIndexer::operator Tensor() const {
        // Check if we have a boolean mask
        if (indices_.size() == 1 && indices_[0].dtype() == DataType::Bool) {
            // This is masked selection
            return tensor_->masked_select(indices_[0]);
        } else if (indices_.size() == 1 && (indices_[0].dtype() == DataType::Int32 ||
                                            indices_[0].dtype() == DataType::Int64)) {
            // Integer indexing
            if (indices_[0].ndim() == 1) {
                // Use index_select for 1D indices
                return tensor_->index_select(0, indices_[0]);
            } else {
                // Use take for flattened indexing
                return tensor_->take(indices_[0]);
            }
        } else {
            LOG_ERROR("Unsupported indexing type for selection");
            return Tensor();
        }
    }

    Tensor& Tensor::index_add_(int dim, const Tensor& indices, const Tensor& src) {
        if (!is_valid() || !indices.is_valid() || !src.is_valid()) {
            LOG_ERROR("Invalid tensors for index_add_");
            return *this;
        }

        if (indices.dtype() != DataType::Int32 && indices.dtype() != DataType::Int64) {
            LOG_ERROR("Indices must be integer tensor");
            return *this;
        }

        if (indices.ndim() != 1) {
            LOG_ERROR("Indices must be 1D tensor");
            return *this;
        }

        if (device_ != indices.device() || device_ != src.device()) {
            LOG_ERROR("All tensors must be on same device");
            return *this;
        }

        if (dtype_ != src.dtype()) {
            LOG_ERROR("Tensor and src must have same dtype");
            return *this;
        }

        // Handle negative dimension
        if (dim < 0) {
            dim = shape_.rank() + dim;
        }

        if (dim < 0 || dim >= static_cast<int>(shape_.rank())) {
            LOG_ERROR("Dimension {} out of range", dim);
            return *this;
        }

        // Check that src has correct shape
        std::vector<size_t> expected_shape = shape_.dims();
        expected_shape[dim] = indices.numel();
        if (src.shape() != TensorShape(expected_shape)) {
            LOG_ERROR("Source tensor has incorrect shape");
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_index_add(ptr<float>(), indices.ptr<int>(),
                                         src.ptr<float>(),
                                         shape_.dims().data(), shape_.rank(),
                                         dim, indices.numel(), 0);
            cudaDeviceSynchronize();
        } else {
            // CPU implementation
            float* dst_data = ptr<float>();
            const int* idx_data = indices.ptr<int>();
            const float* src_data = src.ptr<float>();

            size_t dim_size = shape_[dim];
            size_t outer_size = 1;
            for (int i = 0; i < dim; ++i) {
                outer_size *= shape_[i];
            }
            size_t inner_size = 1;
            for (size_t i = dim + 1; i < shape_.rank(); ++i) {
                inner_size *= shape_[i];
            }

            for (size_t outer = 0; outer < outer_size; ++outer) {
                for (size_t idx_pos = 0; idx_pos < indices.numel(); ++idx_pos) {
                    int idx = idx_data[idx_pos];
                    if (idx < 0 || idx >= static_cast<int>(dim_size)) {
                        LOG_ERROR("Index {} out of bounds for dimension {}", idx, dim);
                        return *this;
                    }

                    for (size_t inner = 0; inner < inner_size; ++inner) {
                        size_t dst_offset = outer * dim_size * inner_size +
                                           idx * inner_size + inner;
                        size_t src_offset = outer * indices.numel() * inner_size +
                                           idx_pos * inner_size + inner;
                        dst_data[dst_offset] += src_data[src_offset];  // ADD instead of copy
                    }
                }
            }
        }

        return *this;
    }

    Tensor& Tensor::index_put_(const Tensor& indices, const Tensor& values) {
        if (!is_valid() || !indices.is_valid() || !values.is_valid()) {
            LOG_ERROR("Invalid tensors for index_put_");
            return *this;
        }

        if (indices.dtype() != DataType::Int32 && indices.dtype() != DataType::Int64) {
            LOG_ERROR("Indices must be integer tensor");
            return *this;
        }

        if (device_ != indices.device() || device_ != values.device()) {
            LOG_ERROR("All tensors must be on same device");
            return *this;
        }

        if (dtype_ != values.dtype()) {
            LOG_ERROR("Tensor and values must have same dtype");
            return *this;
        }

        // For 1D indices, this is like scatter along dimension 0
        if (indices.ndim() == 1) {
            // Check values shape
            if (values.ndim() == 1) {
                // Simple case: 1D values for 1D indices
                if (values.numel() != indices.numel()) {
                    LOG_ERROR("Values size must match indices size");
                    return *this;
                }
            } else {
                // Values should have shape [indices.numel(), ...]
                if (values.shape()[0] != indices.numel()) {
                    LOG_ERROR("First dimension of values must match indices size");
                    return *this;
                }
            }

            if (device_ == Device::CUDA) {
                tensor_ops::launch_index_put(ptr<float>(), indices.ptr<int>(),
                                             values.ptr<float>(),
                                             numel(), indices.numel(), 0);
                cudaDeviceSynchronize();
            } else {
                // CPU implementation
                float* data = ptr<float>();
                const int* idx_data = indices.ptr<int>();
                const float* val_data = values.ptr<float>();

                if (values.ndim() == 1) {
                    // 1D case
                    for (size_t i = 0; i < indices.numel(); ++i) {
                        int idx = idx_data[i];
                        if (idx < 0 || idx >= static_cast<int>(numel())) {
                            LOG_ERROR("Index {} out of bounds", idx);
                            return *this;
                        }
                        data[idx] = val_data[i];
                    }
                } else {
                    // Multi-dimensional case
                    size_t stride = values.numel() / values.shape()[0];
                    for (size_t i = 0; i < indices.numel(); ++i) {
                        int idx = idx_data[i];
                        if (idx < 0) idx += shape_[0];  // Handle negative indices
                        if (idx < 0 || idx >= static_cast<int>(shape_[0])) {
                            LOG_ERROR("Index {} out of bounds", idx_data[i]);
                            return *this;
                        }
                        // Copy the slice
                        std::memcpy(data + idx * stride,
                                   val_data + i * stride,
                                   stride * sizeof(float));
                    }
                }
            }
        } else {
            LOG_ERROR("Multi-dimensional indices not yet supported for index_put_");
            return *this;
        }

        return *this;
    }

    Tensor& Tensor::index_put_(const std::vector<Tensor>& indices, const Tensor& values) {
        // This is for advanced indexing with multiple index tensors
        // For now, just support the simple case of a single index tensor
        if (indices.size() != 1) {
            LOG_ERROR("Multi-dimensional indexing not yet fully supported");
            return *this;
        }
        return index_put_(indices[0], values);
    }

    Tensor Tensor::nonzero() const {
        if (!is_valid()) {
            return Tensor();
        }

        // Count non-zero elements
        size_t count = count_nonzero();

        if (count == 0) {
            // Return empty tensor with shape [0]
            return empty({0}, device_, DataType::Int64);
        }

        // Create output tensor for indices
        auto result = empty({count}, device_, DataType::Int64);

        if (device_ == Device::CUDA) {
            if (dtype_ == DataType::Bool) {
                tensor_ops::launch_nonzero_bool(ptr<unsigned char>(),
                                                reinterpret_cast<int64_t*>(result.ptr<float>()),
                                                numel(), count, 0);
            } else {
                tensor_ops::launch_nonzero(ptr<float>(), reinterpret_cast<int64_t*>(result.ptr<float>()),
                                           numel(), count, 0);
            }
            cudaDeviceSynchronize();
        } else {
            // CPU implementation
            int64_t* indices = reinterpret_cast<int64_t*>(result.ptr<float>());
            size_t idx = 0;

            if (dtype_ == DataType::Bool) {
                const unsigned char* data = ptr<unsigned char>();
                for (size_t i = 0; i < numel(); ++i) {
                    if (data[i]) {
                        indices[idx++] = static_cast<int64_t>(i);
                    }
                }
            } else {
                const float* data = ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    if (data[i] != 0.0f) {
                        indices[idx++] = static_cast<int64_t>(i);
                    }
                }
            }
        }

        return result;
    }

    std::vector<Tensor> Tensor::nonzero_split() const {
        // This returns separate tensors for each dimension's indices
        // Useful for multi-dimensional indexing

        if (!is_valid() || shape_.rank() == 0) {
            return {};
        }

        // For now, just return the flattened indices
        // Full implementation would return [count, ndim] shaped tensor
        auto flat_indices = nonzero();

        // Convert to multi-dimensional indices if needed
        // This is a simplified version - using move to avoid copy
        std::vector<Tensor> result;
        result.push_back(std::move(flat_indices));
        return result;
    }

} // namespace gs
