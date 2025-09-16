/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor_broadcast.hpp"
#include <cmath>
#include <cuda_runtime.h>
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

    // ============= Scalar Operations =============
    Tensor Tensor::add(float scalar) const {
        if (!is_valid()) {
            return Tensor();
        }

        if (numel() == 0) {
            return clone();
        }

        Tensor result = clone();

        if (device_ == Device::CUDA) {
            tensor_ops::launch_scalar_add(result.ptr<float>(), scalar, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] += scalar;
            }
        }

        return result;
    }

    Tensor Tensor::sub(float scalar) const {
        return add(-scalar);
    }

    Tensor Tensor::mul(float scalar) const {
        if (!is_valid()) {
            return Tensor();
        }

        if (numel() == 0) {
            return clone();
        }

        Tensor result = clone();

        if (device_ == Device::CUDA) {
            tensor_ops::launch_scalar_mul(result.ptr<float>(), scalar, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] *= scalar;
            }
        }

        return result;
    }

    Tensor Tensor::div(float scalar) const {
        if (!is_valid()) {
            return Tensor();
        }

        // Allow division by very small numbers but not exactly zero
        if (std::abs(scalar) < 1e-10f) {
            LOG_WARN("Division by near-zero scalar: {}", scalar);
            // Return valid result with large values
            return mul(1e10f); // Multiply by large number instead
        }

        if (numel() == 0) {
            return clone();
        }

        Tensor result = clone();

        if (device_ == Device::CUDA) {
            tensor_ops::launch_scalar_div(result.ptr<float>(), scalar, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] /= scalar;
            }
        }

        return result;
    }

    Tensor Tensor::neg() const {
        return mul(-1.0f);
    }

    // ============= Element-wise Operations with Broadcasting =============
    Tensor Tensor::add(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Tensors must be on the same device for addition");
            return Tensor();
        }

        if (dtype_ != other.dtype_) {
            LOG_ERROR("Dtype mismatch for addition");
            return Tensor();
        }

        // Check if broadcasting is needed
        if (shape_ == other.shape_) {
            // Same shape, use non-broadcast path
            if (numel() == 0) {
                return clone();
            }

            Tensor result = empty(shape_, device_, dtype_);
            if (device_ == Device::CUDA) {
                tensor_ops::launch_element_add(ptr<float>(), other.ptr<float>(),
                                               result.ptr<float>(), numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const float* a = ptr<float>();
                const float* b = other.ptr<float>();
                float* c = result.ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    c[i] = a[i] + b[i];
                }
            }
            return result;
        }

        // Shapes differ, try broadcasting
        if (!BroadcastHelper::can_broadcast(shape_, other.shape_)) {
            LOG_ERROR("Cannot broadcast shapes {} and {} for addition",
                      shape_.str(), other.shape_.str());
            return Tensor();
        }

        // Get broadcast shape
        TensorShape broadcast_shape = BroadcastHelper::broadcast_shape(shape_, other.shape_);
        Tensor result = empty(broadcast_shape, device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_broadcast_add(
                ptr<float>(), other.ptr<float>(), result.ptr<float>(),
                shape_.dims().data(), other.shape_.dims().data(), broadcast_shape.dims().data(),
                shape_.rank(), other.shape_.rank(), broadcast_shape.rank(),
                broadcast_shape.elements(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation with broadcasting
            BroadcastIterator iter_a(shape_, broadcast_shape);
            BroadcastIterator iter_b(other.shape_, broadcast_shape);
            const float* a_data = ptr<float>();
            const float* b_data = other.ptr<float>();
            float* result_data = result.ptr<float>();

            size_t idx = 0;
            while (!iter_a.done()) {
                result_data[idx++] = a_data[iter_a.index()] + b_data[iter_b.index()];
                iter_a.next();
                iter_b.next();
            }
        }

        return result;
    }

    Tensor Tensor::sub(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Tensors must be on the same device for subtraction");
            return Tensor();
        }

        if (dtype_ != other.dtype_) {
            LOG_ERROR("Dtype mismatch for subtraction");
            return Tensor();
        }

        // Check if broadcasting is needed
        if (shape_ == other.shape_) {
            // Same shape, use non-broadcast path
            if (numel() == 0) {
                return clone();
            }

            Tensor result = empty(shape_, device_, dtype_);
            if (device_ == Device::CUDA) {
                tensor_ops::launch_element_sub(ptr<float>(), other.ptr<float>(),
                                               result.ptr<float>(), numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const float* a = ptr<float>();
                const float* b = other.ptr<float>();
                float* c = result.ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    c[i] = a[i] - b[i];
                }
            }
            return result;
        }

        // Shapes differ, try broadcasting
        if (!BroadcastHelper::can_broadcast(shape_, other.shape_)) {
            LOG_ERROR("Cannot broadcast shapes {} and {} for subtraction",
                      shape_.str(), other.shape_.str());
            return Tensor();
        }

        TensorShape broadcast_shape = BroadcastHelper::broadcast_shape(shape_, other.shape_);
        Tensor result = empty(broadcast_shape, device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_broadcast_sub(
                ptr<float>(), other.ptr<float>(), result.ptr<float>(),
                shape_.dims().data(), other.shape_.dims().data(), broadcast_shape.dims().data(),
                shape_.rank(), other.shape_.rank(), broadcast_shape.rank(),
                broadcast_shape.elements(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            BroadcastIterator iter_a(shape_, broadcast_shape);
            BroadcastIterator iter_b(other.shape_, broadcast_shape);
            const float* a_data = ptr<float>();
            const float* b_data = other.ptr<float>();
            float* result_data = result.ptr<float>();

            size_t idx = 0;
            while (!iter_a.done()) {
                result_data[idx++] = a_data[iter_a.index()] - b_data[iter_b.index()];
                iter_a.next();
                iter_b.next();
            }
        }

        return result;
    }

    Tensor Tensor::mul(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Tensors must be on the same device for multiplication");
            return Tensor();
        }

        if (dtype_ != other.dtype_) {
            LOG_ERROR("Dtype mismatch for multiplication");
            return Tensor();
        }

        // Check if broadcasting is needed
        if (shape_ == other.shape_) {
            // Same shape, use non-broadcast path
            if (numel() == 0) {
                return clone();
            }

            Tensor result = empty(shape_, device_, dtype_);
            if (device_ == Device::CUDA) {
                tensor_ops::launch_element_mul(ptr<float>(), other.ptr<float>(),
                                               result.ptr<float>(), numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const float* a = ptr<float>();
                const float* b = other.ptr<float>();
                float* c = result.ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    c[i] = a[i] * b[i];
                }
            }
            return result;
        }

        // Shapes differ, try broadcasting
        if (!BroadcastHelper::can_broadcast(shape_, other.shape_)) {
            LOG_ERROR("Cannot broadcast shapes {} and {} for multiplication",
                      shape_.str(), other.shape_.str());
            return Tensor();
        }

        TensorShape broadcast_shape = BroadcastHelper::broadcast_shape(shape_, other.shape_);
        Tensor result = empty(broadcast_shape, device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_broadcast_mul(
                ptr<float>(), other.ptr<float>(), result.ptr<float>(),
                shape_.dims().data(), other.shape_.dims().data(), broadcast_shape.dims().data(),
                shape_.rank(), other.shape_.rank(), broadcast_shape.rank(),
                broadcast_shape.elements(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            BroadcastIterator iter_a(shape_, broadcast_shape);
            BroadcastIterator iter_b(other.shape_, broadcast_shape);
            const float* a_data = ptr<float>();
            const float* b_data = other.ptr<float>();
            float* result_data = result.ptr<float>();

            size_t idx = 0;
            while (!iter_a.done()) {
                result_data[idx++] = a_data[iter_a.index()] * b_data[iter_b.index()];
                iter_a.next();
                iter_b.next();
            }
        }

        return result;
    }

    Tensor Tensor::div(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("Tensors must be on the same device for division");
            return Tensor();
        }

        if (dtype_ != other.dtype_) {
            LOG_ERROR("Dtype mismatch for division");
            return Tensor();
        }

        // Check if broadcasting is needed
        if (shape_ == other.shape_) {
            // Same shape, use non-broadcast path
            if (numel() == 0) {
                return clone();
            }

            Tensor result = empty(shape_, device_, dtype_);
            if (device_ == Device::CUDA) {
                tensor_ops::launch_element_div(ptr<float>(), other.ptr<float>(),
                                               result.ptr<float>(), numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const float* a = ptr<float>();
                const float* b = other.ptr<float>();
                float* c = result.ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    c[i] = a[i] / (b[i] + 1e-8f); // Safe division
                }
            }
            return result;
        }

        // Shapes differ, try broadcasting
        if (!BroadcastHelper::can_broadcast(shape_, other.shape_)) {
            LOG_ERROR("Cannot broadcast shapes {} and {} for division",
                      shape_.str(), other.shape_.str());
            return Tensor();
        }

        TensorShape broadcast_shape = BroadcastHelper::broadcast_shape(shape_, other.shape_);
        Tensor result = empty(broadcast_shape, device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_broadcast_div(
                ptr<float>(), other.ptr<float>(), result.ptr<float>(),
                shape_.dims().data(), other.shape_.dims().data(), broadcast_shape.dims().data(),
                shape_.rank(), other.shape_.rank(), broadcast_shape.rank(),
                broadcast_shape.elements(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            BroadcastIterator iter_a(shape_, broadcast_shape);
            BroadcastIterator iter_b(other.shape_, broadcast_shape);
            const float* a_data = ptr<float>();
            const float* b_data = other.ptr<float>();
            float* result_data = result.ptr<float>();

            size_t idx = 0;
            while (!iter_a.done()) {
                result_data[idx++] = a_data[iter_a.index()] / (b_data[iter_b.index()] + 1e-8f);
                iter_a.next();
                iter_b.next();
            }
        }

        return result;
    }

    // ============= In-place Operations =============
    Tensor& Tensor::add_(float scalar) {
        if (!is_valid() || numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_scalar_add(ptr<float>(), scalar, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] += scalar;
            }
        }

        return *this;
    }

    Tensor& Tensor::sub_(float scalar) {
        return add_(-scalar);
    }

    Tensor& Tensor::mul_(float scalar) {
        if (!is_valid() || numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_scalar_mul(ptr<float>(), scalar, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] *= scalar;
            }
        }

        return *this;
    }

    Tensor& Tensor::div_(float scalar) {
        if (std::abs(scalar) < 1e-10f) {
            LOG_WARN("Division by near-zero scalar: {}", scalar);
            return mul_(1e10f);
        }

        if (!is_valid() || numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_scalar_div(ptr<float>(), scalar, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] /= scalar;
            }
        }

        return *this;
    }

    Tensor& Tensor::add_(const Tensor& other) {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for in-place addition");
            return *this;
        }

        if (shape_ != other.shape_) {
            LOG_ERROR("In-place operations require same shape: {} vs {}",
                      shape_.str(), other.shape_.str());
            return *this;
        }

        if (device_ != other.device_) {
            LOG_ERROR("Tensors must be on the same device");
            return *this;
        }

        if (numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_element_add_inplace(ptr<float>(), other.ptr<float>(),
                                                   numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* a = ptr<float>();
            const float* b = other.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                a[i] += b[i];
            }
        }

        return *this;
    }

    Tensor& Tensor::sub_(const Tensor& other) {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for in-place subtraction");
            return *this;
        }

        if (shape_ != other.shape_) {
            LOG_ERROR("In-place operations require same shape");
            return *this;
        }

        if (device_ != other.device_) {
            LOG_ERROR("Tensors must be on the same device");
            return *this;
        }

        if (numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_element_sub_inplace(ptr<float>(), other.ptr<float>(),
                                                   numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* a = ptr<float>();
            const float* b = other.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                a[i] -= b[i];
            }
        }

        return *this;
    }

    Tensor& Tensor::mul_(const Tensor& other) {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for in-place multiplication");
            return *this;
        }

        if (shape_ != other.shape_) {
            LOG_ERROR("In-place operations require same shape");
            return *this;
        }

        if (device_ != other.device_) {
            LOG_ERROR("Tensors must be on the same device");
            return *this;
        }

        if (numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_element_mul_inplace(ptr<float>(), other.ptr<float>(),
                                                   numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* a = ptr<float>();
            const float* b = other.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                a[i] *= b[i];
            }
        }

        return *this;
    }

    Tensor& Tensor::div_(const Tensor& other) {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensors for in-place division");
            return *this;
        }

        if (shape_ != other.shape_) {
            LOG_ERROR("In-place operations require same shape");
            return *this;
        }

        if (device_ != other.device_) {
            LOG_ERROR("Tensors must be on the same device");
            return *this;
        }

        if (numel() == 0) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_element_div_inplace(ptr<float>(), other.ptr<float>(),
                                                   numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* a = ptr<float>();
            const float* b = other.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                a[i] /= (b[i] + 1e-8f);
            }
        }

        return *this;
    }

#undef CHECK_CUDA

} // namespace gs
