/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor_broadcast.hpp"
#include <cmath>
#include <cuda_runtime.h>

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

    // Single unified implementation for ALL arithmetic operations
    template<typename T>
    Tensor Tensor::binary_op_impl(const T& other, BinaryOp op) const {
        if (!is_valid()) return Tensor();

        if constexpr (std::is_arithmetic_v<T>) {
            // Scalar operations
            if (numel() == 0) return clone();  // Empty tensor + scalar = empty tensor

            Tensor result = clone();
            float scalar = static_cast<float>(other);

            if (device_ == Device::CUDA) {
                void(*fn)(float*, float, size_t, cudaStream_t) = nullptr;
                switch (op) {
                    case BinaryOp::Add: fn = tensor_ops::launch_scalar_add; break;
                    case BinaryOp::Sub: fn = tensor_ops::launch_scalar_sub; break;
                    case BinaryOp::Mul: fn = tensor_ops::launch_scalar_mul; break;
                    case BinaryOp::Div: fn = tensor_ops::launch_scalar_div; break;
                    case BinaryOp::Pow:
                        CHECK_CUDA(cudaMemcpy(result.ptr<float>(), ptr<float>(), bytes(), cudaMemcpyDeviceToDevice));
                        tensor_ops::launch_pow_scalar(result.ptr<float>(), scalar, numel(), 0);
                        CHECK_CUDA(cudaDeviceSynchronize());
                        return result;
                }
                if (fn) {
                    fn(result.ptr<float>(), scalar, numel(), 0);
                    CHECK_CUDA(cudaDeviceSynchronize());
                }
            } else {
                float* data = result.ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    switch (op) {
                        case BinaryOp::Add: data[i] += scalar; break;
                        case BinaryOp::Sub: data[i] -= scalar; break;
                        case BinaryOp::Mul: data[i] *= scalar; break;
                        case BinaryOp::Div: data[i] /= scalar; break;
                        case BinaryOp::Pow: data[i] = std::pow(data[i], scalar); break;
                    }
                }
            }
            return result;
        } else {
            // Tensor operations
            const Tensor& b = other;
            if (!b.is_valid()) return Tensor();

            if (device_ != b.device_ || dtype_ != b.dtype_) {
                LOG_ERROR("Tensors must have same device and dtype");
                return Tensor();
            }

            // Check for empty tensors - operations with empty should fail
            if (numel() == 0 || b.numel() == 0) {
                // Special case: if shapes are identical and both empty, return empty clone
                if (shape_ == b.shape_ && numel() == 0) {
                    return clone();
                }
                // Otherwise, can't broadcast empty with non-empty
                LOG_ERROR("Cannot perform operations between empty and non-empty tensors");
                return Tensor();
            }

            // Determine if broadcasting needed
            TensorShape out_shape;
            if (shape_ == b.shape_) {
                out_shape = shape_;
            } else {
                if (!BroadcastHelper::can_broadcast(shape_, b.shape_)) {
                    LOG_ERROR("Cannot broadcast shapes {} and {}", shape_.str(), b.shape_.str());
                    return Tensor();
                }
                out_shape = BroadcastHelper::broadcast_shape(shape_, b.shape_);
                if (!out_shape.is_initialized()) {
                    return Tensor();
                }
            }

            Tensor result = empty(out_shape, device_, dtype_);

            if (device_ == Device::CUDA) {
                if (shape_ == b.shape_) {
                    // Fast path - no broadcast
                    void(*fn)(const float*, const float*, float*, size_t, cudaStream_t) = nullptr;
                    switch (op) {
                        case BinaryOp::Add: fn = tensor_ops::launch_element_add; break;
                        case BinaryOp::Sub: fn = tensor_ops::launch_element_sub; break;
                        case BinaryOp::Mul: fn = tensor_ops::launch_element_mul; break;
                        case BinaryOp::Div: fn = tensor_ops::launch_element_div; break;
                        case BinaryOp::Pow:
                            tensor_ops::launch_pow_tensor(ptr<float>(), b.ptr<float>(), result.ptr<float>(),
                                shape_.dims().data(), b.shape_.dims().data(), out_shape.dims().data(),
                                shape_.rank(), b.shape_.rank(), out_shape.rank(), out_shape.elements(), 0);
                            CHECK_CUDA(cudaDeviceSynchronize());
                            return result;
                    }
                    if (fn) {
                        fn(ptr<float>(), b.ptr<float>(), result.ptr<float>(), numel(), 0);
                        CHECK_CUDA(cudaDeviceSynchronize());
                    }
                } else {
                    // Broadcast path
                    void(*fn)(const float*, const float*, float*, const size_t*, const size_t*, const size_t*,
                              size_t, size_t, size_t, size_t, cudaStream_t) = nullptr;
                    switch (op) {
                        case BinaryOp::Add: fn = tensor_ops::launch_broadcast_add; break;
                        case BinaryOp::Sub: fn = tensor_ops::launch_broadcast_sub; break;
                        case BinaryOp::Mul: fn = tensor_ops::launch_broadcast_mul; break;
                        case BinaryOp::Div: fn = tensor_ops::launch_broadcast_div; break;
                        case BinaryOp::Pow:
                            tensor_ops::launch_pow_tensor(ptr<float>(), b.ptr<float>(), result.ptr<float>(),
                                shape_.dims().data(), b.shape_.dims().data(), out_shape.dims().data(),
                                shape_.rank(), b.shape_.rank(), out_shape.rank(), out_shape.elements(), 0);
                            CHECK_CUDA(cudaDeviceSynchronize());
                            return result;
                    }
                    if (fn) {
                        fn(ptr<float>(), b.ptr<float>(), result.ptr<float>(),
                           shape_.dims().data(), b.shape_.dims().data(), out_shape.dims().data(),
                           shape_.rank(), b.shape_.rank(), out_shape.rank(), out_shape.elements(), 0);
                        CHECK_CUDA(cudaDeviceSynchronize());
                    }
                }
            } else {
                // CPU path
                const float* a_data = ptr<float>();
                const float* b_data = b.ptr<float>();
                float* r_data = result.ptr<float>();

                if (shape_ == b.shape_) {
                    for (size_t i = 0; i < numel(); ++i) {
                        switch (op) {
                            case BinaryOp::Add: r_data[i] = a_data[i] + b_data[i]; break;
                            case BinaryOp::Sub: r_data[i] = a_data[i] - b_data[i]; break;
                            case BinaryOp::Mul: r_data[i] = a_data[i] * b_data[i]; break;
                            case BinaryOp::Div: r_data[i] = a_data[i] / (b_data[i] + 1e-8f); break;
                            case BinaryOp::Pow: r_data[i] = std::pow(a_data[i], b_data[i]); break;
                        }
                    }
                } else {
                    BroadcastIterator iter_a(shape_, out_shape);
                    BroadcastIterator iter_b(b.shape_, out_shape);
                    for (size_t i = 0; !iter_a.done(); ++i, iter_a.next(), iter_b.next()) {
                        switch (op) {
                            case BinaryOp::Add: r_data[i] = a_data[iter_a.index()] + b_data[iter_b.index()]; break;
                            case BinaryOp::Sub: r_data[i] = a_data[iter_a.index()] - b_data[iter_b.index()]; break;
                            case BinaryOp::Mul: r_data[i] = a_data[iter_a.index()] * b_data[iter_b.index()]; break;
                            case BinaryOp::Div: r_data[i] = a_data[iter_a.index()] / (b_data[iter_b.index()] + 1e-8f); break;
                            case BinaryOp::Pow: r_data[i] = std::pow(a_data[iter_a.index()], b_data[iter_b.index()]); break;
                        }
                    }
                }
            }
            return result;
        }
    }

    // In-place version - much simpler, no broadcasting
    template<typename T>
    Tensor& Tensor::binary_op_inplace_impl(const T& other, BinaryOp op) {
        if (!is_valid() || numel() == 0) return *this;

        if constexpr (std::is_arithmetic_v<T>) {
            float scalar = static_cast<float>(other);
            if (device_ == Device::CUDA) {
                void(*fn)(float*, float, size_t, cudaStream_t) = nullptr;
                switch (op) {
                    case BinaryOp::Add: fn = tensor_ops::launch_scalar_add; break;
                    case BinaryOp::Sub: fn = tensor_ops::launch_scalar_sub; break;
                    case BinaryOp::Mul: fn = tensor_ops::launch_scalar_mul; break;
                    case BinaryOp::Div:
                        if (std::abs(scalar) < 1e-10f) {
                            LOG_WARN("Division by near-zero scalar: {}", scalar);
                            scalar = 1e-10f;
                        }
                        fn = tensor_ops::launch_scalar_div;
                        break;
                }
                if (fn) {
                    fn(ptr<float>(), scalar, numel(), 0);
                    CHECK_CUDA(cudaDeviceSynchronize());
                }
            } else {
                float* data = ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    switch (op) {
                        case BinaryOp::Add: data[i] += scalar; break;
                        case BinaryOp::Sub: data[i] -= scalar; break;
                        case BinaryOp::Mul: data[i] *= scalar; break;
                        case BinaryOp::Div:
                            if (std::abs(scalar) < 1e-10f) {
                                LOG_WARN("Division by near-zero scalar: {}", scalar);
                                data[i] /= 1e-10f;
                            } else {
                                data[i] /= scalar;
                            }
                            break;
                    }
                }
            }
        } else {
            const Tensor& b = other;
            if (!b.is_valid() || shape_ != b.shape_ || device_ != b.device_) {
                LOG_ERROR("Invalid tensors for in-place operation");
                return *this;
            }

            if (device_ == Device::CUDA) {
                void(*fn)(float*, const float*, size_t, cudaStream_t) = nullptr;
                switch (op) {
                    case BinaryOp::Add: fn = tensor_ops::launch_element_add_inplace; break;
                    case BinaryOp::Sub: fn = tensor_ops::launch_element_sub_inplace; break;
                    case BinaryOp::Mul: fn = tensor_ops::launch_element_mul_inplace; break;
                    case BinaryOp::Div: fn = tensor_ops::launch_element_div_inplace; break;
                }
                if (fn) {
                    fn(ptr<float>(), b.ptr<float>(), numel(), 0);
                    CHECK_CUDA(cudaDeviceSynchronize());
                }
            } else {
                float* a = ptr<float>();
                const float* b_data = b.ptr<float>();
                for (size_t i = 0; i < numel(); ++i) {
                    switch (op) {
                        case BinaryOp::Add: a[i] += b_data[i]; break;
                        case BinaryOp::Sub: a[i] -= b_data[i]; break;
                        case BinaryOp::Mul: a[i] *= b_data[i]; break;
                        case BinaryOp::Div: a[i] /= (b_data[i] + 1e-8f); break;
                    }
                }
            }
        }
        return *this;
    }

    // Explicit instantiations
    template Tensor Tensor::binary_op_impl<float>(const float&, BinaryOp) const;
    template Tensor Tensor::binary_op_impl<double>(const double&, BinaryOp) const;
    template Tensor Tensor::binary_op_impl<int>(const int&, BinaryOp) const;
    template Tensor Tensor::binary_op_impl<Tensor>(const Tensor&, BinaryOp) const;

    template Tensor& Tensor::binary_op_inplace_impl<float>(const float&, BinaryOp);
    template Tensor& Tensor::binary_op_inplace_impl<double>(const double&, BinaryOp);
    template Tensor& Tensor::binary_op_inplace_impl<int>(const int&, BinaryOp);
    template Tensor& Tensor::binary_op_inplace_impl<Tensor>(const Tensor&, BinaryOp);

#undef CHECK_CUDA

} // namespace gs
