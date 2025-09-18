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

// Broadcasting and type promotion helper implementation
std::pair<Tensor, Tensor> Tensor::_broadcasted(const Tensor& other, bool match_dtype) const {
    // Step 1: Get broadcast shape
    auto result_shape = broadcast::shape(shape_.dims(), other.shape_.dims());
    if (result_shape.empty()) {
        LOG_ERROR("Cannot broadcast shapes {} and {}", shape_.str(), other.shape_.str());
        return {Tensor(), Tensor()};
    }

    TensorShape out_shape(result_shape);

    // Step 2: Determine result dtype if matching
    DataType result_dtype = dtype_;
    if (match_dtype && dtype_ != other.dtype_) {
        result_dtype = promote_types(dtype_, other.dtype_);
    }

    // Step 3: Prepare both tensors (use clone to avoid copy constructor)
    Tensor a_ready = this->clone();  // Use clone instead of copy
    Tensor b_ready = other.clone();  // Use clone instead of copy

    // Convert dtypes if needed
    if (match_dtype) {
        if (a_ready.dtype_ != result_dtype) {
            a_ready = a_ready.to(result_dtype);
        }
        if (b_ready.dtype_ != result_dtype) {
            b_ready = b_ready.to(result_dtype);
        }
    }

    // Broadcast shapes if needed
    if (a_ready.shape_ != out_shape) {
        a_ready = gs::broadcast_to(a_ready, out_shape);
    }
    if (b_ready.shape_ != out_shape) {
        b_ready = gs::broadcast_to(b_ready, out_shape);
    }

    return {std::move(a_ready), std::move(b_ready)};
}

// Overload for scalar inputs
template<typename T>
std::pair<Tensor, Tensor> Tensor::_broadcasted(const T& scalar) const {
    static_assert(std::is_arithmetic_v<T>, "Scalar must be arithmetic type");

    // Create scalar tensor with same shape and device
    Tensor scalar_tensor = full(shape_, static_cast<float>(scalar), device_, dtype_);
    return {this->clone(), std::move(scalar_tensor)};  // Use clone for first element
}

// Simplified unified binary operation implementation
template<typename T>
Tensor Tensor::binary_op_impl(const T& other, BinaryOp op) const {
    if (!is_valid()) return Tensor();

    // Special handling for logical operations - require bool tensors
    if (op >= BinaryOp::LogicalAnd && op <= BinaryOp::LogicalXor) {
        if constexpr (std::is_same_v<T, Tensor>) {
            if (dtype_ != DataType::Bool || other.dtype_ != DataType::Bool) {
                LOG_ERROR("Logical operations require boolean tensors");
                return Tensor();
            }
        } else {
            LOG_ERROR("Logical operations require tensor operands");
            return Tensor();
        }
    }

    // Step 1: Get broadcasted tensors with matching shapes and dtypes
    Tensor lhs, rhs;
    if constexpr (std::is_arithmetic_v<T>) {
        auto [a, b] = _broadcasted(static_cast<float>(other));
        lhs = std::move(a);
        rhs = std::move(b);
    } else {
        // For comparison ops, we don't need dtype matching
        bool match_dtype = !(op >= BinaryOp::Equal && op <= BinaryOp::GreaterEqual);
        auto [a, b] = _broadcasted(other, match_dtype);
        lhs = std::move(a);
        rhs = std::move(b);
    }

    if (!lhs.is_valid() || !rhs.is_valid()) return Tensor();

    // Step 2: Determine output dtype based on operation
    DataType output_dtype = lhs.dtype_;
    if (op >= BinaryOp::Equal && op <= BinaryOp::GreaterEqual) {
        output_dtype = DataType::Bool;
    } else if (op >= BinaryOp::LogicalAnd && op <= BinaryOp::LogicalXor) {
        output_dtype = DataType::Bool;
    }

    // Step 3: Create result and dispatch
    Tensor result = empty(lhs.shape_, device_, output_dtype);

    if (device_ == Device::CUDA) {
        tensor_ops::launch_binary_op(
            lhs.raw_ptr(), rhs.raw_ptr(), result.raw_ptr(),
            result.numel(), op,
            lhs.dtype_, rhs.dtype_, output_dtype, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
    } else {
        // CPU implementation
        if (output_dtype == DataType::Bool && lhs.dtype_ == DataType::Float32) {
            // Comparison operations on float tensors
            const float* a_data = lhs.ptr<float>();
            const float* b_data = rhs.ptr<float>();
            unsigned char* r_data = result.ptr<unsigned char>();

            for (size_t i = 0; i < result.numel(); ++i) {
                bool res = false;
                switch (op) {
                    case BinaryOp::Equal: res = (a_data[i] == b_data[i]); break;
                    case BinaryOp::NotEqual: res = (a_data[i] != b_data[i]); break;
                    case BinaryOp::Less: res = (a_data[i] < b_data[i]); break;
                    case BinaryOp::LessEqual: res = (a_data[i] <= b_data[i]); break;
                    case BinaryOp::Greater: res = (a_data[i] > b_data[i]); break;
                    case BinaryOp::GreaterEqual: res = (a_data[i] >= b_data[i]); break;
                    default: break;
                }
                r_data[i] = res ? 1 : 0;
            }
        } else if (output_dtype == DataType::Bool && lhs.dtype_ == DataType::Bool) {
            // Logical operations on bool tensors
            const unsigned char* a_data = lhs.ptr<unsigned char>();
            const unsigned char* b_data = rhs.ptr<unsigned char>();
            unsigned char* r_data = result.ptr<unsigned char>();

            for (size_t i = 0; i < result.numel(); ++i) {
                bool res = false;
                switch (op) {
                    case BinaryOp::LogicalAnd: res = (a_data[i] && b_data[i]); break;
                    case BinaryOp::LogicalOr: res = (a_data[i] || b_data[i]); break;
                    case BinaryOp::LogicalXor: res = ((a_data[i] != 0) != (b_data[i] != 0)); break;
                    default: break;
                }
                r_data[i] = res ? 1 : 0;
            }
        } else {
            // Arithmetic operations on float tensors
            const float* a_data = lhs.ptr<float>();
            const float* b_data = rhs.ptr<float>();
            float* r_data = result.ptr<float>();

            for (size_t i = 0; i < result.numel(); ++i) {
                switch (op) {
                    case BinaryOp::Add: r_data[i] = a_data[i] + b_data[i]; break;
                    case BinaryOp::Sub: r_data[i] = a_data[i] - b_data[i]; break;
                    case BinaryOp::Mul: r_data[i] = a_data[i] * b_data[i]; break;
                    case BinaryOp::Div: r_data[i] = a_data[i] / (b_data[i] + 1e-8f); break;
                    case BinaryOp::Pow: r_data[i] = std::pow(a_data[i], b_data[i]); break;
                    case BinaryOp::Mod: r_data[i] = std::fmod(a_data[i], b_data[i]); break;
                    case BinaryOp::Maximum: r_data[i] = std::max(a_data[i], b_data[i]); break;
                    case BinaryOp::Minimum: r_data[i] = std::min(a_data[i], b_data[i]); break;
                    default: break;
                }
            }
        }
    }

    return result;
}

// In-place version - no broadcasting or type conversion
template<typename T>
Tensor& Tensor::binary_op_inplace_impl(const T& other, BinaryOp op) {
    if (!is_valid() || numel() == 0) return *this;

    // In-place operations don't support type change
    if (op >= BinaryOp::Equal && op <= BinaryOp::GreaterEqual) {
        LOG_ERROR("Comparison operations cannot be done in-place");
        return *this;
    }

    if constexpr (std::is_arithmetic_v<T>) {
        float scalar = static_cast<float>(other);
        if (device_ == Device::CUDA) {
            tensor_ops::launch_binary_scalar_inplace(
                ptr<float>(), scalar, numel(), op, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                switch (op) {
                    case BinaryOp::Add: data[i] += scalar; break;
                    case BinaryOp::Sub: data[i] -= scalar; break;
                    case BinaryOp::Mul: data[i] *= scalar; break;
                    case BinaryOp::Div: data[i] /= scalar; break;
                    case BinaryOp::Pow: data[i] = std::pow(data[i], scalar); break;
                    case BinaryOp::Mod: data[i] = std::fmod(data[i], scalar); break;
                    case BinaryOp::Maximum: data[i] = std::max(data[i], scalar); break;
                    case BinaryOp::Minimum: data[i] = std::min(data[i], scalar); break;
                    default: break;
                }
            }
        }
    } else {
        const Tensor& b = other;
        if (!b.is_valid() || shape_ != b.shape_ || device_ != b.device_) {
            LOG_ERROR("Invalid tensors for in-place operation");
            return *this;
        }

        // Check dtype compatibility
        if (dtype_ != b.dtype_) {
            LOG_ERROR("In-place operations require same dtype");
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_binary_op_inplace(
                ptr<float>(), b.ptr<float>(), numel(), op, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* a = ptr<float>();
            const float* b_data = b.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                switch (op) {
                    case BinaryOp::Add: a[i] += b_data[i]; break;
                    case BinaryOp::Sub: a[i] -= b_data[i]; break;
                    case BinaryOp::Mul: a[i] *= b_data[i]; break;
                    case BinaryOp::Div: a[i] /= (b_data[i] + 1e-8f); break;
                    case BinaryOp::Pow: a[i] = std::pow(a[i], b_data[i]); break;
                    case BinaryOp::Mod: a[i] = std::fmod(a[i], b_data[i]); break;
                    case BinaryOp::Maximum: a[i] = std::max(a[i], b_data[i]); break;
                    case BinaryOp::Minimum: a[i] = std::min(a[i], b_data[i]); break;
                    default: break;
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

// Explicit instantiations for _broadcasted
template std::pair<Tensor, Tensor> Tensor::_broadcasted<float>(const float&) const;
template std::pair<Tensor, Tensor> Tensor::_broadcasted<double>(const double&) const;
template std::pair<Tensor, Tensor> Tensor::_broadcasted<int>(const int&) const;

#undef CHECK_CUDA

} // namespace gs