/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include "core/tensor_broadcast.hpp"  // Add this include for broadcast namespace
#include <algorithm>
#include <cmath>
#include <numeric>
#include <execution>

#define CHECK_CUDA(call) do { \
    if (auto e = call; e != cudaSuccess) { \
        LOG_ERROR("CUDA error: {}", cudaGetErrorString(e)); \
    } \
} while(0)

namespace gs {

// ============= Unified Unary Operation =============
Tensor Tensor::unary(UnaryOp op) const {
    if (!is_valid()) return {};

    // Special case for logical_not - different dtype handling
    if (op == UnaryOp::LogicalNot) {
        if (dtype_ != DataType::Bool) {
            LOG_ERROR("logical_not requires boolean tensor");
            return {};
        }
        auto result = empty(shape_, device_, DataType::Bool);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_unary_op(raw_ptr(), result.raw_ptr(),
                                       numel(), op, dtype_, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const unsigned char* src = ptr<unsigned char>();
            unsigned char* dst = result.ptr<unsigned char>();
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = !src[i];
            }
        }
        return result;
    }

    // Determine output dtype
    DataType out_dtype = dtype_;
    if (op == UnaryOp::IsNan || op == UnaryOp::IsInf || op == UnaryOp::IsFinite) {
        out_dtype = DataType::Bool;
    }

    auto result = empty(shape_, device_, out_dtype);
    if (numel() == 0) return result;

    if (device_ == Device::CUDA) {
        tensor_ops::launch_unary_op(raw_ptr(), result.raw_ptr(),
                                   numel(), op, dtype_, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
    } else {
        // CPU implementation
        if (dtype_ == DataType::Float32) {
            const float* src = ptr<float>();

            if (out_dtype == DataType::Bool) {
                unsigned char* dst = result.ptr<unsigned char>();

                #pragma omp parallel for if(numel() > 1024)
                for (size_t i = 0; i < numel(); ++i) {
                    switch (op) {
                        case UnaryOp::IsNan: dst[i] = std::isnan(src[i]); break;
                        case UnaryOp::IsInf: dst[i] = std::isinf(src[i]); break;
                        case UnaryOp::IsFinite: dst[i] = std::isfinite(src[i]); break;
                        default: dst[i] = 0; break;
                    }
                }
            } else {
                float* dst = result.ptr<float>();

                #pragma omp parallel for if(numel() > 1024)
                for (size_t i = 0; i < numel(); ++i) {
                    switch (op) {
                        case UnaryOp::Neg: dst[i] = -src[i]; break;
                        case UnaryOp::Abs: dst[i] = std::abs(src[i]); break;
                        case UnaryOp::Sign: dst[i] = (src[i] > 0) - (src[i] < 0); break;
                        case UnaryOp::Reciprocal: dst[i] = 1.0f / (src[i] + 1e-8f); break;

                        case UnaryOp::Exp: dst[i] = std::exp(src[i]); break;
                        case UnaryOp::Exp2: dst[i] = std::exp2(src[i]); break;
                        case UnaryOp::Log: dst[i] = std::log(std::max(1e-10f, src[i])); break;
                        case UnaryOp::Log2: dst[i] = std::log2(std::max(1e-10f, src[i])); break;
                        case UnaryOp::Log10: dst[i] = std::log10(std::max(1e-10f, src[i])); break;
                        case UnaryOp::Log1p: dst[i] = std::log1p(src[i]); break;

                        case UnaryOp::Sqrt: dst[i] = std::sqrt(std::max(0.0f, src[i])); break;
                        case UnaryOp::Rsqrt: dst[i] = 1.0f / std::sqrt(std::max(1e-10f, src[i])); break;
                        case UnaryOp::Square: dst[i] = src[i] * src[i]; break;

                        case UnaryOp::Sin: dst[i] = std::sin(src[i]); break;
                        case UnaryOp::Cos: dst[i] = std::cos(src[i]); break;
                        case UnaryOp::Tan: dst[i] = std::tan(src[i]); break;
                        case UnaryOp::Asin: dst[i] = std::asin(std::clamp(src[i], -1.0f, 1.0f)); break;
                        case UnaryOp::Acos: dst[i] = std::acos(std::clamp(src[i], -1.0f, 1.0f)); break;
                        case UnaryOp::Atan: dst[i] = std::atan(src[i]); break;

                        case UnaryOp::Sinh: dst[i] = std::sinh(src[i]); break;
                        case UnaryOp::Cosh: dst[i] = std::cosh(src[i]); break;
                        case UnaryOp::Tanh: dst[i] = std::tanh(src[i]); break;

                        case UnaryOp::Sigmoid: dst[i] = 1.0f / (1.0f + std::exp(-src[i])); break;
                        case UnaryOp::Relu: dst[i] = std::max(0.0f, src[i]); break;
                        case UnaryOp::Gelu: {
                            // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                            float x = src[i];
                            float inner = std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
                            dst[i] = 0.5f * x * (1.0f + std::tanh(inner));
                            break;
                        }
                        case UnaryOp::Swish: dst[i] = src[i] / (1.0f + std::exp(-src[i])); break;

                        case UnaryOp::Floor: dst[i] = std::floor(src[i]); break;
                        case UnaryOp::Ceil: dst[i] = std::ceil(src[i]); break;
                        case UnaryOp::Round: dst[i] = std::round(src[i]); break;
                        case UnaryOp::Trunc: dst[i] = std::trunc(src[i]); break;

                        default: dst[i] = src[i]; break;
                    }
                }
            }
        }
    }

    return result;
}

// ============= Unified Reduce Operation =============
Tensor Tensor::reduce(ReduceOp op, std::span<const int> axes, bool keepdim) const {
    if (!is_valid()) return {};

    // Resolve axes
    std::vector<int> resolved_axes;
    if (axes.empty()) {
        // Reduce all dimensions
        resolved_axes.reserve(shape_.rank());
        for (int i = 0; i < static_cast<int>(shape_.rank()); ++i) {
            resolved_axes.push_back(i);
        }
    } else {
        resolved_axes.reserve(axes.size());
        for (int ax : axes) {
            int resolved = resolve_dim(ax);
            if (resolved < 0 || resolved >= static_cast<int>(shape_.rank())) {
                LOG_ERROR("Invalid axis {} for tensor with {} dimensions", ax, shape_.rank());
                return {};
            }
            resolved_axes.push_back(resolved);
        }
        // Sort and unique
        std::sort(resolved_axes.begin(), resolved_axes.end());
        resolved_axes.erase(std::unique(resolved_axes.begin(), resolved_axes.end()),
                           resolved_axes.end());
    }

    // Calculate output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < shape_.rank(); ++i) {
        bool is_reduced = std::find(resolved_axes.begin(), resolved_axes.end(), i)
                         != resolved_axes.end();
        if (!is_reduced) {
            out_shape.push_back(shape_[i]);
        } else if (keepdim) {
            out_shape.push_back(1);
        }
    }

    // Handle empty shape (scalar result)
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    // Determine output dtype
    DataType out_dtype = dtype_;
    if (op == ReduceOp::Any || op == ReduceOp::All) {
        if (dtype_ != DataType::Bool) {
            LOG_ERROR("Any/All reduction requires boolean tensor");
            return {};
        }
        out_dtype = DataType::Bool;
    } else if (op == ReduceOp::Argmax || op == ReduceOp::Argmin ||
               op == ReduceOp::CountNonzero) {
        out_dtype = DataType::Int64;
    }

    auto result = empty(TensorShape(out_shape), device_, out_dtype);

    if (device_ == Device::CUDA) {
        tensor_ops::launch_reduce_op(raw_ptr(), result.raw_ptr(),
                                    shape_.dims().data(), shape_.rank(),
                                    resolved_axes.data(), resolved_axes.size(),
                                    keepdim, op, dtype_, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
    } else {
        // CPU implementation - simplified for now
        // Full implementation would handle all axes properly
        if (resolved_axes.size() == shape_.rank() && !keepdim) {
            // Reducing to scalar
            float val = 0.0f;

            switch (op) {
                case ReduceOp::Sum: val = sum_scalar(); break;
                case ReduceOp::Mean: val = mean_scalar(); break;
                case ReduceOp::Max: val = max_scalar(); break;
                case ReduceOp::Min: val = min_scalar(); break;
                case ReduceOp::Std: val = std_scalar(); break;
                case ReduceOp::Var: val = var_scalar(); break;
                default:
                    LOG_WARN("Reduce op {} not fully implemented on CPU", static_cast<int>(op));
                    val = 0.0f;
            }

            if (out_dtype == DataType::Float32) {
                *result.ptr<float>() = val;
            } else if (out_dtype == DataType::Int64) {
                *result.ptr<int64_t>() = static_cast<int64_t>(val);
            }
        } else {
            // Partial reduction - would need proper implementation
            LOG_WARN("Partial reduction not fully implemented on CPU");
            result.fill_(0.0f);
        }
    }

    return result;
}

// ============= Unified Ternary Operation =============
Tensor Tensor::ternary(const Tensor& b, const Tensor& c, TernaryOp op) const {
    if (!is_valid() || !b.is_valid() || !c.is_valid()) return {};

    switch (op) {
        case TernaryOp::Where: {
            // where(condition, x, y) - this is condition, b is x, c is y
            if (dtype_ != DataType::Bool) {
                LOG_ERROR("Where condition must be boolean tensor");
                return {};
            }
            return Tensor::where(*this, b, c);
        }

        case TernaryOp::MulAdd: {
            // a * b + c (FMA - Fused Multiply-Add)
            // First broadcast all to common shape
            auto ab_shape = broadcast_shape(b.shape());
            auto result_shape = TensorShape(broadcast::shape(ab_shape.dims(), c.shape().dims()));

            if (result_shape.elements() == 0) {
                LOG_ERROR("Cannot broadcast shapes for muladd");
                return {};
            }

            auto result = empty(result_shape, device_, dtype_);

            if (device_ == Device::CUDA) {
                // Broadcast inputs if needed
                auto a_bcast = broadcast_to(result_shape);
                auto b_bcast = b.broadcast_to(result_shape);
                auto c_bcast = c.broadcast_to(result_shape);

                tensor_ops::launch_ternary_op(a_bcast.raw_ptr(), b_bcast.raw_ptr(),
                                             c_bcast.raw_ptr(), result.raw_ptr(),
                                             result.numel(), op, dtype_, 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                // CPU implementation
                auto a_bcast = broadcast_to(result_shape);
                auto b_bcast = b.broadcast_to(result_shape);
                auto c_bcast = c.broadcast_to(result_shape);

                const float* a_data = a_bcast.template ptr<float>();  // Use template keyword
                const float* b_data = b_bcast.template ptr<float>();  // Use template keyword
                const float* c_data = c_bcast.template ptr<float>();  // Use template keyword
                float* r_data = result.template ptr<float>();         // Use template keyword

                #pragma omp parallel for if(result.numel() > 1024)
                for (size_t i = 0; i < result.numel(); ++i) {
                    r_data[i] = a_data[i] * b_data[i] + c_data[i];
                }
            }

            return result;
        }

        case TernaryOp::Clamp: {
            // clamp(x, min, max) - this is x, b is min, c is max
            if (!b.shape().elements() == 1 || !c.shape().elements() == 1) {
                LOG_ERROR("Clamp min/max must be scalars");
                return {};
            }

            float min_val = b.item();
            float max_val = c.item();

            return clamp(min_val, max_val);
        }

        default:
            LOG_ERROR("Unknown ternary operation");
            return {};
    }
}

// Clamp implementation using ternary
Tensor Tensor::clamp(float min_val, float max_val) const {
    if (!is_valid()) return {};

    auto result = clone();
    if (numel() == 0) return result;

    if (device_ == Device::CUDA) {
        // For CUDA, we'll reuse existing clamp functionality through unary ops
        // or implement it directly here
        // We can use minimum(max_val).maximum(min_val) for now
        result = minimum(max_val).maximum(min_val);
    } else {
        float* data = result.ptr<float>();
        #pragma omp parallel for if(numel() > 1024)
        for (size_t i = 0; i < result.numel(); ++i) {
            data[i] = std::min(std::max(data[i], min_val), max_val);
        }
    }

    return result;
}

#undef CHECK_CUDA

} // namespace gs