/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include "core/tensor_broadcast.hpp"
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
Tensor Tensor::unary(UnaryOp op, const UnaryArgs& args) const {
    if (!is_valid()) return {};

    // Special handling for operations that need parameters
    if (op == UnaryOp::Normalize) {
        int dim = -1;
        float eps = 1e-12f;
        if (auto* i = std::get_if<int>(&args.args)) {
            dim = *i;
        } else if (auto* f = std::get_if<float>(&args.args)) {
            eps = *f;
        }
        return normalize(dim, eps);
    }

    if (op == UnaryOp::Logit) {
        float eps = 1e-7f;
        if (auto* f = std::get_if<float>(&args.args)) {
            eps = *f;
        }
        return logit(eps);
    }

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
Tensor Tensor::reduce(ReduceOp op, const ReduceArgs& args) const {
    if (!is_valid()) return {};

    // Special handling for norm - convert to tensor
    if (op == ReduceOp::Norm) {
        float p = 2.0f;
        if (auto* f = std::get_if<float>(&args.args)) {
            p = *f;
        }
        // Create a scalar tensor from the float result
        float norm_val = norm(p);
        auto result = full({1}, norm_val, device_, dtype_);
        return result;
    }

    // Resolve axes
    std::vector<int> resolved_axes;
    if (args.axes.empty()) {
        // Reduce all dimensions
        resolved_axes.reserve(shape_.rank());
        for (int i = 0; i < static_cast<int>(shape_.rank()); ++i) {
            resolved_axes.push_back(i);
        }
    } else {
        resolved_axes.reserve(args.axes.size());
        for (int ax : args.axes) {
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
        } else if (args.keepdim) {
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
                                    args.keepdim, op, dtype_, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
    } else {
        // CPU implementation - simplified for now
        // Full implementation would handle all axes properly
        if (resolved_axes.size() == shape_.rank() && !args.keepdim) {
            // Reducing to scalar
            float val = 0.0f;

            switch (op) {
                case ReduceOp::Sum: {
                    const float* data = ptr<float>();
                    val = std::accumulate(data, data + numel(), 0.0f);
                    break;
                }
                case ReduceOp::Mean: {
                    const float* data = ptr<float>();
                    val = std::accumulate(data, data + numel(), 0.0f) / numel();
                    break;
                }
                case ReduceOp::Max: {
                    const float* data = ptr<float>();
                    val = *std::max_element(data, data + numel());
                    break;
                }
                case ReduceOp::Min: {
                    const float* data = ptr<float>();
                    val = *std::min_element(data, data + numel());
                    break;
                }
                case ReduceOp::Prod: {
                    const float* data = ptr<float>();
                    val = std::accumulate(data, data + numel(), 1.0f, std::multiplies<float>());
                    break;
                }
                case ReduceOp::Std:
                case ReduceOp::Var: {
                    const float* data = ptr<float>();
                    float mean = std::accumulate(data, data + numel(), 0.0f) / numel();
                    float variance = 0.0f;
                    for (size_t i = 0; i < numel(); ++i) {
                        float diff = data[i] - mean;
                        variance += diff * diff;
                    }
                    variance /= numel();
                    val = (op == ReduceOp::Std) ? std::sqrt(variance) : variance;
                    break;
                }
                case ReduceOp::CountNonzero: {
                    if (dtype_ == DataType::Bool) {
                        const unsigned char* data = ptr<unsigned char>();
                        val = static_cast<float>(std::count(data, data + numel(), 1));
                    } else {
                        const float* data = ptr<float>();
                        val = static_cast<float>(std::count_if(data, data + numel(),
                                                               [](float x) { return x != 0.0f; }));
                    }
                    break;
                }
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

            // Broadcast all to common shape
            auto common_shape = broadcast_shape(b.shape());
            common_shape = TensorShape(broadcast::shape(common_shape.dims(), c.shape().dims()));

            if (common_shape.elements() == 0) {
                LOG_ERROR("Cannot broadcast shapes for where");
                return {};
            }

            auto cond_bcast = broadcast_to(common_shape);
            auto x_bcast = b.broadcast_to(common_shape);
            auto y_bcast = c.broadcast_to(common_shape);

            auto result = empty(common_shape, b.device(), b.dtype());

            if (device_ == Device::CUDA) {
                tensor_ops::launch_where(
                    cond_bcast.ptr<unsigned char>(),
                    x_bcast.ptr<float>(), y_bcast.ptr<float>(), result.ptr<float>(),
                    cond_bcast.shape().dims().data(), x_bcast.shape().dims().data(),
                    y_bcast.shape().dims().data(), result.shape().dims().data(),
                    cond_bcast.shape().rank(), x_bcast.shape().rank(),
                    y_bcast.shape().rank(), result.shape().rank(),
                    result.numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                const unsigned char* cond_data = cond_bcast.ptr<unsigned char>();
                const float* x_data = x_bcast.ptr<float>();
                const float* y_data = y_bcast.ptr<float>();
                float* r_data = result.ptr<float>();

                #pragma omp parallel for if(result.numel() > 1024)
                for (size_t i = 0; i < result.numel(); ++i) {
                    r_data[i] = cond_data[i] ? x_data[i] : y_data[i];
                }
            }

            return result;
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

                const float* a_data = a_bcast.ptr<float>();
                const float* b_data = b_bcast.ptr<float>();
                const float* c_data = c_bcast.ptr<float>();
                float* r_data = result.ptr<float>();

                #pragma omp parallel for if(result.numel() > 1024)
                for (size_t i = 0; i < result.numel(); ++i) {
                    r_data[i] = a_data[i] * b_data[i] + c_data[i];
                }
            }

            return result;
        }

        case TernaryOp::Clamp: {
            // clamp(x, min, max) - this is x, b is min, c is max
            if (b.numel() != 1 || c.numel() != 1) {
                LOG_ERROR("Clamp min/max must be scalars");
                return {};
            }

            float min_val = b.item();
            float max_val = c.item();

            auto result = clone();
            if (numel() == 0) return result;

            if (device_ == Device::CUDA) {
                tensor_ops::launch_clamp(result.ptr<float>(), min_val, max_val, result.numel(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                float* data = result.ptr<float>();
                #pragma omp parallel for if(numel() > 1024)
                for (size_t i = 0; i < result.numel(); ++i) {
                    data[i] = std::min(std::max(data[i], min_val), max_val);
                }
            }

            return result;
        }

        default:
            LOG_ERROR("Unknown ternary operation");
            return {};
    }
}

// ============= Special Operations (that can't be fully unified) =============

Tensor Tensor::normalize(int dim, float eps) const {
    if (!is_valid()) return {};

    if (dim == -1) {
        // Normalize entire tensor
        auto m = mean();
        auto s = std().add(eps);
        return sub(m).div(s);
    }

    // Per-dimension normalization
    std::vector<int> axes = {dim};
    auto m = reduce(ReduceOp::Mean, {axes, true});
    auto s = reduce(ReduceOp::Std, {axes, true}).add(eps);
    return sub(m).div(s);
}

Tensor Tensor::logit(float eps) const {
    if (!is_valid()) return {};

    // logit(x) = log(x / (1 - x))
    // But we need to clamp x to [eps, 1-eps] first
    auto x_clamped = clamp(eps, 1.0f - eps);
    auto one_minus_x = full(shape_, 1.0f, device_, dtype_).sub(x_clamped);
    return x_clamped.div(one_minus_x).log();
}

float Tensor::norm(float p) const {
    if (!is_valid()) return 0.0f;

    if (p == 2.0f) {
        auto sq = mul(*this);  // square
        return std::sqrt(sq.sum_scalar());
    } else if (p == 1.0f) {
        return abs().sum_scalar();
    } else if (std::isinf(p)) {
        return abs().max_scalar();
    } else {
        // General p-norm
        auto powered = abs().pow(p);
        return std::pow(powered.sum_scalar(), 1.0f / p);
    }
}

float Tensor::item() const {
    if (numel() != 1) {
        LOG_ERROR("item() only works for single-element tensors");
        return 0.0f;
    }

    if (device_ == Device::CUDA) {
        float result;
        cudaMemcpy(&result, ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    } else {
        return *ptr<float>();
    }
}

size_t Tensor::count_nonzero() const {
    if (!is_valid() || numel() == 0) return 0;

    auto result = reduce(ReduceOp::CountNonzero, {});
    if (!result.is_valid()) return 0;

    if (result.dtype() == DataType::Int64) {
        int64_t count;
        if (result.device() == Device::CUDA) {
            cudaMemcpy(&count, result.ptr<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost);
        } else {
            count = *result.ptr<int64_t>();
        }
        return static_cast<size_t>(count);
    }

    return static_cast<size_t>(result.item());
}

// ============= Static factory operations =============
Tensor Tensor::cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        LOG_ERROR("Cannot cat empty vector of tensors");
        return {};
    }

    if (tensors.size() == 1) {
        return tensors[0].clone();
    }

    // Use the first tensor to start concatenation
    Tensor result = tensors[0].clone();
    for (size_t i = 1; i < tensors.size(); ++i) {
        MovementArgs args;
        // Store pointer to tensor and dimension
        args.args = std::pair<void*, int>{
            const_cast<void*>(static_cast<const void*>(&tensors[i])),
            dim
        };
        result = result.movement(MovementOp::Cat, args);
    }

    return result;
}

Tensor Tensor::stack(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        LOG_ERROR("Cannot stack empty vector of tensors");
        return {};
    }

    // Stack is like cat but adds a new dimension
    // First unsqueeze all tensors along dim, then cat
    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());

    for (const auto& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(dim));
    }

    return cat(unsqueezed, dim);
}

// ============= Where static method =============
Tensor Tensor::where(const Tensor& condition, const Tensor& x, const Tensor& y) {
    return condition.ternary(x, y, TernaryOp::Where);
}

#undef CHECK_CUDA

} // namespace gs