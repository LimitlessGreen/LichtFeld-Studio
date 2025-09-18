/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
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

    // ============= Helper for unary operations =============
    template<typename Op>
    static void dispatch_unary(Tensor& t, void(*cuda_fn)(float*, size_t, cudaStream_t), Op cpu_op) {
        if (t.device() == Device::CUDA) {
            cuda_fn(t.ptr<float>(), t.numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = t.ptr<float>();
            for (size_t i = 0; i < t.numel(); ++i) {
                cpu_op(data[i]);
            }
        }
    }

    // ============= Math Operations =============

    #define DEFINE_UNARY_OP(name, cuda_fn, cpu_lambda) \
    Tensor Tensor::name() const { \
        if (!is_valid()) return Tensor(); \
        auto result = empty(shape_, device_, dtype_); \
        CHECK_CUDA(cudaMemcpy(result.ptr<float>(), ptr<float>(), bytes(), \
                             device_ == Device::CUDA ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToHost)); \
        dispatch_unary(result, tensor_ops::cuda_fn, cpu_lambda); \
        return result; \
    }

    DEFINE_UNARY_OP(abs, launch_abs, [](float& x) { x = std::abs(x); })
    DEFINE_UNARY_OP(sqrt, launch_sqrt, [](float& x) { x = std::sqrt(std::max(0.0f, x)); })
    DEFINE_UNARY_OP(exp, launch_exp, [](float& x) { x = std::exp(x); })
    DEFINE_UNARY_OP(log, launch_log, [](float& x) { x = std::log(std::max(1e-10f, x)); })
    DEFINE_UNARY_OP(sigmoid, launch_sigmoid, [](float& x) { x = 1.0f / (1.0f + std::exp(-x)); })
    DEFINE_UNARY_OP(relu, launch_relu, [](float& x) { x = std::max(0.0f, x); })

    #undef DEFINE_UNARY_OP

    Tensor Tensor::logit(float eps) const {
        if (!is_valid()) return Tensor();
        auto result = empty(shape_, device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_logit(ptr<float>(), result.ptr<float>(), numel(), eps, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                float x = std::clamp(src[i], eps, 1.0f - eps);
                dst[i] = std::log(x / (1.0f - x));
            }
        }
        return result;
    }

    Tensor Tensor::clamp(float min_val, float max_val) const {
        if (!is_valid()) return Tensor();
        auto result = empty(shape_, device_, dtype_);

        CHECK_CUDA(cudaMemcpy(result.ptr<float>(), ptr<float>(), bytes(),
                             device_ == Device::CUDA ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToHost));

        if (device_ == Device::CUDA) {
            tensor_ops::launch_clamp(result.ptr<float>(), min_val, max_val, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = std::clamp(dst[i], min_val, max_val);
            }
        }
        return result;
    }

    Tensor Tensor::clamp_min(float min_val) const {
        return clamp(min_val, std::numeric_limits<float>::max());
    }

    Tensor Tensor::clamp_max(float max_val) const {
        return clamp(std::numeric_limits<float>::lowest(), max_val);
    }

    Tensor Tensor::normalize(int dim, float eps) const {
        if (!is_valid()) return Tensor();
        float m = mean();
        float s = std();
        if (s < eps) s = eps;
        return (*this - m) / s;
    }

    // ============= Comparison Operations =============
    bool Tensor::has_nan() const {
        if (!is_valid() || numel() == 0) return false;
        auto values = to_vector();
        return std::any_of(values.begin(), values.end(), [](float v) { return std::isnan(v); });
    }

    bool Tensor::has_inf() const {
        if (!is_valid() || numel() == 0) return false;
        auto values = to_vector();
        return std::any_of(values.begin(), values.end(), [](float v) { return std::isinf(v); });
    }

    bool Tensor::all_close(const Tensor& other, float rtol, float atol) const {
        if (!is_valid() || !other.is_valid() || shape_ != other.shape_) return false;

        if (device_ != other.device_) {
            auto cpu_this = this->to(Device::CPU);
            auto cpu_other = other.to(Device::CPU);
            return cpu_this.all_close(cpu_other, rtol, atol);
        }

        auto values1 = to_vector();
        auto values2 = other.to_vector();

        for (size_t i = 0; i < values1.size(); ++i) {
            float diff = std::abs(values1[i] - values2[i]);
            float tol = atol + rtol * std::abs(values2[i]);
            if (diff > tol) return false;
        }
        return true;
    }

    // ============= Assertions =============
    Tensor& Tensor::assert_shape(TensorShape expected, const std::string& msg) {
        if (shape_ != expected) {
            std::string error_msg = msg.empty() ? "Shape assertion failed" : msg;
            throw TensorError(error_msg + ": expected " + expected.str() +
                                  ", got " + shape_.str(), this);
        }
        return *this;
    }

    Tensor& Tensor::assert_device(Device expected) {
        if (device_ != expected) {
            throw TensorError("Device assertion failed", this);
        }
        return *this;
    }

    Tensor& Tensor::assert_dtype(DataType expected) {
        if (dtype_ != expected) {
            throw TensorError("Dtype assertion failed", this);
        }
        return *this;
    }

    Tensor& Tensor::assert_finite() {
        if (has_nan() || has_inf()) {
            throw TensorError("Tensor contains NaN or Inf values", this);
        }
        return *this;
    }

#undef CHECK_CUDA

} // namespace gs