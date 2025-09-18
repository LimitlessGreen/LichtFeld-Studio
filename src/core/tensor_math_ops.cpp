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

    // ============= Math Operations Implementation =============

    Tensor Tensor::abs() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = clone();
        if (result.numel() == 0) {
            return result;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_abs(result.ptr<float>(), result.numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < result.numel(); ++i) {
                data[i] = std::abs(data[i]);
            }
        }

        return result;
    }

    Tensor Tensor::sqrt() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = clone();
        if (result.numel() == 0) {
            return result;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_sqrt(result.ptr<float>(), result.numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < result.numel(); ++i) {
                data[i] = std::sqrt(std::max(0.0f, data[i]));
            }
        }

        return result;
    }

    Tensor Tensor::exp() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = clone();
        if (result.numel() == 0) {
            return result;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_exp(result.ptr<float>(), result.numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < result.numel(); ++i) {
                data[i] = std::exp(data[i]);
            }
        }

        return result;
    }

    Tensor Tensor::log() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = clone();
        if (result.numel() == 0) {
            return result;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_log(result.ptr<float>(), result.numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < result.numel(); ++i) {
                data[i] = std::log(std::max(1e-10f, data[i]));
            }
        }

        return result;
    }

    Tensor Tensor::sigmoid() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = clone();
        if (result.numel() == 0) {
            return result;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_sigmoid(result.ptr<float>(), result.numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < result.numel(); ++i) {
                data[i] = 1.0f / (1.0f + std::exp(-data[i]));
            }
        }

        return result;
    }

    Tensor Tensor::logit(float eps) const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = empty(shape_, device_, dtype_);
        if (result.numel() == 0) {
            return result;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_logit(ptr<float>(), result.ptr<float>(), result.numel(), eps, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < result.numel(); ++i) {
                float x = src[i];
                x = std::max(std::min(x, 1.0f - eps), eps);
                dst[i] = std::log(x / (1.0f - x));
            }
        }

        return result;
    }

    Tensor Tensor::relu() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = clone();
        if (result.numel() == 0) {
            return result;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_relu(result.ptr<float>(), result.numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < result.numel(); ++i) {
                data[i] = std::max(0.0f, data[i]);
            }
        }

        return result;
    }

    Tensor Tensor::clamp(float min_val, float max_val) const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = clone();
        if (result.numel() == 0) {
            return result;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_clamp(result.ptr<float>(), min_val, max_val, result.numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < result.numel(); ++i) {
                data[i] = std::min(std::max(data[i], min_val), max_val);
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

    // ============= Deep Learning Operations =============

    Tensor Tensor::normalize(int dim, float eps) const {
        if (!is_valid()) {
            return Tensor();
        }

        // For simplicity, normalize over the entire tensor if dim == -1
        if (dim == -1) {
            float m = mean();
            float s = std();

            // result = (x - mean) / (std + eps)
            auto result = sub(m);
            result = result.div(s + eps);

            return result;
        }

        // More complex per-dimension normalization would go here
        LOG_WARN("Per-dimension normalization not fully implemented");
        return clone();
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

        if (shape_ != other.shape_ || device_ != other.device_) {
            return false;
        }

        auto a_values = to_vector();
        auto b_values = other.to_vector();

        for (size_t i = 0; i < a_values.size(); ++i) {
            float diff = std::abs(a_values[i] - b_values[i]);
            float tol = atol + rtol * std::abs(b_values[i]);
            if (diff > tol) {
                return false;
            }
        }

        return true;
    }

#undef CHECK_CUDA

} // namespace gs