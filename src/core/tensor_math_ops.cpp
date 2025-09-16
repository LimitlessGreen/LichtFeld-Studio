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

    // ============= Math Operations =============

    Tensor Tensor::abs() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = empty(shape_, device_, dtype_);

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.ptr<float>(), ptr<float>(), bytes(), cudaMemcpyDeviceToDevice));
            tensor_ops::launch_abs(result.ptr<float>(), numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = std::abs(src[i]);
            }
        }

        return result;
    }

    Tensor Tensor::sqrt() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = empty(shape_, device_, dtype_);

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.ptr<float>(), ptr<float>(), bytes(), cudaMemcpyDeviceToDevice));
            tensor_ops::launch_sqrt(result.ptr<float>(), numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = std::sqrt(std::max(0.0f, src[i])); // Avoid NaN for negative values
            }
        }

        return result;
    }

    Tensor Tensor::exp() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = empty(shape_, device_, dtype_);

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.ptr<float>(), ptr<float>(), bytes(), cudaMemcpyDeviceToDevice));
            tensor_ops::launch_exp(result.ptr<float>(), numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = std::exp(src[i]);
            }
        }

        return result;
    }

    Tensor Tensor::log() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = empty(shape_, device_, dtype_);

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.ptr<float>(), ptr<float>(), bytes(), cudaMemcpyDeviceToDevice));
            tensor_ops::launch_log(result.ptr<float>(), numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = std::log(std::max(1e-10f, src[i])); // Avoid NaN for negative/zero values
            }
        }

        return result;
    }

    Tensor Tensor::sigmoid() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = empty(shape_, device_, dtype_);

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.ptr<float>(), ptr<float>(), bytes(), cudaMemcpyDeviceToDevice));
            tensor_ops::launch_sigmoid(result.ptr<float>(), numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
            }
        }

        return result;
    }

    Tensor Tensor::relu() const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = empty(shape_, device_, dtype_);

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.ptr<float>(), ptr<float>(), bytes(), cudaMemcpyDeviceToDevice));
            tensor_ops::launch_relu(result.ptr<float>(), numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = std::max(0.0f, src[i]);
            }
        }

        return result;
    }

    Tensor Tensor::clamp(float min_val, float max_val) const {
        if (!is_valid()) {
            return Tensor();
        }

        auto result = empty(shape_, device_, dtype_);

        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(result.ptr<float>(), ptr<float>(), bytes(), cudaMemcpyDeviceToDevice));
            tensor_ops::launch_clamp(result.ptr<float>(), min_val, max_val, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = std::clamp(src[i], min_val, max_val);
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

    // ============= Normalization =============

    Tensor Tensor::normalize(int dim, float eps) const {
        if (!is_valid()) {
            return Tensor();
        }

        // For simplicity, normalize over all elements if dim == -1
        float m = mean();

        // Calculate standard deviation
        float s = std(); // Don't pass eps here, std() will handle it internally

        // If std is too small, use eps to avoid divide by zero
        if (s < eps) {
            s = eps;
        }

        return (*this - m) / s;
    }

    // ============= Comparison Operations =============

    bool Tensor::has_nan() const {
        if (!is_valid() || numel() == 0) {
            return false;
        }

        auto values = to_vector();
        for (float val : values) {
            if (std::isnan(val)) {
                return true;
            }
        }
        return false;
    }

    bool Tensor::has_inf() const {
        if (!is_valid() || numel() == 0) {
            return false;
        }

        auto values = to_vector();
        for (float val : values) {
            if (std::isinf(val)) {
                return true;
            }
        }
        return false;
    }

    bool Tensor::all_close(const Tensor& other, float rtol, float atol) const {
        if (!is_valid() || !other.is_valid()) {
            return false;
        }

        if (shape_ != other.shape_) {
            return false;
        }

        if (device_ != other.device_) {
            // Compare across devices by copying to CPU
            auto cpu_this = this->to(Device::CPU);
            auto cpu_other = other.to(Device::CPU);
            return cpu_this.all_close(cpu_other, rtol, atol);
        }

        auto values1 = to_vector();
        auto values2 = other.to_vector();

        for (size_t i = 0; i < values1.size(); ++i) {
            float diff = std::abs(values1[i] - values2[i]);
            float tol = atol + rtol * std::abs(values2[i]);
            if (diff > tol) {
                return false;
            }
        }

        return true;
    }

    // ============= Assertions =============

    Tensor& Tensor::assert_shape(TensorShape expected, const std::string& msg) {
        if (shape_ != expected) {
            std::string error_msg = msg.empty() ? "Shape assertion failed" : msg;
            throw TensorError(error_msg + ": expected " + expected.str() +
                                  ", got " + shape_.str(),
                              this);
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
