/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include "core/tensor.hpp"
#include "kernels/training_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <numeric>

namespace gs {

// Helper macro
#define CHECK_CUDA(x)                                                                          \
    do {                                                                                       \
        cudaError_t err = x;                                                                   \
        if (err != cudaSuccess) {                                                              \
            LOG_ERROR("CUDA error: {} at {}:{}", cudaGetErrorString(err), __FILE__, __LINE__); \
        }                                                                                      \
    } while (0)

    // ============= Basic Math Operations =============
    Tensor Tensor::add(float scalar) const {
        if (dtype_ != DataType::Float32) {
            LOG_ERROR("Add only implemented for float32 tensors");
            return Tensor();
        }

        Tensor result = clone();
        if (!result.is_valid()) {
            return result;
        }

        if (device_ == Device::CUDA) {
            gs::training::launch_add_scalar_to_tensor(
                result.ptr<float>(), scalar, numel(), 0);
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
        if (dtype_ != DataType::Float32) {
            LOG_ERROR("Mul only implemented for float32 tensors");
            return Tensor();
        }

        Tensor result = clone();
        if (!result.is_valid()) {
            return result;
        }

        if (device_ == Device::CUDA) {
            // Use a simple kernel for scalar multiplication
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
        if (std::abs(scalar) < 1e-8f) {
            LOG_ERROR("Division by near-zero scalar: {}", scalar);
            return Tensor();
        }
        return mul(1.0f / scalar);
    }

    Tensor Tensor::neg() const {
        return mul(-1.0f);
    }

    // Element-wise operations
    Tensor Tensor::add(const Tensor& other) const {
        if (shape_ != other.shape_) {
            LOG_ERROR("Shape mismatch for addition: {} vs {}",
                      shape_.str(), other.shape_.str());
            return Tensor();
        }

        if (dtype_ != other.dtype_) {
            LOG_ERROR("Dtype mismatch for addition");
            return Tensor();
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

    Tensor Tensor::sub(const Tensor& other) const {
        if (shape_ != other.shape_) {
            LOG_ERROR("Shape mismatch for subtraction: {} vs {}",
                      shape_.str(), other.shape_.str());
            return Tensor();
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

    Tensor Tensor::mul(const Tensor& other) const {
        if (shape_ != other.shape_) {
            LOG_ERROR("Shape mismatch for multiplication: {} vs {}",
                      shape_.str(), other.shape_.str());
            return Tensor();
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

    Tensor Tensor::div(const Tensor& other) const {
        if (shape_ != other.shape_) {
            LOG_ERROR("Shape mismatch for division: {} vs {}",
                      shape_.str(), other.shape_.str());
            return Tensor();
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

    // In-place operations
    Tensor& Tensor::add_(float scalar) {
        if (device_ == Device::CUDA) {
            gs::training::launch_add_scalar_to_tensor(ptr<float>(), scalar, numel(), 0);
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
        if (std::abs(scalar) < 1e-8f) {
            LOG_ERROR("Division by near-zero scalar: {}", scalar);
            return *this;
        }
        return mul_(1.0f / scalar);
    }

    // In-place tensor operations
    Tensor& Tensor::add_(const Tensor& other) {
        if (shape_ != other.shape_) {
            LOG_ERROR("Shape mismatch for in-place addition");
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
        if (shape_ != other.shape_) {
            LOG_ERROR("Shape mismatch for in-place subtraction");
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
        if (shape_ != other.shape_) {
            LOG_ERROR("Shape mismatch for in-place multiplication");
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
        if (shape_ != other.shape_) {
            LOG_ERROR("Shape mismatch for in-place division");
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

    // ============= Reduction Operations =============
    float Tensor::sum() const {
        if (dtype_ != DataType::Float32) {
            LOG_ERROR("Sum only implemented for float32");
            return 0.0f;
        }

        if (device_ == Device::CUDA) {
            float result = 0.0f;
            tensor_ops::launch_reduce_sum(ptr<float>(), &result, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
            return result;
        } else {
            const float* data = ptr<float>();
            return std::accumulate(data, data + numel(), 0.0f);
        }
    }

    float Tensor::mean() const {
        if (numel() == 0)
            return 0.0f;
        return sum() / static_cast<float>(numel());
    }

    float Tensor::min() const {
        auto values = to_vector();
        if (values.empty())
            return 0.0f;
        return *std::min_element(values.begin(), values.end());
    }

    float Tensor::max() const {
        auto values = to_vector();
        if (values.empty())
            return 0.0f;
        return *std::max_element(values.begin(), values.end());
    }

    float Tensor::std(float eps) const {
        float m = mean();
        auto diff = (*this - m);
        return std::sqrt((diff * diff).mean() + eps);
    }

    float Tensor::var(float eps) const {
        float m = mean();
        auto diff = (*this - m);
        return (diff * diff).mean() + eps;
    }

    float Tensor::norm(float p) const {
        if (p == 2.0f) {
            auto squared = (*this) * (*this);
            return std::sqrt(squared.sum());
        } else if (p == 1.0f) {
            return abs().sum();
        } else {
            LOG_ERROR("Norm only implemented for p=1 or p=2");
            return 0.0f;
        }
    }

    float Tensor::item() const {
        if (numel() != 1) {
            LOG_ERROR("item() only works on single-element tensors");
            return 0.0f;
        }

        float value;
        if (device_ == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(&value, data_, sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            value = *ptr<float>();
        }
        return value;
    }

    std::pair<float, float> Tensor::minmax() const {
        auto values = to_vector();
        if (values.empty())
            return {0, 0};
        auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
        return {*min_it, *max_it};
    }

    // ============= Math Functions =============
    Tensor Tensor::abs() const {
        Tensor result = clone();
        if (device_ == Device::CUDA) {
            tensor_ops::launch_abs(result.ptr<float>(), numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] = std::abs(data[i]);
            }
        }
        return result;
    }

    Tensor Tensor::sqrt() const {
        Tensor result = clone();
        if (device_ == Device::CUDA) {
            tensor_ops::launch_sqrt(result.ptr<float>(), numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] = std::sqrt(std::max(0.0f, data[i]));
            }
        }
        return result;
    }

    Tensor Tensor::exp() const {
        Tensor result = clone();
        if (device_ == Device::CUDA) {
            tensor_ops::launch_exp(result.ptr<float>(), numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] = std::exp(data[i]);
            }
        }
        return result;
    }

    Tensor Tensor::log() const {
        Tensor result = clone();
        if (device_ == Device::CUDA) {
            tensor_ops::launch_log(result.ptr<float>(), numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] = std::log(std::max(1e-8f, data[i]));
            }
        }
        return result;
    }

    Tensor Tensor::sigmoid() const {
        Tensor result = clone();
        if (device_ == Device::CUDA) {
            tensor_ops::launch_sigmoid(result.ptr<float>(), numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] = 1.0f / (1.0f + std::exp(-data[i]));
            }
        }
        return result;
    }

    Tensor Tensor::relu() const {
        return clamp_min(0.0f);
    }

    Tensor Tensor::clamp(float min_val, float max_val) const {
        Tensor result = clone();
        if (device_ == Device::CUDA) {
            tensor_ops::launch_clamp(result.ptr<float>(), min_val, max_val, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            float* data = result.ptr<float>();
            for (size_t i = 0; i < numel(); ++i) {
                data[i] = std::clamp(data[i], min_val, max_val);
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
        // For now, simple implementation for whole tensor
        float m = mean();
        float s = std(eps);
        return (*this - m) / s;
    }

    // ============= Validation & Assertions =============
    Tensor& Tensor::assert_shape(TensorShape expected, const std::string& msg) {
        if (shape_ != expected) {
            std::string error = msg.empty() ? "Shape mismatch" : msg;
            LOG_ERROR("{}: expected {}, got {}", error, expected.str(), shape_.str());
            throw TensorError(error, this);
        }
        return *this;
    }

    Tensor& Tensor::assert_device(Device expected) {
        if (device_ != expected) {
            LOG_ERROR("Device mismatch: expected {}, got {}",
                      device_name(expected), device_name(device_));
            throw TensorError("Device mismatch", this);
        }
        return *this;
    }

    Tensor& Tensor::assert_dtype(DataType expected) {
        if (dtype_ != expected) {
            LOG_ERROR("Dtype mismatch: expected {}, got {}",
                      dtype_name(expected), dtype_name(dtype_));
            throw TensorError("Dtype mismatch", this);
        }
        return *this;
    }

    Tensor& Tensor::assert_finite() {
        if (has_nan()) {
            LOG_ERROR("Tensor contains NaN values");
            throw TensorError("NaN detected", this);
        }
        if (has_inf()) {
            LOG_ERROR("Tensor contains Inf values");
            throw TensorError("Inf detected", this);
        }
        return *this;
    }

    // ============= Quick Checks =============
    bool Tensor::has_nan() const {
        auto values = debug_values(std::min(size_t(1000), numel()));
        return std::any_of(values.begin(), values.end(),
                           [](float v) { return std::isnan(v); });
    }

    bool Tensor::has_inf() const {
        auto values = debug_values(std::min(size_t(1000), numel()));
        return std::any_of(values.begin(), values.end(),
                           [](float v) { return std::isinf(v); });
    }

    bool Tensor::all_close(const Tensor& other, float rtol, float atol) const {
        if (shape_ != other.shape_)
            return false;

        auto vals1 = to_vector();
        auto vals2 = other.to_vector();

        for (size_t i = 0; i < vals1.size(); ++i) {
            float diff = std::abs(vals1[i] - vals2[i]);
            float tol = atol + rtol * std::abs(vals2[i]);
            if (diff > tol)
                return false;
        }
        return true;
    }

    // ============= Error Handling =============
    std::expected<Tensor, std::string> Tensor::try_reshape(TensorShape new_shape) const {
        if (new_shape.elements() != numel()) {
            return std::unexpected(
                std::format("Cannot reshape {} to {} (element count mismatch)",
                            shape_.str(), new_shape.str()));
        }
        return view(new_shape);
    }

    // ============= Batch Operations =============
    std::vector<Tensor> Tensor::split_batch(const Tensor& t, size_t batch_size) {
        std::vector<Tensor> batches;
        size_t total = t.shape()[0];
        for (size_t i = 0; i < total; i += batch_size) {
            size_t end = std::min(i + batch_size, total);
            batches.push_back(t.slice(0, i, end));
        }
        return batches;
    }

    // ============= Template Implementation =============
    template <typename Func>
    auto Tensor::timed(const std::string& name, Func func) -> decltype(func(*this)) {
        if (profiling_enabled_) {
            TensorTimer timer(name);
            return func(*this);
        }
        return func(*this);
    }

#undef CHECK_CUDA

} // namespace gs
