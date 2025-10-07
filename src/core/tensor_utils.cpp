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

namespace gs::tensor {

    // ============= Creation Utilities =============

    Tensor diag(const Tensor& diagonal) {
        if (diagonal.ndim() != 1) {
            LOG_ERROR("diag requires 1D tensor");
            return Tensor();
        }

        size_t n = diagonal.numel();
        auto result = Tensor::zeros({n, n}, diagonal.device());

        if (diagonal.device() == Device::CUDA) {
            tensor_ops::launch_diag(diagonal.ptr<float>(), result.ptr<float>(), n, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* diag_data = diagonal.ptr<float>();
            float* mat_data = result.ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                mat_data[i * n + i] = diag_data[i];
            }
        }

        return result;
    }

    Tensor linspace(float start, float end, size_t steps) {
        if (steps == 0) {
            LOG_ERROR("Steps must be > 0");
            return Tensor();
        }

        if (steps == 1) {
            return Tensor::full({1}, start, Device::CUDA);
        }

        auto t = Tensor::empty({steps}, Device::CUDA);

        // Generate on CPU and copy
        std::vector<float> data(steps);
        float step = (end - start) / (steps - 1);
        for (size_t i = 0; i < steps; ++i) {
            data[i] = start + i * step;
        }

        CHECK_CUDA(cudaMemcpy(t.ptr<float>(), data.data(), steps * sizeof(float),
                              cudaMemcpyHostToDevice));

        return t;
    }

    // ============= Wrappers for Tensor Static Methods =============
    // These maintain the tensor:: namespace API style

    Tensor stack(std::vector<Tensor>&& tensors, int dim) {
        return Tensor::stack(tensors, dim);
    }

    Tensor cat(std::vector<Tensor>&& tensors, int dim) {
        return Tensor::cat(tensors, dim);
    }

    // ============= Like Operations =============

    Tensor zeros_like(const Tensor& other) {
        return Tensor::zeros(other.shape(), other.device(), other.dtype());
    }

    Tensor ones_like(const Tensor& other) {
        return Tensor::ones(other.shape(), other.device(), other.dtype());
    }

    Tensor ones_like(const Tensor& other, DataType dtype) {
        return Tensor::ones(other.shape(), other.device(), dtype);
    }

    Tensor rand_like(const Tensor& other) {
        return Tensor::rand(other.shape(), other.device(), other.dtype());
    }

    Tensor randn_like(const Tensor& other) {
        return Tensor::randn(other.shape(), other.device(), other.dtype());
    }

    // ============= Utility Functions =============

    bool check_valid(const Tensor& t, const std::string& name) {
        if (!t.is_valid()) {
            LOG_ERROR("Tensor '{}' is invalid", name);
            return false;
        }
        return true;
    }

    void assert_same_shape(const Tensor& a, const Tensor& b) {
        if (a.shape() != b.shape()) {
            std::string error_msg = "Shape mismatch: " + a.shape().str() + " vs " + b.shape().str();
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg);
        }
    }

    void assert_same_device(const Tensor& a, const Tensor& b) {
        if (a.device() != b.device()) {
            std::string error_msg = "Device mismatch";
            LOG_ERROR("{}", error_msg);
            throw TensorError(error_msg);
        }
    }

#undef CHECK_CUDA

} // namespace gs::tensor

// ============= SafeOps Implementation =============
namespace gs {

    SafeOps::Tensor SafeOps::divide(const Tensor& a, const Tensor& b, float epsilon) {
        // Safe division: a / (b + epsilon) to avoid division by zero
        auto safe_b = b.add(epsilon);
        return a.div(safe_b);
    }

    SafeOps::Tensor SafeOps::log(const Tensor& input, float epsilon) {
        // Safe log: clamp input to be at least epsilon
        auto safe_input = input.clamp_min(epsilon);
        return safe_input.log();
    }

    SafeOps::Tensor SafeOps::sqrt(const Tensor& input, float epsilon) {
        // Safe sqrt: clamp input to be non-negative
        auto safe_input = input.clamp_min(epsilon);
        return safe_input.sqrt();
    }

} // namespace gs

// ============= MemoryInfo Implementation =============
namespace gs {

    MemoryInfo MemoryInfo::cuda() {
        MemoryInfo info;

        size_t free_bytes, total_bytes;
        cudaMemGetInfo(&free_bytes, &total_bytes);

        info.free_bytes = free_bytes;
        info.total_bytes = total_bytes;
        info.allocated_bytes = total_bytes - free_bytes;
        info.device_id = 0; // Default to device 0

        return info;
    }

    MemoryInfo MemoryInfo::cpu() {
        MemoryInfo info;

        // For CPU, we can't easily get memory info portably
        // This is a placeholder implementation
        info.free_bytes = 0;
        info.total_bytes = 0;
        info.allocated_bytes = 0;
        info.device_id = -1; // CPU device

        return info;
    }

    void MemoryInfo::log() const {
        LOG_INFO("Memory Info - Device: {}, Allocated: {:.2f} MB, Free: {:.2f} MB, Total: {:.2f} MB",
                 device_id,
                 allocated_bytes / (1024.0 * 1024.0),
                 free_bytes / (1024.0 * 1024.0),
                 total_bytes / (1024.0 * 1024.0));
    }

} // namespace gs

// ============= Functional Operations Implementation =============
namespace gs::functional {

    Tensor map(const Tensor& input, std::function<float(float)> func) {
        auto result = Tensor::empty(input.shape(), input.device());

        if (input.device() == Device::CUDA) {
            // For CUDA, we need to copy to CPU, apply function, copy back
            auto cpu_input = input.to(Device::CPU);
            const float* src = cpu_input.ptr<float>();
            std::vector<float> dst_data(input.numel());

            for (size_t i = 0; i < input.numel(); ++i) {
                dst_data[i] = func(src[i]);
            }

            cudaMemcpy(result.ptr<float>(), dst_data.data(),
                       dst_data.size() * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            const float* src = input.ptr<float>();
            float* dst = result.ptr<float>();

            for (size_t i = 0; i < input.numel(); ++i) {
                dst[i] = func(src[i]);
            }
        }

        return result;
    }

    float reduce(const Tensor& input, float init, std::function<float(float, float)> func) {
        auto values = input.to_vector();
        float result = init;

        for (float val : values) {
            result = func(result, val);
        }

        return result;
    }

    Tensor filter(const Tensor& input, std::function<bool(float)> predicate) {
        auto result = Tensor::empty(input.shape(), input.device());

        if (input.device() == Device::CUDA) {
            auto cpu_input = input.to(Device::CPU);
            const float* src = cpu_input.ptr<float>();
            std::vector<float> dst_data(input.numel());

            for (size_t i = 0; i < input.numel(); ++i) {
                dst_data[i] = predicate(src[i]) ? 1.0f : 0.0f;
            }

            cudaMemcpy(result.ptr<float>(), dst_data.data(),
                       dst_data.size() * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            const float* src = input.ptr<float>();
            float* dst = result.ptr<float>();

            for (size_t i = 0; i < input.numel(); ++i) {
                dst[i] = predicate(src[i]) ? 1.0f : 0.0f;
            }
        }

        return result;
    }

} // namespace gs::functional