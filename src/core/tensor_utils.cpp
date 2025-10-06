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
    // These functions are in the tensor namespace, not Tensor class
    // They delegate to Tensor static methods

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

    // ============= Stack/Concatenate Operations =============
    Tensor stack(std::vector<Tensor>&& tensors, int dim) {
        if (tensors.empty()) {
            LOG_ERROR("Cannot stack empty vector of tensors");
            return Tensor();
        }

        // Check all shapes are the same
        const auto& first_shape = tensors[0].shape();
        const auto first_device = tensors[0].device();
        const auto first_dtype = tensors[0].dtype();

        for (size_t i = 1; i < tensors.size(); ++i) {
            if (tensors[i].shape() != first_shape) {
                LOG_ERROR("All tensors must have the same shape for stack");
                return Tensor();
            }
            if (tensors[i].device() != first_device) {
                LOG_ERROR("All tensors must be on the same device");
                return Tensor();
            }
            if (tensors[i].dtype() != first_dtype) {
                LOG_ERROR("All tensors must have the same dtype");
                return Tensor();
            }
        }

        // Create new shape with added dimension
        std::vector<size_t> new_dims = first_shape.dims();

        // Adjust dim to be positive
        if (dim < 0) {
            dim = first_shape.rank() + dim + 1;
        }

        if (dim < 0 || dim > static_cast<int>(first_shape.rank())) {
            LOG_ERROR("Invalid dimension for stack: {}", dim);
            return Tensor();
        }

        new_dims.insert(new_dims.begin() + dim, tensors.size());

        // Create result tensor
        auto result = Tensor::empty(TensorShape(new_dims), first_device);

        // Copy tensors
        size_t elements_per_tensor = first_shape.elements();
        size_t bytes_per_tensor = elements_per_tensor * dtype_size(first_dtype);

        if (dim == 0) {
            // Simple case: stack along first dimension
            for (size_t i = 0; i < tensors.size(); ++i) {
                void* dst = static_cast<char*>(result.raw_ptr()) + i * bytes_per_tensor;
                if (first_device == Device::CUDA) {
                    CHECK_CUDA(cudaMemcpy(dst, tensors[i].raw_ptr(), bytes_per_tensor,
                                          cudaMemcpyDeviceToDevice));
                } else {
                    std::memcpy(dst, tensors[i].raw_ptr(), bytes_per_tensor);
                }
            }
        } else {
            // More complex stacking - would need proper implementation
            LOG_ERROR("Stack along dimension {} not fully implemented", dim);
            return Tensor();
        }

        return result;
    }

    Tensor cat(std::vector<Tensor>&& tensors, int dim) {
        if (tensors.empty()) {
            LOG_ERROR("Cannot concatenate empty vector of tensors");
            return Tensor();
        }

        if (tensors.size() == 1) {
            return std::move(tensors[0]);
        }

        // For now, only implement dim=0
        if (dim != 0) {
            LOG_ERROR("Concatenation only implemented for dim=0");
            return Tensor();
        }

        // Check shapes (all dimensions except dim must match)
        const auto& first_shape = tensors[0].shape();
        const auto first_device = tensors[0].device();
        const auto first_dtype = tensors[0].dtype();

        size_t total_size_along_dim = first_shape[0];

        for (size_t i = 1; i < tensors.size(); ++i) {
            const auto& shape = tensors[i].shape();

            if (shape.rank() != first_shape.rank()) {
                LOG_ERROR("All tensors must have the same number of dimensions");
                return Tensor();
            }

            for (size_t d = 1; d < shape.rank(); ++d) {
                if (shape[d] != first_shape[d]) {
                    LOG_ERROR("All dimensions except dim={} must match", dim);
                    return Tensor();
                }
            }

            if (tensors[i].device() != first_device) {
                LOG_ERROR("All tensors must be on the same device");
                return Tensor();
            }

            if (tensors[i].dtype() != first_dtype) {
                LOG_ERROR("All tensors must have the same dtype");
                return Tensor();
            }

            total_size_along_dim += shape[0];
        }

        // Create result shape
        std::vector<size_t> result_dims = first_shape.dims();
        result_dims[0] = total_size_along_dim;

        // Create result tensor
        auto result = Tensor::empty(TensorShape(result_dims), first_device);

        // Copy data
        size_t offset = 0;
        for (auto& t : tensors) {
            size_t bytes = t.bytes();
            void* dst = static_cast<char*>(result.raw_ptr()) + offset;

            if (first_device == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(dst, t.raw_ptr(), bytes, cudaMemcpyDeviceToDevice));
            } else {
                std::memcpy(dst, t.raw_ptr(), bytes);
            }

            offset += bytes;
        }

        return result;
    }

    // ============= Like Operations =============
    Tensor zeros_like(const Tensor& other) {
        return Tensor::zeros(other.shape(), other.device(), other.dtype());
    }

    Tensor ones_like(const Tensor& other) {
        return Tensor::ones(other.shape(), other.device(), other.dtype());
    }

    // NEW: ones_like with dtype parameter
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
            LOG_ERROR("Shape mismatch: {} vs {}", a.shape().str(), b.shape().str());
            throw std::runtime_error("Shape mismatch");
        }
    }

    void assert_same_device(const Tensor& a, const Tensor& b) {
        if (a.device() != b.device()) {
            LOG_ERROR("Device mismatch");
            throw std::runtime_error("Device mismatch");
        }
    }

#undef CHECK_CUDA

} // namespace gs::tensor

// ============= SafeOps Implementation =============
namespace gs {

    SafeOps::Tensor SafeOps::divide(const Tensor& a, const Tensor& b, float epsilon) {
        // Safe division: add epsilon to denominator to avoid division by zero
        auto safe_b = b.abs().add(epsilon);
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