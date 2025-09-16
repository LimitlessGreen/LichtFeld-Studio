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

    float Tensor::sum() const {
        if (!is_valid() || numel() == 0) {
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
        if (!is_valid() || numel() == 0) {
            return 0.0f;
        }

        if (device_ == Device::CUDA) {
            float result = 0.0f;
            tensor_ops::launch_reduce_mean(ptr<float>(), &result, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
            return result;
        } else {
            return sum() / static_cast<float>(numel());
        }
    }

    float Tensor::min() const {
        if (!is_valid() || numel() == 0) {
            return 0.0f;
        }

        if (device_ == Device::CUDA) {
            float result = 0.0f;
            tensor_ops::launch_reduce_min(ptr<float>(), &result, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
            return result;
        } else {
            const float* data = ptr<float>();
            return *std::min_element(data, data + numel());
        }
    }

    float Tensor::max() const {
        if (!is_valid() || numel() == 0) {
            return 0.0f;
        }

        if (device_ == Device::CUDA) {
            float result = 0.0f;
            tensor_ops::launch_reduce_max(ptr<float>(), &result, numel(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
            return result;
        } else {
            const float* data = ptr<float>();
            return *std::max_element(data, data + numel());
        }
    }

    std::pair<float, float> Tensor::minmax() const {
        return {min(), max()};
    }

    float Tensor::std(float eps) const {
        if (!is_valid() || numel() == 0) {
            return 0.0f;
        }

        return std::sqrt(var() + eps); // Add eps before sqrt to avoid sqrt(0)
    }

    float Tensor::var(float eps) const {
        if (!is_valid() || numel() == 0) {
            return 0.0f;
        }

        float m = mean();

        // Compute variance
        auto values = to_vector();
        float sum_sq = 0.0f;
        for (float val : values) {
            float diff = val - m;
            sum_sq += diff * diff;
        }

        // Use population variance (divide by N, not N-1)
        // eps parameter is not used here - it's only used in std() before sqrt
        return sum_sq / static_cast<float>(numel());
    }

    float Tensor::norm(float p) const {
        if (!is_valid() || numel() == 0) {
            return 0.0f;
        }

        auto values = to_vector();

        if (p == 1.0f) {
            // L1 norm
            float sum = 0.0f;
            for (float val : values) {
                sum += std::abs(val);
            }
            return sum;
        } else if (p == 2.0f) {
            // L2 norm
            float sum_sq = 0.0f;
            for (float val : values) {
                sum_sq += val * val;
            }
            return std::sqrt(sum_sq);
        } else if (std::isinf(p)) {
            // Inf norm (max absolute value)
            float max_abs = 0.0f;
            for (float val : values) {
                max_abs = std::max(max_abs, std::abs(val));
            }
            return max_abs;
        } else {
            // General p-norm
            float sum = 0.0f;
            for (float val : values) {
                sum += std::pow(std::abs(val), p);
            }
            return std::pow(sum, 1.0f / p);
        }
    }

    float Tensor::item() const {
        if (!is_valid()) {
            LOG_ERROR("Cannot get item from invalid tensor");
            return 0.0f;
        }

        if (numel() != 1) {
            LOG_ERROR("item() only works for tensors with a single element, got {} elements", numel());
            return 0.0f;
        }

        if (device_ == Device::CUDA) {
            float result;
            CHECK_CUDA(cudaMemcpy(&result, ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost));
            return result;
        } else {
            return *ptr<float>();
        }
    }

#undef CHECK_CUDA

} // namespace gs
