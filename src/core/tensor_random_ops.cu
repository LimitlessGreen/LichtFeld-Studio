/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace gs::tensor_ops {

    // ============= Random Operations Kernels =============

    // Uniform random generation
    __global__ void uniform_kernel(float* data, size_t n, float low, float high,
                                   unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            curandState state;
            // Use different seed per thread by adding thread index to base seed
            // This ensures each thread gets a different random sequence
            curand_init(seed + idx, 0, 0, &state);
            float val = curand_uniform(&state);
            data[idx] = val * (high - low) + low;
        }
    }

    // Normal random generation
    __global__ void normal_kernel(float* data, size_t n, float mean, float std,
                                  unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            curandState state;
            // Use different seed per thread by adding thread index to base seed
            curand_init(seed + idx, 0, 0, &state);
            data[idx] = curand_normal(&state) * std + mean;
        }
    }

    // Bernoulli random generation
    __global__ void bernoulli_kernel(float* data, size_t n, float p,
                                     unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            curandState state;
            // Use different seed per thread by adding thread index to base seed
            curand_init(seed + idx, 0, 0, &state);
            float val = curand_uniform(&state);
            data[idx] = (val < p) ? 1.0f : 0.0f;
        }
    }

    __global__ void randint_kernel(int* data, size_t n, int low, int high,
                                   unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            curandState state;
            curand_init(seed + idx, 0, 0, &state);

            // Generate uniform [0, 1) and scale to [low, high)
            float val = curand_uniform(&state);
            int range = high - low;

            // Properly scale to integer range [low, high)
            int result = low + static_cast<int>(val * range);

            // Ensure we're within bounds
            if (result >= high) {
                result = high - 1;
            }
            if (result < low) {
                result = low;
            }

            data[idx] = result;
        }
    }

    __global__ void multinomial_kernel(const float* weights, int* samples,
                                       unsigned long n, unsigned long num_samples,
                                       unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_samples) return;

        curandState state;
        curand_init(seed + idx, 0, 0, &state);

        // Calculate sum of weights (this could be optimized with parallel reduction)
        float sum = 0.0f;
        for (unsigned long i = 0; i < n; ++i) {
            sum += weights[i];
        }

        if (sum <= 0) {
            samples[idx] = 0;
            return;
        }

        // Generate random value
        float u = curand_uniform(&state) * sum;

        // Find the index using linear search (could be optimized with binary search)
        float cumsum = 0.0f;
        for (unsigned long i = 0; i < n; ++i) {
            cumsum += weights[i];
            if (u <= cumsum) {
                samples[idx] = i;
                return;
            }
        }

        // Fallback (shouldn't happen)
        samples[idx] = n - 1;
    }

    // ============= Launch Functions =============

    void launch_uniform(float* data, size_t n, float low, float high,
                        unsigned long long seed, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        uniform_kernel<<<grid_size, block_size, 0, stream>>>(data, n, low, high, seed);
    }

    void launch_normal(float* data, size_t n, float mean, float std,
                       unsigned long long seed, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        normal_kernel<<<grid_size, block_size, 0, stream>>>(data, n, mean, std, seed);
    }

    void launch_bernoulli(float* data, size_t n, float p,
                          unsigned long long seed, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        bernoulli_kernel<<<grid_size, block_size, 0, stream>>>(data, n, p, seed);
    }

    void launch_randint(int* data, size_t n, int low, int high,
                        unsigned long long seed, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        randint_kernel<<<grid_size, block_size, 0, stream>>>(data, n, low, high, seed);
    }

    void launch_multinomial(const float* weights, int* samples,
                             unsigned long n, unsigned long num_samples, bool replacement,
                             unsigned long long seed, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (num_samples + block_size - 1) / block_size;

        if (!replacement && num_samples > 1) {
            // Without replacement is more complex - would need to track selected indices
            // For now, just use with replacement
        }

        multinomial_kernel<<<grid_size, block_size, 0, stream>>>(
            weights, samples, n, num_samples, seed);
    }

} // namespace gs::tensor_ops