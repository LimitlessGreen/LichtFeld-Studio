/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Thrust headers for multinomial without replacement
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

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

    // Kernel for multinomial sampling with replacement
    __global__ void multinomial_with_replacement_kernel(const float* weights, int* samples,
                                                        unsigned long n, unsigned long num_samples,
                                                        float sum, unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_samples) return;

        curandState state;
        curand_init(seed + idx, 0, 0, &state);

        // Generate random value in [0, sum)
        float u = curand_uniform(&state) * sum;

        // Find the index using linear search
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

    // Kernel to generate random keys for each index (Gumbel-max trick)
    __global__ void generate_gumbel_keys_kernel(const float* weights, float* keys,
                                                unsigned long n, unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= n) return;

        curandState state;
        curand_init(seed + idx, 0, 0, &state);

        // Generate Gumbel noise: -log(-log(uniform))
        float u = curand_uniform(&state);
        // Clamp to avoid log(0)
        u = fmaxf(u, 1e-10f);
        u = fminf(u, 1.0f - 1e-10f);

        float gumbel = -logf(-logf(u));

        // Add log-weight
        float log_weight = logf(fmaxf(weights[idx], 1e-10f));
        keys[idx] = log_weight + gumbel;
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
        if (n == 0 || num_samples == 0) return;

        // First, compute sum of weights using Thrust
        auto weights_ptr = thrust::device_pointer_cast(weights);
        float sum = thrust::reduce(
            thrust::cuda::par.on(stream),
            weights_ptr, weights_ptr + n,
            0.0f,
            thrust::plus<float>()
        );

        if (sum <= 0) {
            // Invalid weights, fill with zeros
            cudaMemsetAsync(samples, 0, num_samples * sizeof(int), stream);
            return;
        }

        if (replacement) {
            // With replacement: simple multinomial sampling
            int block_size = 256;
            int grid_size = (num_samples + block_size - 1) / block_size;
            multinomial_with_replacement_kernel<<<grid_size, block_size, 0, stream>>>(
                weights, samples, n, num_samples, sum, seed);
        } else {
            // Without replacement: use Gumbel-max trick
            // 1. Generate Gumbel keys for each index
            // 2. Sort indices by keys (descending)
            // 3. Take first num_samples indices

            // Allocate temporary storage for keys and indices
            thrust::device_vector<float> keys(n);
            thrust::device_vector<int> indices(n);

            // Generate keys
            int block_size = 256;
            int grid_size = (n + block_size - 1) / block_size;
            generate_gumbel_keys_kernel<<<grid_size, block_size, 0, stream>>>(
                weights, thrust::raw_pointer_cast(keys.data()), n, seed);

            // Initialize indices [0, 1, 2, ..., n-1]
            thrust::sequence(
                thrust::cuda::par.on(stream),
                indices.begin(), indices.end()
            );

            // Sort indices by keys (descending order)
            thrust::sort_by_key(
                thrust::cuda::par.on(stream),
                keys.begin(), keys.end(),
                indices.begin(),
                thrust::greater<float>()
            );

            // Copy first num_samples indices to output
            // (num_samples is already capped by Tensor::multinomial for without replacement)
            thrust::copy_n(
                thrust::cuda::par.on(stream),
                indices.begin(),
                num_samples,
                thrust::device_pointer_cast(samples)
            );
        }
    }

} // namespace gs::tensor_ops