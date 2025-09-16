/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include <algorithm> // For std::min_element, std::max_element, std::min
#include <cfloat>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector> // For std::vector

namespace gs::tensor_ops {

    // ============= Scalar Operations Kernels =============

    __global__ void scalar_add_kernel(float* data, float scalar, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] += scalar;
        }
    }

    __global__ void scalar_sub_kernel(float* data, float scalar, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] -= scalar;
        }
    }

    __global__ void scalar_mul_kernel(float* data, float scalar, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] *= scalar;
        }
    }

    __global__ void scalar_div_kernel(float* data, float scalar, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] /= scalar;
        }
    }

    // ============= Element-wise Operations Kernels =============

    __global__ void element_add_kernel(const float* a, const float* b, float* c, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }

    __global__ void element_sub_kernel(const float* a, const float* b, float* c, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] - b[idx];
        }
    }

    __global__ void element_mul_kernel(const float* a, const float* b, float* c, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] * b[idx];
        }
    }

    __global__ void element_div_kernel(const float* a, const float* b, float* c, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] / (b[idx] + 1e-8f); // Safe division
        }
    }

    // ============= In-place Element-wise Operations Kernels =============

    __global__ void element_add_inplace_kernel(float* a, const float* b, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            a[idx] += b[idx];
        }
    }

    __global__ void element_sub_inplace_kernel(float* a, const float* b, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            a[idx] -= b[idx];
        }
    }

    __global__ void element_mul_inplace_kernel(float* a, const float* b, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            a[idx] *= b[idx];
        }
    }

    __global__ void element_div_inplace_kernel(float* a, const float* b, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            a[idx] /= (b[idx] + 1e-8f); // Safe division
        }
    }

    // ============= Math Function Kernels =============

    __global__ void abs_kernel(float* data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = fabsf(data[idx]);
        }
    }

    __global__ void sqrt_kernel(float* data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = sqrtf(fmaxf(0.0f, data[idx]));
        }
    }

    __global__ void exp_kernel(float* data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = expf(data[idx]);
        }
    }

    __global__ void log_kernel(float* data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = logf(fmaxf(1e-10f, data[idx]));
        }
    }

    __global__ void sigmoid_kernel(float* data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float val = data[idx];
            data[idx] = 1.0f / (1.0f + expf(-val));
        }
    }

    __global__ void relu_kernel(float* data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = fmaxf(0.0f, data[idx]);
        }
    }

    __global__ void clamp_kernel(float* data, float min_val, float max_val, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = fminf(fmaxf(data[idx], min_val), max_val);
        }
    }

    // ============= Reduction Kernels - FIXED =============

    // Simple reduction kernel that works correctly with grid-stride loop
    __global__ void reduce_sum_kernel_simple(const float* data, float* partial_sums, size_t n) {
        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;
        unsigned int gridSize = blockDim.x * gridDim.x;

        // Grid-stride loop to handle arrays larger than grid
        float mySum = 0.0f;
        while (i < n) {
            mySum += data[i];
            i += gridSize;
        }

        // Store in shared memory
        sdata[tid] = mySum;
        __syncthreads();

        // Reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write result for this block to global mem
        if (tid == 0)
            partial_sums[blockIdx.x] = sdata[0];
    }

    __global__ void reduce_min_kernel_simple(const float* data, float* partial_mins, size_t n) {
        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;

        float myMin = (i < n) ? data[i] : FLT_MAX;
        sdata[tid] = myMin;
        __syncthreads();

        // Reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0)
            partial_mins[blockIdx.x] = sdata[0];
    }

    __global__ void reduce_max_kernel_simple(const float* data, float* partial_maxs, size_t n) {
        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;

        float myMax = (i < n) ? data[i] : -FLT_MAX;
        sdata[tid] = myMax;
        __syncthreads();

        // Reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0)
            partial_maxs[blockIdx.x] = sdata[0];
    }

    // ============= Launch Functions Implementation =============

    // Scalar operations
    void launch_scalar_add(float* data, float scalar, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        scalar_add_kernel<<<grid_size, block_size, 0, stream>>>(data, scalar, n);
    }

    void launch_scalar_sub(float* data, float scalar, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        scalar_sub_kernel<<<grid_size, block_size, 0, stream>>>(data, scalar, n);
    }

    void launch_scalar_mul(float* data, float scalar, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        scalar_mul_kernel<<<grid_size, block_size, 0, stream>>>(data, scalar, n);
    }

    void launch_scalar_div(float* data, float scalar, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        scalar_div_kernel<<<grid_size, block_size, 0, stream>>>(data, scalar, n);
    }

    // Element-wise operations
    void launch_element_add(const float* a, const float* b, float* c, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        element_add_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, n);
    }

    void launch_element_sub(const float* a, const float* b, float* c, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        element_sub_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, n);
    }

    void launch_element_mul(const float* a, const float* b, float* c, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        element_mul_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, n);
    }

    void launch_element_div(const float* a, const float* b, float* c, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        element_div_kernel<<<grid_size, block_size, 0, stream>>>(a, b, c, n);
    }

    // In-place element-wise operations
    void launch_element_add_inplace(float* a, const float* b, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        element_add_inplace_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n);
    }

    void launch_element_sub_inplace(float* a, const float* b, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        element_sub_inplace_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n);
    }

    void launch_element_mul_inplace(float* a, const float* b, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        element_mul_inplace_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n);
    }

    void launch_element_div_inplace(float* a, const float* b, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        element_div_inplace_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n);
    }

    // Math functions
    void launch_abs(float* data, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        abs_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_sqrt(float* data, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        sqrt_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_exp(float* data, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        exp_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_log(float* data, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        log_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_sigmoid(float* data, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        sigmoid_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_relu(float* data, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        relu_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_clamp(float* data, float min_val, float max_val, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        clamp_kernel<<<grid_size, block_size, 0, stream>>>(data, min_val, max_val, n);
    }

    // Reduction operations - COMPLETELY FIXED
    void launch_reduce_sum(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            *result = 0.0f;
            return;
        }

        const int threads = 256;
        const int blocks = std::min(1024, (int)((n + threads - 1) / threads));

        // Allocate temporary buffer for block results
        float* d_block_results;
        cudaMalloc(&d_block_results, blocks * sizeof(float));

        // First pass: reduce each block
        reduce_sum_kernel_simple<<<blocks, threads, threads * sizeof(float), stream>>>(
            data, d_block_results, n);

        if (stream == 0) {
            cudaDeviceSynchronize();
        }

        // If we have multiple blocks, do final reduction on CPU
        if (blocks > 1) {
            std::vector<float> block_results(blocks);
            cudaMemcpy(block_results.data(), d_block_results, blocks * sizeof(float), cudaMemcpyDeviceToHost);
            float sum = 0.0f;
            for (float val : block_results) {
                sum += val;
            }
            *result = sum;
        } else {
            // Single block, copy result directly
            cudaMemcpy(result, d_block_results, sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_block_results);
    }

    void launch_reduce_mean(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            *result = 0.0f;
            return;
        }

        float sum = 0.0f;
        launch_reduce_sum(data, &sum, n, stream);
        *result = sum / (float)n;
    }

    void launch_reduce_min(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            *result = 0.0f;
            return;
        }

        const int threads = 256;
        const int blocks = std::min(1024, (int)((n + threads - 1) / threads));

        float* d_block_results;
        cudaMalloc(&d_block_results, blocks * sizeof(float));

        reduce_min_kernel_simple<<<blocks, threads, threads * sizeof(float), stream>>>(
            data, d_block_results, n);

        if (stream == 0) {
            cudaDeviceSynchronize();
        }

        if (blocks > 1) {
            std::vector<float> block_results(blocks);
            cudaMemcpy(block_results.data(), d_block_results, blocks * sizeof(float), cudaMemcpyDeviceToHost);
            *result = *std::min_element(block_results.begin(), block_results.end());
        } else {
            cudaMemcpy(result, d_block_results, sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_block_results);
    }

    void launch_reduce_max(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            *result = 0.0f;
            return;
        }

        const int threads = 256;
        const int blocks = std::min(1024, (int)((n + threads - 1) / threads));

        float* d_block_results;
        cudaMalloc(&d_block_results, blocks * sizeof(float));

        reduce_max_kernel_simple<<<blocks, threads, threads * sizeof(float), stream>>>(
            data, d_block_results, n);

        if (stream == 0) {
            cudaDeviceSynchronize();
        }

        if (blocks > 1) {
            std::vector<float> block_results(blocks);
            cudaMemcpy(block_results.data(), d_block_results, blocks * sizeof(float), cudaMemcpyDeviceToHost);
            *result = *std::max_element(block_results.begin(), block_results.end());
        } else {
            cudaMemcpy(result, d_block_results, sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_block_results);
    }

} // namespace gs::tensor_ops