/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gs::tensor_ops {

// ============= Scalar Operations =============
__global__ void scalar_mul_kernel(float* data, float scalar, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scalar;
    }
}

__global__ void scalar_div_kernel(float* data, float scalar, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] /= scalar;
    }
}

// ============= Element-wise Operations =============
__global__ void element_add_kernel(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void element_sub_kernel(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void element_mul_kernel(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void element_div_kernel(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / (b[idx] + 1e-8f);  // Safe division
    }
}

// ============= In-place Element-wise Operations =============
__global__ void element_add_inplace_kernel(float* a, const float* b, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

__global__ void element_sub_inplace_kernel(float* a, const float* b, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] -= b[idx];
    }
}

__global__ void element_mul_inplace_kernel(float* a, const float* b, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= b[idx];
    }
}

__global__ void element_div_inplace_kernel(float* a, const float* b, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] /= (b[idx] + 1e-8f);
    }
}

// ============= Math Functions =============
__global__ void abs_kernel(float* data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fabsf(data[idx]);
    }
}

__global__ void sqrt_kernel(float* data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(fmaxf(0.0f, data[idx]));
    }
}

__global__ void exp_kernel(float* data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = expf(data[idx]);
    }
}

__global__ void log_kernel(float* data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = logf(fmaxf(1e-8f, data[idx]));
    }
}

__global__ void sigmoid_kernel(float* data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

__global__ void clamp_kernel(float* data, float min_val, float max_val, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fminf(fmaxf(data[idx], min_val), max_val);
    }
}

// ============= Reduction Operations =============
__global__ void reduce_sum_kernel(const float* data, float* result, size_t n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? data[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// ============= Launch Functions =============
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

void launch_clamp(float* data, float min_val, float max_val, size_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    clamp_kernel<<<grid_size, block_size, 0, stream>>>(data, min_val, max_val, n);
}

void launch_reduce_sum(const float* data, float* result, size_t n, cudaStream_t stream) {
    // Zero the result first
    cudaMemsetAsync(result, 0, sizeof(float), stream);
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    reduce_sum_kernel<<<grid_size, block_size, block_size * sizeof(float), stream>>>(data, result, n);
    
    // Copy result to host
    float host_result;
    cudaMemcpyAsync(&host_result, result, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Copy back to device (for consistency)
    cudaMemcpyAsync(result, &host_result, sizeof(float), cudaMemcpyHostToDevice, stream);
}

void launch_reduce_mean(const float* data, float* result, size_t n, cudaStream_t stream) {
    launch_reduce_sum(data, result, n, stream);
    
    float sum;
    cudaMemcpyAsync(&sum, result, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    float mean = sum / static_cast<float>(n);
    cudaMemcpyAsync(result, &mean, sizeof(float), cudaMemcpyHostToDevice, stream);
}

void launch_reduce_min(const float* data, float* result, size_t n, cudaStream_t stream) {
    // For now, use CUB
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Min(nullptr, temp_storage_bytes, data, result, n, stream);
    
    void* temp_storage = nullptr;
    cudaMalloc(&temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Min(temp_storage, temp_storage_bytes, data, result, n, stream);
    cudaFree(temp_storage);
}

void launch_reduce_max(const float* data, float* result, size_t n, cudaStream_t stream) {
    // For now, use CUB
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(nullptr, temp_storage_bytes, data, result, n, stream);
    
    void* temp_storage = nullptr;
    cudaMalloc(&temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, data, result, n, stream);
    cudaFree(temp_storage);
}

} // namespace gs::tensor_ops
