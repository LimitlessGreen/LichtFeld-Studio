/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace gs::training {

    __global__ void background_blend_kernel(
    float* output,
    const float* base_bg,
    const float* sine_bg,
    float mix_weight) {
        // Properly blend base background with sine background
        output[0] = base_bg[0] * (1.0f - mix_weight) + sine_bg[0] * mix_weight;
        output[1] = base_bg[1] * (1.0f - mix_weight) + sine_bg[1] * mix_weight;
        output[2] = base_bg[2] * (1.0f - mix_weight) + sine_bg[2] * mix_weight;
    }

__global__ void add_scale_regularization_kernel(
    float* grad_scales,
    const float* scales_raw,
    float reg_weight,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // exp(scale_raw) * reg_weight / n
        float scale = expf(scales_raw[idx]);
        grad_scales[idx] += scale * reg_weight / float(n);
    }
}

__global__ void add_opacity_regularization_kernel(
    float* grad_opacity,
    const float* opacity_raw,
    float reg_weight,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // sigmoid(opacity_raw) * (1 - sigmoid(opacity_raw)) * reg_weight / n
        float sig = 1.0f / (1.0f + expf(-opacity_raw[idx]));
        float sigmoid_grad = sig * (1.0f - sig);
        grad_opacity[idx] += sigmoid_grad * reg_weight / float(n);
    }
}

__global__ void zero_tensor_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 0.0f;
    }
}

__global__ void fill_tensor_kernel(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

// NEW: L1 loss kernels
__global__ void compute_l1_loss_forward_kernel(
    const float* pred,
    const float* gt,
    float* diff,
    float* abs_diff,
    float* l1_grad,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float d = pred[idx] - gt[idx];
        diff[idx] = d;
        abs_diff[idx] = fabsf(d);
        l1_grad[idx] = (d > 0.0f) ? (1.0f / float(n)) : (-1.0f / float(n));
    }
}

// Kernel to compute mean of array (sum then divide)
__global__ void compute_mean_kernel(
    const float* data,
    float* result,
    int n) {
    // Simple reduction - for production use CUB
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
        atomicAdd(result, sdata[0] / float(n));
    }
}

// Combine gradients with weights
__global__ void combine_gradients_kernel(
    float* output,
    const float* grad1,
    float weight1,
    const float* grad2,
    float weight2,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = grad1[idx] * weight1 + grad2[idx] * weight2;
    }
}

// Copy kernel (device to device)
__global__ void copy_kernel(
    float* dst,
    const float* src,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

    __global__ void compute_sine_background_kernel(
        float* output,
        int step,
        int periodR, int periodG, int periodB,
        float jitter_seed) {

    const float two_pi = 6.28318530718f;
    const float eps = 1e-4f;

    // Phase calculation
    float tR = (periodR > 0) ? float(step % periodR) / float(periodR) : 0.0f;
    float tG = (periodG > 0) ? float(step % periodG) / float(periodG) : 0.0f;
    float tB = (periodB > 0) ? float(step % periodB) / float(periodB) : 0.0f;

    float phaseR = two_pi * tR;
    float phaseG = two_pi * tG;
    float phaseB = two_pi * tB;

    // Compute sine values with phase shift
    output[0] = 0.5f * (1.0f + sinf(phaseR + 0.0f * two_pi / 3.0f));
    output[1] = 0.5f * (1.0f + sinf(phaseG + 1.0f * two_pi / 3.0f));
    output[2] = 0.5f * (1.0f + sinf(phaseB + 2.0f * two_pi / 3.0f));

    // Simple pseudo-random jitter (using step as seed)
    if (jitter_seed > 0.0f) {
        // Simple hash function for pseudo-random
        unsigned int hash = step * 1664525u + 1013904223u;
        float rand_r = (hash & 0xFFFF) / 65535.0f - 0.5f;
        hash = hash * 1664525u + 1013904223u;
        float rand_g = (hash & 0xFFFF) / 65535.0f - 0.5f;
        hash = hash * 1664525u + 1013904223u;
        float rand_b = (hash & 0xFFFF) / 65535.0f - 0.5f;

        output[0] = fmaxf(eps, fminf(1.0f - eps, output[0] + rand_r * 2.0f * jitter_seed));
        output[1] = fmaxf(eps, fminf(1.0f - eps, output[1] + rand_g * 2.0f * jitter_seed));
        output[2] = fmaxf(eps, fminf(1.0f - eps, output[2] + rand_b * 2.0f * jitter_seed));
    }
}



    void launch_compute_sine_background(
        float* output,
        int step,
        int periodR, int periodG, int periodB,
        float jitter_amp,
        cudaStream_t stream) {
    compute_sine_background_kernel<<<1, 1, 0, stream>>>(
        output, step, periodR, periodG, periodB, jitter_amp);
}

// Wrapper functions to call from C++
    void launch_background_blend(
    float* output,
    const float* base_bg,
    const float* sine_bg,
    float mix_weight,
    cudaStream_t stream) {
    background_blend_kernel<<<1, 1, 0, stream>>>(output, base_bg, sine_bg, mix_weight);
}

void launch_add_scale_regularization(
    float* grad_scales,
    const float* scales_raw,
    float reg_weight,
    int n,
    cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    add_scale_regularization_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_scales, scales_raw, reg_weight, n);
}

void launch_add_opacity_regularization(
    float* grad_opacity,
    const float* opacity_raw,
    float reg_weight,
    int n,
    cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    add_opacity_regularization_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_opacity, opacity_raw, reg_weight, n);
}

void launch_zero_tensor(float* data, int n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    zero_tensor_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
}

void launch_fill_tensor(float* data, float value, int n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    fill_tensor_kernel<<<grid_size, block_size, 0, stream>>>(data, value, n);
}

void launch_compute_l1_loss_forward(
    const float* pred,
    const float* gt,
    float* diff,
    float* abs_diff,
    float* l1_grad,
    int n,
    cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    compute_l1_loss_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        pred, gt, diff, abs_diff, l1_grad, n);
}

void launch_compute_mean(
    const float* data,
    float* result,
    int n,
    cudaStream_t stream) {
    // Zero the result first
    cudaMemsetAsync(result, 0, sizeof(float), stream);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    compute_mean_kernel<<<grid_size, block_size, block_size * sizeof(float), stream>>>(
        data, result, n);
}

void launch_combine_gradients(
    float* output,
    const float* grad1,
    float weight1,
    const float* grad2,
    float weight2,
    int n,
    cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    combine_gradients_kernel<<<grid_size, block_size, 0, stream>>>(
        output, grad1, weight1, grad2, weight2, n);
}

void launch_copy(
    float* dst,
    const float* src,
    int n,
    cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    copy_kernel<<<grid_size, block_size, 0, stream>>>(dst, src, n);
}

} // namespace gs::training