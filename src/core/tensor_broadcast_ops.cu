/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include "core/cuda_memory_guard.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace gs::tensor_ops {

// Broadcasting index calculation device function
__device__ inline size_t map_broadcast_index_device(size_t linear_idx,
                                                    const size_t* out_shape,
                                                    const size_t* in_shape,
                                                    size_t out_rank,
                                                    size_t in_rank) {
    size_t in_idx = 0;
    size_t out_stride = 1;

    for (int i = out_rank - 1; i >= 0; --i) {
        size_t out_coord = (linear_idx / out_stride) % out_shape[i];

        int in_dim = i - (out_rank - in_rank);
        if (in_dim >= 0) {
            size_t in_coord = (in_shape[in_dim] == 1) ? 0 : out_coord;
            size_t in_stride = 1;
            for (int j = in_dim + 1; j < in_rank; ++j) {
                in_stride *= in_shape[j];
            }
            in_idx += in_coord * in_stride;
        }

        out_stride *= out_shape[i];
    }

    return in_idx;
}

// Broadcast expansion kernel
__global__ void broadcast_kernel(const float* src, float* dst,
                                 const size_t* src_shape, const size_t* dst_shape,
                                 size_t src_rank, size_t dst_rank,
                                 size_t dst_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_elements) return;

    size_t src_idx = map_broadcast_index_device(idx, dst_shape, src_shape,
                                                dst_rank, src_rank);
    dst[idx] = src[src_idx];
}

__global__ void broadcast_bool_kernel(const unsigned char* src, unsigned char* dst,
                                      const size_t* src_shape, const size_t* dst_shape,
                                      size_t src_rank, size_t dst_rank,
                                      size_t dst_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_elements) return;

    size_t src_idx = map_broadcast_index_device(idx, dst_shape, src_shape,
                                                dst_rank, src_rank);
    dst[idx] = src[src_idx];
}

// Broadcasting binary operations
template<typename Op>
__global__ void broadcast_binary_kernel(const float* a, const float* b, float* c,
                                       const size_t* a_shape, const size_t* b_shape,
                                       const size_t* c_shape,
                                       size_t a_rank, size_t b_rank, size_t c_rank,
                                       size_t c_elements, Op op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= c_elements) return;

    size_t a_idx = map_broadcast_index_device(idx, c_shape, a_shape, c_rank, a_rank);
    size_t b_idx = map_broadcast_index_device(idx, c_shape, b_shape, c_rank, b_rank);

    c[idx] = op(a[a_idx], b[b_idx]);
}

// Public API functions
void launch_broadcast(const float* src, float* dst,
                     const size_t* src_shape, const size_t* dst_shape,
                     size_t src_rank, size_t dst_rank,
                     size_t dst_elements, cudaStream_t stream) {
    if (dst_elements == 0) return;

    CudaDeviceMemory<size_t> d_src_shape(src_rank);
    CudaDeviceMemory<size_t> d_dst_shape(dst_rank);

    cudaMemcpyAsync(d_src_shape.get(), src_shape, src_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_dst_shape.get(), dst_shape, dst_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);

    dim3 block(256);
    dim3 grid((dst_elements + block.x - 1) / block.x);

    broadcast_kernel<<<grid, block, 0, stream>>>(
        src, dst, d_src_shape.get(), d_dst_shape.get(),
        src_rank, dst_rank, dst_elements);
}

void launch_broadcast_bool(const unsigned char* src, unsigned char* dst,
                          const size_t* src_shape, const size_t* dst_shape,
                          size_t src_rank, size_t dst_rank,
                          size_t dst_elements, cudaStream_t stream) {
    if (dst_elements == 0) return;

    CudaDeviceMemory<size_t> d_src_shape(src_rank);
    CudaDeviceMemory<size_t> d_dst_shape(dst_rank);

    cudaMemcpyAsync(d_src_shape.get(), src_shape, src_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_dst_shape.get(), dst_shape, dst_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);

    dim3 block(256);
    dim3 grid((dst_elements + block.x - 1) / block.x);

    broadcast_bool_kernel<<<grid, block, 0, stream>>>(
        src, dst, d_src_shape.get(), d_dst_shape.get(),
        src_rank, dst_rank, dst_elements);
}

// Broadcasting operations
void launch_broadcast_add(const float* a, const float* b, float* c,
                         const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                         size_t a_rank, size_t b_rank, size_t c_rank,
                         size_t c_elements, cudaStream_t stream) {
    if (c_elements == 0) return;

    CudaDeviceMemory<size_t> d_a_shape(a_rank);
    CudaDeviceMemory<size_t> d_b_shape(b_rank);
    CudaDeviceMemory<size_t> d_c_shape(c_rank);

    cudaMemcpyAsync(d_a_shape.get(), a_shape, a_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_shape.get(), b_shape, b_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_c_shape.get(), c_shape, c_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);

    dim3 block(256);
    dim3 grid((c_elements + block.x - 1) / block.x);

    auto op = [] __device__ (float x, float y) { return x + y; };
    broadcast_binary_kernel<<<grid, block, 0, stream>>>(
        a, b, c, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
        a_rank, b_rank, c_rank, c_elements, op);
}

void launch_broadcast_sub(const float* a, const float* b, float* c,
                         const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                         size_t a_rank, size_t b_rank, size_t c_rank,
                         size_t c_elements, cudaStream_t stream) {
    if (c_elements == 0) return;

    CudaDeviceMemory<size_t> d_a_shape(a_rank);
    CudaDeviceMemory<size_t> d_b_shape(b_rank);
    CudaDeviceMemory<size_t> d_c_shape(c_rank);

    cudaMemcpyAsync(d_a_shape.get(), a_shape, a_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_shape.get(), b_shape, b_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_c_shape.get(), c_shape, c_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);

    dim3 block(256);
    dim3 grid((c_elements + block.x - 1) / block.x);

    auto op = [] __device__ (float x, float y) { return x - y; };
    broadcast_binary_kernel<<<grid, block, 0, stream>>>(
        a, b, c, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
        a_rank, b_rank, c_rank, c_elements, op);
}

void launch_broadcast_mul(const float* a, const float* b, float* c,
                         const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                         size_t a_rank, size_t b_rank, size_t c_rank,
                         size_t c_elements, cudaStream_t stream) {
    if (c_elements == 0) return;

    CudaDeviceMemory<size_t> d_a_shape(a_rank);
    CudaDeviceMemory<size_t> d_b_shape(b_rank);
    CudaDeviceMemory<size_t> d_c_shape(c_rank);

    cudaMemcpyAsync(d_a_shape.get(), a_shape, a_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_shape.get(), b_shape, b_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_c_shape.get(), c_shape, c_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);

    dim3 block(256);
    dim3 grid((c_elements + block.x - 1) / block.x);

    auto op = [] __device__ (float x, float y) { return x * y; };
    broadcast_binary_kernel<<<grid, block, 0, stream>>>(
        a, b, c, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
        a_rank, b_rank, c_rank, c_elements, op);
}

void launch_broadcast_div(const float* a, const float* b, float* c,
                         const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                         size_t a_rank, size_t b_rank, size_t c_rank,
                         size_t c_elements, cudaStream_t stream) {
    if (c_elements == 0) return;

    CudaDeviceMemory<size_t> d_a_shape(a_rank);
    CudaDeviceMemory<size_t> d_b_shape(b_rank);
    CudaDeviceMemory<size_t> d_c_shape(c_rank);

    cudaMemcpyAsync(d_a_shape.get(), a_shape, a_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_shape.get(), b_shape, b_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_c_shape.get(), c_shape, c_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);

    dim3 block(256);
    dim3 grid((c_elements + block.x - 1) / block.x);

    auto op = [] __device__ (float x, float y) { return x / (y + 1e-8f); };
    broadcast_binary_kernel<<<grid, block, 0, stream>>>(
        a, b, c, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
        a_rank, b_rank, c_rank, c_elements, op);
}

void launch_pow_tensor(const float* a, const float* b, float* c,
                      const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                      size_t a_rank, size_t b_rank, size_t c_rank,
                      size_t c_elements, cudaStream_t stream) {
    if (c_elements == 0) return;

    CudaDeviceMemory<size_t> d_a_shape(a_rank);
    CudaDeviceMemory<size_t> d_b_shape(b_rank);
    CudaDeviceMemory<size_t> d_c_shape(c_rank);

    cudaMemcpyAsync(d_a_shape.get(), a_shape, a_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_shape.get(), b_shape, b_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_c_shape.get(), c_shape, c_rank * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);

    dim3 block(256);
    dim3 grid((c_elements + block.x - 1) / block.x);

    auto op = [] __device__ (float x, float y) { return powf(x, y); };
    broadcast_binary_kernel<<<grid, block, 0, stream>>>(
        a, b, c, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
        a_rank, b_rank, c_rank, c_elements, op);
}

} // namespace gs::tensor_ops