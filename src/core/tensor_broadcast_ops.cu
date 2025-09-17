/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include "core/cuda_memory_guard.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gs::tensor_ops {

    // Helper function to compute linear index from multi-dimensional indices
    __device__ inline size_t compute_index(const size_t* indices, const size_t* strides, size_t rank) {
        size_t index = 0;
        for (size_t i = 0; i < rank; ++i) {
            index += indices[i] * strides[i];
        }
        return index;
    }

    // Convert linear index to multi-dimensional indices
    __device__ inline void linear_to_indices(size_t linear_idx, size_t* indices,
                                             const size_t* shape, size_t rank) {
        for (int i = rank - 1; i >= 0; --i) {
            indices[i] = linear_idx % shape[i];
            linear_idx /= shape[i];
        }
    }

    // Broadcast expansion kernel
    __global__ void broadcast_kernel(const float* src, float* dst,
                                     const size_t* src_shape, const size_t* dst_shape,
                                     size_t src_rank, size_t dst_rank,
                                     size_t dst_elements) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= dst_elements)
            return;

        // Convert linear index to multi-dimensional indices in dst
        size_t dst_indices[10]; // Max 10 dimensions
        linear_to_indices(idx, dst_indices, dst_shape, dst_rank);

        // Map dst indices to src indices (considering broadcasting)
        size_t src_indices[10];
        int rank_diff = dst_rank - src_rank;

        for (size_t i = 0; i < dst_rank; ++i) {
            int src_dim_idx = i - rank_diff;

            if (src_dim_idx < 0) {
                // This dimension doesn't exist in source (implicitly broadcast)
                continue;
            }

            if (src_shape[src_dim_idx] == 1) {
                // Broadcast dimension
                src_indices[src_dim_idx] = 0;
            } else {
                // Normal dimension
                src_indices[src_dim_idx] = dst_indices[i];
            }
        }

        // Compute source index
        size_t src_idx = 0;
        size_t stride = 1;
        for (int i = src_rank - 1; i >= 0; --i) {
            src_idx += src_indices[i] * stride;
            stride *= src_shape[i];
        }

        dst[idx] = src[src_idx];
    }

    // Binary operation with broadcasting
    template <typename Op>
    __global__ void broadcast_binary_kernel(const float* a, const float* b, float* c,
                                            const size_t* a_shape, const size_t* b_shape,
                                            const size_t* c_shape,
                                            size_t a_rank, size_t b_rank, size_t c_rank,
                                            size_t c_elements, Op op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= c_elements)
            return;

        // Convert linear index to multi-dimensional indices
        size_t c_indices[10];
        linear_to_indices(idx, c_indices, c_shape, c_rank);

        // Map to a indices
        size_t a_idx = 0;
        int a_rank_diff = c_rank - a_rank;
        size_t a_stride = 1;
        for (int i = a_rank - 1; i >= 0; --i) {
            int c_dim_idx = i + a_rank_diff;
            size_t dim_idx = (a_shape[i] == 1) ? 0 : c_indices[c_dim_idx];
            a_idx += dim_idx * a_stride;
            a_stride *= a_shape[i];
        }

        // Map to b indices
        size_t b_idx = 0;
        int b_rank_diff = c_rank - b_rank;
        size_t b_stride = 1;
        for (int i = b_rank - 1; i >= 0; --i) {
            int c_dim_idx = i + b_rank_diff;
            size_t dim_idx = (b_shape[i] == 1) ? 0 : c_indices[c_dim_idx];
            b_idx += dim_idx * b_stride;
            b_stride *= b_shape[i];
        }

        c[idx] = op(a[a_idx], b[b_idx]);
    }

    // Functors for different operations
    struct AddOp {
        __device__ float operator()(float a, float b) const { return a + b; }
    };

    struct SubOp {
        __device__ float operator()(float a, float b) const { return a - b; }
    };

    struct MulOp {
        __device__ float operator()(float a, float b) const { return a * b; }
    };

    struct DivOp {
        __device__ float operator()(float a, float b) const { return a / (b + 1e-8f); }
    };

    // Launch functions for broadcasting - UPDATED with RAII
    void launch_broadcast(const float* src, float* dst,
                          const size_t* src_shape, const size_t* dst_shape,
                          size_t src_rank, size_t dst_rank,
                          size_t dst_elements, cudaStream_t stream) {
        // Allocate device memory for shapes using RAII
        CudaDeviceMemory<size_t> d_src_shape(src_rank);
        CudaDeviceMemory<size_t> d_dst_shape(dst_rank);

        if (!d_src_shape.valid() || !d_dst_shape.valid()) {
            return;
        }

        d_src_shape.copy_from_host(src_shape, src_rank);
        d_dst_shape.copy_from_host(dst_shape, dst_rank);

        int block_size = 256;
        int grid_size = (dst_elements + block_size - 1) / block_size;

        broadcast_kernel<<<grid_size, block_size, 0, stream>>>(
            src, dst, d_src_shape.get(), d_dst_shape.get(), src_rank, dst_rank, dst_elements);
    }

    void launch_broadcast_add(const float* a, const float* b, float* c,
                              const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                              size_t a_rank, size_t b_rank, size_t c_rank,
                              size_t c_elements, cudaStream_t stream) {
        // Allocate device memory for shapes using RAII
        CudaDeviceMemory<size_t> d_a_shape(a_rank);
        CudaDeviceMemory<size_t> d_b_shape(b_rank);
        CudaDeviceMemory<size_t> d_c_shape(c_rank);

        if (!d_a_shape.valid() || !d_b_shape.valid() || !d_c_shape.valid()) {
            return;
        }

        d_a_shape.copy_from_host(a_shape, a_rank);
        d_b_shape.copy_from_host(b_shape, b_rank);
        d_c_shape.copy_from_host(c_shape, c_rank);

        int block_size = 256;
        int grid_size = (c_elements + block_size - 1) / block_size;

        broadcast_binary_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, c, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
            a_rank, b_rank, c_rank, c_elements, AddOp());
    }

    void launch_broadcast_sub(const float* a, const float* b, float* c,
                              const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                              size_t a_rank, size_t b_rank, size_t c_rank,
                              size_t c_elements, cudaStream_t stream) {
        CudaDeviceMemory<size_t> d_a_shape(a_rank);
        CudaDeviceMemory<size_t> d_b_shape(b_rank);
        CudaDeviceMemory<size_t> d_c_shape(c_rank);

        if (!d_a_shape.valid() || !d_b_shape.valid() || !d_c_shape.valid()) {
            return;
        }

        d_a_shape.copy_from_host(a_shape, a_rank);
        d_b_shape.copy_from_host(b_shape, b_rank);
        d_c_shape.copy_from_host(c_shape, c_rank);

        int block_size = 256;
        int grid_size = (c_elements + block_size - 1) / block_size;

        broadcast_binary_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, c, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
            a_rank, b_rank, c_rank, c_elements, SubOp());
    }

    void launch_broadcast_mul(const float* a, const float* b, float* c,
                              const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                              size_t a_rank, size_t b_rank, size_t c_rank,
                              size_t c_elements, cudaStream_t stream) {
        CudaDeviceMemory<size_t> d_a_shape(a_rank);
        CudaDeviceMemory<size_t> d_b_shape(b_rank);
        CudaDeviceMemory<size_t> d_c_shape(c_rank);

        if (!d_a_shape.valid() || !d_b_shape.valid() || !d_c_shape.valid()) {
            return;
        }

        d_a_shape.copy_from_host(a_shape, a_rank);
        d_b_shape.copy_from_host(b_shape, b_rank);
        d_c_shape.copy_from_host(c_shape, c_rank);

        int block_size = 256;
        int grid_size = (c_elements + block_size - 1) / block_size;

        broadcast_binary_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, c, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
            a_rank, b_rank, c_rank, c_elements, MulOp());
    }

    void launch_broadcast_div(const float* a, const float* b, float* c,
                              const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                              size_t a_rank, size_t b_rank, size_t c_rank,
                              size_t c_elements, cudaStream_t stream) {
        CudaDeviceMemory<size_t> d_a_shape(a_rank);
        CudaDeviceMemory<size_t> d_b_shape(b_rank);
        CudaDeviceMemory<size_t> d_c_shape(c_rank);

        if (!d_a_shape.valid() || !d_b_shape.valid() || !d_c_shape.valid()) {
            return;
        }

        d_a_shape.copy_from_host(a_shape, a_rank);
        d_b_shape.copy_from_host(b_shape, b_rank);
        d_c_shape.copy_from_host(c_shape, c_rank);

        int block_size = 256;
        int grid_size = (c_elements + block_size - 1) / block_size;

        broadcast_binary_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, c, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
            a_rank, b_rank, c_rank, c_elements, DivOp());
    }

} // namespace gs::tensor_ops