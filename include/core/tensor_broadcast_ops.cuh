/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/memory_pool.hpp"
#include "core/logger.hpp"
#include <cuda_runtime.h>
#include <type_traits>

namespace gs::tensor_ops {

    // ============================================================================
    // BROADCAST PATTERN DETECTION (tiny-cuda-nn style optimization)
    // ============================================================================

    enum class BroadcastPattern {
        Scalar,        // One operand is scalar (size 1)
        Row,           // e.g., (M×N) op (1×N) - broadcast along rows
        Column,        // e.g., (M×N) op (M×1) - broadcast along columns
        Channel3D,     // e.g., (H×W×C) op (1×1×C) - channel-wise broadcast (neural rendering)
        Generic        // Complex pattern - use generic kernel
    };

    // Detect broadcast pattern for binary operations
    inline BroadcastPattern detect_broadcast_pattern(
        const size_t* a_shape, const size_t* b_shape,
        size_t a_rank, size_t b_rank, size_t c_rank) {

        // Scalar pattern: One operand has size 1
        size_t a_size = 1, b_size = 1;
        for (size_t i = 0; i < a_rank; ++i) a_size *= a_shape[i];
        for (size_t i = 0; i < b_rank; ++i) b_size *= b_shape[i];

        if (a_size == 1 || b_size == 1) {
            return BroadcastPattern::Scalar;
        }

        // 3D channel-wise pattern: (H×W×C) op (1×1×C)
        if (c_rank == 3 && a_rank == 3 && b_rank == 3) {
            // Check if one operand is (H×W×C) and the other is (1×1×C)
            if (a_shape[0] > 1 && a_shape[1] > 1 &&
                b_shape[0] == 1 && b_shape[1] == 1 &&
                a_shape[2] == b_shape[2]) {
                return BroadcastPattern::Channel3D;
            }
            if (b_shape[0] > 1 && b_shape[1] > 1 &&
                a_shape[0] == 1 && a_shape[1] == 1 &&
                a_shape[2] == b_shape[2]) {
                return BroadcastPattern::Channel3D;
            }
        }

        // 2D patterns only
        if (c_rank != 2) return BroadcastPattern::Generic;

        // Row broadcast: (M×N) op (1×N)
        if (a_rank == 2 && b_rank == 2) {
            if (a_shape[0] > 1 && b_shape[0] == 1 && a_shape[1] == b_shape[1]) {
                return BroadcastPattern::Row;
            }
            if (b_shape[0] > 1 && a_shape[0] == 1 && a_shape[1] == b_shape[1]) {
                return BroadcastPattern::Row;
            }
        }

        // Column broadcast: (M×N) op (M×1)
        if (a_rank == 2 && b_rank == 2) {
            if (a_shape[1] > 1 && b_shape[1] == 1 && a_shape[0] == b_shape[0]) {
                return BroadcastPattern::Column;
            }
            if (b_shape[1] > 1 && a_shape[1] == 1 && a_shape[0] == b_shape[0]) {
                return BroadcastPattern::Column;
            }
        }

        return BroadcastPattern::Generic;
    }

    // ============================================================================
    // SPECIALIZED BROADCAST KERNELS (tiny-cuda-nn style vectorized)
    // ============================================================================

#ifdef __CUDACC__  // Only compile CUDA kernels with nvcc

    // Scalar broadcast kernel - Simple version without float4 (safer)
    template<typename BinaryOp>
    __global__ void broadcast_scalar_kernel_float(
        const float* a, const float* b, float* c,
        size_t a_size, size_t b_size, size_t c_elements,
        BinaryOp op)
    {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= c_elements) return;

        const float scalar_val = (a_size == 1) ? a[0] : b[0];
        const float* array = (a_size == 1) ? b : a;
        const bool a_is_scalar = (a_size == 1);

        c[idx] = a_is_scalar ? op(scalar_val, array[idx]) : op(array[idx], scalar_val);
    }

    // Row broadcast kernel - Simple version without float4
    template<typename BinaryOp>
    __global__ void broadcast_row_kernel_float(
        const float* a, const float* b, float* c,
        size_t M, size_t N, bool a_is_row,
        BinaryOp op)
    {
        const size_t row = blockIdx.y;
        const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

        if (col >= N) return;

        const size_t idx = row * N + col;
        const float* row_data = a_is_row ? b : a;
        const float* broadcast_row = a_is_row ? a : b;

        c[idx] = a_is_row ?
            op(broadcast_row[col], row_data[idx]) :
            op(row_data[idx], broadcast_row[col]);
    }

    // Column broadcast kernel - Simple version without float4
    template<typename BinaryOp>
    __global__ void broadcast_column_kernel_float(
        const float* a, const float* b, float* c,
        size_t M, size_t N, bool a_is_col,
        BinaryOp op)
    {
        const size_t row = blockIdx.y;
        const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

        if (col >= N) return;

        const size_t idx = row * N + col;
        const float* row_data = a_is_col ? b : a;
        const float col_val = a_is_col ? a[row] : b[row];

        c[idx] = a_is_col ?
            op(col_val, row_data[idx]) :
            op(row_data[idx], col_val);
    }

    // Channel3D broadcast kernel - optimized for neural rendering
    // Pattern: (H×W×C) op (1×1×C) - extremely common in neural rendering!
    // Note: Cannot use float4 because pixel_offset may not be 16-byte aligned
    template<typename BinaryOp>
    __global__ void broadcast_channel3d_kernel_float(
        const float* a, const float* b, float* c,
        size_t H, size_t W, size_t C, bool a_is_broadcast,
        BinaryOp op)
    {
        // 2D grid: process pixels in parallel
        const size_t pixel_idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total_pixels = H * W;

        if (pixel_idx >= total_pixels) return;

        const size_t pixel_offset = pixel_idx * C;
        const float* pixel_data = a_is_broadcast ? b + pixel_offset : a + pixel_offset;
        const float* channel_vals = a_is_broadcast ? a : b;

        // Fully unroll for RGB (most common case)
        if (C == 3) {
            float r = pixel_data[0];
            float g = pixel_data[1];
            float b_val = pixel_data[2];

            if (a_is_broadcast) {
                r = op(channel_vals[0], r);
                g = op(channel_vals[1], g);
                b_val = op(channel_vals[2], b_val);
            } else {
                r = op(r, channel_vals[0]);
                g = op(g, channel_vals[1]);
                b_val = op(b_val, channel_vals[2]);
            }

            c[pixel_offset + 0] = r;
            c[pixel_offset + 1] = g;
            c[pixel_offset + 2] = b_val;
            return;
        }

        // For larger C: process channel-by-channel
        for (size_t ch = 0; ch < C; ++ch) {
            c[pixel_offset + ch] = a_is_broadcast ?
                op(channel_vals[ch], pixel_data[ch]) :
                op(pixel_data[ch], channel_vals[ch]);
        }
    }

    // ============================================================================
    // BINARY BROADCAST KERNEL (Generic fallback)
    // ============================================================================

    template<typename T, typename OutputT, typename BinaryOp>
    __global__ void broadcast_binary_kernel(
        const T* a, const T* b, OutputT* c,
        const int* a_shape, const int* b_shape, const int* c_shape,
        int a_rank, int b_rank, int c_rank,
        size_t c_elements, BinaryOp op)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= c_elements) return;

        // Compute strides inline
        int c_strides[8], a_strides[8], b_strides[8];

        c_strides[c_rank - 1] = 1;
        for (int i = c_rank - 2; i >= 0; --i) {
            c_strides[i] = c_strides[i + 1] * c_shape[i + 1];
        }

        if (a_rank > 0) {
            a_strides[a_rank - 1] = 1;
            for (int i = a_rank - 2; i >= 0; --i) {
                a_strides[i] = a_strides[i + 1] * a_shape[i + 1];
            }
        }

        if (b_rank > 0) {
            b_strides[b_rank - 1] = 1;
            for (int i = b_rank - 2; i >= 0; --i) {
                b_strides[i] = b_strides[i + 1] * b_shape[i + 1];
            }
        }

        // Broadcast indexing
        int a_idx = 0, b_idx = 0;
        size_t remaining = idx;

        for (int i = 0; i < c_rank; ++i) {
            int c_coord = remaining / c_strides[i];
            remaining %= c_strides[i];

            // Map to a's coordinate
            int offset_a = c_rank - a_rank;
            if (i >= offset_a) {
                int dim = i - offset_a;
                int coord = (a_shape[dim] == 1) ? 0 : c_coord;
                a_idx += coord * a_strides[dim];
            }

            // Map to b's coordinate
            int offset_b = c_rank - b_rank;
            if (i >= offset_b) {
                int dim = i - offset_b;
                int coord = (b_shape[dim] == 1) ? 0 : c_coord;
                b_idx += coord * b_strides[dim];
            }
        }

        c[idx] = op(a[a_idx], b[b_idx]);
    }

#endif  // __CUDACC__

    // ============================================================================
    // BINARY BROADCAST HOST LAUNCHER TEMPLATE (INLINE FOR CORRECT INSTANTIATION)
    // ============================================================================

    template<typename T, typename OutputT, typename BinaryOp>
    void launch_broadcast_binary(const T* a, const T* b, OutputT* c,
                                 const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                                 size_t a_rank, size_t b_rank, size_t c_rank,
                                 size_t c_elements, BinaryOp op, cudaStream_t stream) {
        if (c_elements == 0) return;

        // DEBUG: Disabled for production - enable only for debugging
        // fprintf(stderr, "BROADCAST CALLED: c_elem=%zu, a_rank=%zu, b_rank=%zu\n", c_elements, a_rank, b_rank);
        // fflush(stderr);

        // Only use specialized kernels for float→float operations
        constexpr bool use_vectorized = std::is_same<T, float>::value && std::is_same<OutputT, float>::value;

        if constexpr (use_vectorized) {
            // Detect broadcast pattern for optimization
            BroadcastPattern pattern = detect_broadcast_pattern(a_shape, b_shape, a_rank, b_rank, c_rank);
            const int block_size = 256;

            // DEBUG: Disabled for production
            // const char* pattern_names[] = {"Scalar", "Row", "Column", "Channel3D", "Generic"};
            // printf("BROADCAST: Pattern=%s, c_elem=%zu, a_rank=%zu, b_rank=%zu, c_rank=%zu\n",
            //        pattern_names[static_cast<int>(pattern)], c_elements, a_rank, b_rank, c_rank);

            switch (pattern) {
                case BroadcastPattern::Scalar: {
                    // Scalar broadcast: (M×N) op scalar
                    size_t a_size = 1, b_size = 1;
                    for (size_t i = 0; i < a_rank; ++i) a_size *= a_shape[i];
                    for (size_t i = 0; i < b_rank; ++i) b_size *= b_shape[i];

                    const int grid_size = (c_elements + block_size - 1) / block_size;

#ifdef __CUDACC__
                    broadcast_scalar_kernel_float<<<grid_size, block_size, 0, stream>>>(
                        a, b, c, a_size, b_size, c_elements, op);
#else
                    static_assert(sizeof(T) == 0, "CUDA compiler required for broadcast operations");
#endif
                    return;
                }

                case BroadcastPattern::Row: {
                    // Row broadcast: (M×N) op (1×N)
                    const size_t M = c_shape[0];
                    const size_t N = c_shape[1];
                    const bool a_is_row = (a_shape[0] == 1);

                    dim3 grid((N + block_size - 1) / block_size, M);

#ifdef __CUDACC__
                    broadcast_row_kernel_float<<<grid, block_size, 0, stream>>>(
                        a, b, c, M, N, a_is_row, op);
#else
                    static_assert(sizeof(T) == 0, "CUDA compiler required for broadcast operations");
#endif
                    return;
                }

                case BroadcastPattern::Column: {
                    // Column broadcast: (M×N) op (M×1)
                    const size_t M = c_shape[0];
                    const size_t N = c_shape[1];
                    const bool a_is_col = (a_shape[1] == 1);

                    dim3 grid((N + block_size - 1) / block_size, M);

#ifdef __CUDACC__
                    broadcast_column_kernel_float<<<grid, block_size, 0, stream>>>(
                        a, b, c, M, N, a_is_col, op);
#else
                    static_assert(sizeof(T) == 0, "CUDA compiler required for broadcast operations");
#endif
                    return;
                }

                case BroadcastPattern::Channel3D: {
                    // Channel3D broadcast: (H×W×C) op (1×1×C) - VECTORIZED FAST PATH!
                    // Critical for neural rendering: color transforms, normalization, etc.
                    const size_t H = c_shape[0];
                    const size_t W = c_shape[1];
                    const size_t C = c_shape[2];
                    const bool a_is_broadcast = (a_shape[0] == 1 && a_shape[1] == 1);

                    // Grid: process all pixels in parallel
                    const size_t total_pixels = H * W;
                    const int grid_size = (total_pixels + block_size - 1) / block_size;
                    const int max_grid_dim = 65535; // CUDA limit

                    // Use 2D grid if 1D grid would exceed limit
                    dim3 grid;
                    if (grid_size <= max_grid_dim) {
                        grid = dim3(grid_size, 1);
                    } else {
                        // Split into 2D grid
                        const int grid_x = (grid_size + max_grid_dim - 1) / max_grid_dim;
                        const int grid_y = (grid_size + grid_x - 1) / grid_x;
                        grid = dim3(grid_x, grid_y);
                    }

#ifdef __CUDACC__
                    broadcast_channel3d_kernel_float<<<grid, block_size, 0, stream>>>(
                        a, b, c, H, W, C, a_is_broadcast, op);
#else
                    static_assert(sizeof(T) == 0, "CUDA compiler required for broadcast operations");
#endif
                    return;
                }

                case BroadcastPattern::Generic:
                    // Fall through to generic kernel
                    break;
            }
        }

        // Generic kernel for all types and complex patterns
        const int block_size = 256;

        // Generic N-D broadcast - use existing implementation
        int* d_a_shape = static_cast<int*>(
            CudaMemoryPool::instance().allocate(a_rank * sizeof(int), stream));
        int* d_b_shape = static_cast<int*>(
            CudaMemoryPool::instance().allocate(b_rank * sizeof(int), stream));
        int* d_c_shape = static_cast<int*>(
            CudaMemoryPool::instance().allocate(c_rank * sizeof(int), stream));

        if (!d_a_shape || !d_b_shape || !d_c_shape) {
            LOG_ERROR("Failed to allocate shape arrays from memory pool");
            if (d_a_shape) CudaMemoryPool::instance().deallocate(d_a_shape, stream);
            if (d_b_shape) CudaMemoryPool::instance().deallocate(d_b_shape, stream);
            if (d_c_shape) CudaMemoryPool::instance().deallocate(d_c_shape, stream);
            return;
        }

        std::vector<int> a_vec(a_shape, a_shape + a_rank);
        std::vector<int> b_vec(b_shape, b_shape + b_rank);
        std::vector<int> c_vec(c_shape, c_shape + c_rank);

        cudaMemcpy(d_a_shape, a_vec.data(), a_rank * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_shape, b_vec.data(), b_rank * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c_shape, c_vec.data(), c_rank * sizeof(int), cudaMemcpyHostToDevice);

        const int grid_size = (c_elements + block_size - 1) / block_size;

#ifdef __CUDACC__
        broadcast_binary_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, c, d_a_shape, d_b_shape, d_c_shape,
            a_rank, b_rank, c_rank, c_elements, op);
#else
        static_assert(sizeof(T) == 0, "CUDA compiler required for broadcast operations");
#endif

        CudaMemoryPool::instance().deallocate(d_a_shape, stream);
        CudaMemoryPool::instance().deallocate(d_b_shape, stream);
        CudaMemoryPool::instance().deallocate(d_c_shape, stream);
    }

} // namespace gs::tensor_ops
