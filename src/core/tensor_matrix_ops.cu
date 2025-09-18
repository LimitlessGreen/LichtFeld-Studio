/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace gs::tensor_ops {

    // ============= Matrix Operations Kernels =============

    // Optimized transpose kernel using shared memory
    template <int TILE_DIM, int BLOCK_ROWS>
    __global__ void transpose_kernel(const float* input, float* output,
                                     size_t rows, size_t cols) {
        __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

        int x = blockIdx.x * TILE_DIM + threadIdx.x;
        int y = blockIdx.y * TILE_DIM + threadIdx.y;
        int width = cols;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            if (x < cols && (y + j) < rows) {
                tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
            }
        }

        __syncthreads();

        x = blockIdx.y * TILE_DIM + threadIdx.x;
        y = blockIdx.x * TILE_DIM + threadIdx.y;
        width = rows;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            if (x < rows && (y + j) < cols) {
                output[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }

    // Simple matrix multiply for batch operations (when cuBLAS batch isn't available)
    __global__ void batch_matmul_kernel(const float* a, const float* b, float* c,
                                        size_t batch_size, size_t m, size_t n, size_t k) {
        int batch_idx = blockIdx.z;
        if (batch_idx >= batch_size)
            return;

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < n) {
            float sum = 0.0f;

            const float* a_batch = a + batch_idx * m * k;
            const float* b_batch = b + batch_idx * k * n;
            float* c_batch = c + batch_idx * m * n;

            for (size_t i = 0; i < k; ++i) {
                sum += a_batch[row * k + i] * b_batch[i * n + col];
            }

            c_batch[row * n + col] = sum;
        }
    }

    // Eye matrix creation
    __global__ void eye_kernel(float* data, size_t m, size_t n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total = m * n;

        if (idx < total) {
            int row = idx / n;
            int col = idx % n;
            data[idx] = (row == col) ? 1.0f : 0.0f;
        }
    }

    // Create diagonal matrix from vector
    __global__ void diag_kernel(const float* diagonal, float* matrix, size_t n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total = n * n;

        if (idx < total) {
            int row = idx / n;
            int col = idx % n;
            matrix[idx] = (row == col) ? diagonal[row] : 0.0f;
        }
    }

    // Extract diagonal from matrix
    __global__ void extract_diag_kernel(const float* matrix, float* diagonal, size_t n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            diagonal[idx] = matrix[idx * n + idx];
        }
    }

    // ============= Launch Functions Implementation =============

    void launch_matmul(const float* a, const float* b, float* c,
                       size_t m, size_t n, size_t k,
                       cudaStream_t stream) {
        // For simple matrix multiply, use cuBLAS (handled in tensor_matrix_ops.cpp)
        // This is a fallback kernel if needed
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

        // Simple naive kernel (cuBLAS is preferred)
        // Note: This is just a placeholder - actual implementation uses cuBLAS
    }

    void launch_batch_matmul(const float* a, const float* b, float* c,
                             size_t batch_size, size_t m, size_t n, size_t k,
                             cudaStream_t stream) {
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x,
                  (m + block.y - 1) / block.y,
                  batch_size);

        batch_matmul_kernel<<<grid, block, 0, stream>>>(a, b, c, batch_size, m, n, k);
    }

    void launch_transpose(const float* input, float* output,
                          size_t rows, size_t cols,
                          cudaStream_t stream) {
        const int TILE_DIM = 32;
        const int BLOCK_ROWS = 8;

        dim3 block(TILE_DIM, BLOCK_ROWS);
        dim3 grid((cols + TILE_DIM - 1) / TILE_DIM,
                  (rows + TILE_DIM - 1) / TILE_DIM);

        transpose_kernel<TILE_DIM, BLOCK_ROWS><<<grid, block, 0, stream>>>(
            input, output, rows, cols);
    }

    void launch_dot_product(const float* a, const float* b, float* result,
                            size_t n, cudaStream_t stream) {
        // For dot product, use cuBLAS (handled in tensor_matrix_ops.cpp)
        // This is just a placeholder
    }

    void launch_eye(float* data, size_t m, size_t n, cudaStream_t stream) {
        size_t total = m * n;
        int block_size = 256;
        int grid_size = (total + block_size - 1) / block_size;
        eye_kernel<<<grid_size, block_size, 0, stream>>>(data, m, n);
    }

    void launch_diag(const float* diagonal, float* matrix, size_t n, cudaStream_t stream) {
        size_t total = n * n;
        int block_size = 256;
        int grid_size = (total + block_size - 1) / block_size;
        diag_kernel<<<grid_size, block_size, 0, stream>>>(diagonal, matrix, n);
    }

    void launch_extract_diag(const float* matrix, float* diagonal, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        extract_diag_kernel<<<grid_size, block_size, 0, stream>>>(matrix, diagonal, n);
    }

} // namespace gs::tensor_ops