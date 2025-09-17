/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include "core/cuda_memory_guard.hpp"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gs::tensor_ops {

    // ============= Comparison Kernels =============
    template <typename Op>
    __global__ void compare_broadcast_kernel(const float* a, const float* b, unsigned char* c,
                                             const size_t* a_shape, const size_t* b_shape,
                                             const size_t* c_shape,
                                             size_t a_rank, size_t b_rank, size_t c_rank,
                                             size_t c_elements, Op op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= c_elements)
            return;

        // Calculate indices for result tensor
        size_t indices[10];
        size_t temp = idx;
        for (int i = c_rank - 1; i >= 0; --i) {
            indices[i] = temp % c_shape[i];
            temp /= c_shape[i];
        }

        // Map to a index - FIXED: properly handle rank differences and broadcasting
        size_t a_idx = 0;
        if (a_rank > 0) {
            int rank_diff = c_rank - a_rank;
            size_t a_stride = 1;
            for (int i = a_rank - 1; i >= 0; --i) {
                int c_dim_idx = i + rank_diff;
                // Only use the index if this dimension exists and is not broadcast (size 1)
                size_t dim_idx = (a_shape[i] == 1) ? 0 : indices[c_dim_idx];
                a_idx += dim_idx * a_stride;
                a_stride *= a_shape[i];
            }
        }

        // Map to b index - FIXED: properly handle rank differences and broadcasting
        size_t b_idx = 0;
        if (b_rank > 0) {
            int rank_diff = c_rank - b_rank;
            size_t b_stride = 1;
            for (int i = b_rank - 1; i >= 0; --i) {
                int c_dim_idx = i + rank_diff;
                // Only use the index if this dimension exists and is not broadcast (size 1)
                size_t dim_idx = (b_shape[i] == 1) ? 0 : indices[c_dim_idx];
                b_idx += dim_idx * b_stride;
                b_stride *= b_shape[i];
            }
        }

        c[idx] = op(a[a_idx], b[b_idx]) ? 1 : 0;
    }

    template <typename Op>
    __global__ void compare_scalar_kernel(const float* a, float value, unsigned char* result,
                                          size_t n, Op op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            result[idx] = op(a[idx], value) ? 1 : 0;
        }
    }

    struct EqOp {
        __device__ bool operator()(float a, float b) const { return a == b; }
    };

    struct LtOp {
        __device__ bool operator()(float a, float b) const { return a < b; }
    };

    struct GtOp {
        __device__ bool operator()(float a, float b) const { return a > b; }
    };

    // Launch functions for comparisons - UPDATED with RAII
    void launch_compare_eq(const float* a, const float* b, unsigned char* result,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream) {
        CudaDeviceMemory<size_t> d_a_shape(a_rank);
        CudaDeviceMemory<size_t> d_b_shape(b_rank);
        CudaDeviceMemory<size_t> d_c_shape(c_rank);

        if (!d_a_shape.valid() || !d_b_shape.valid() || !d_c_shape.valid()) {
            return; // Memory allocation failed
        }

        d_a_shape.copy_from_host(a_shape, a_rank);
        d_b_shape.copy_from_host(b_shape, b_rank);
        d_c_shape.copy_from_host(c_shape, c_rank);

        int block_size = 256;
        int grid_size = (c_elements + block_size - 1) / block_size;

        compare_broadcast_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, result, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
            a_rank, b_rank, c_rank, c_elements, EqOp());
    }

    void launch_compare_lt(const float* a, const float* b, unsigned char* result,
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

        compare_broadcast_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, result, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
            a_rank, b_rank, c_rank, c_elements, LtOp());
    }

    void launch_compare_gt(const float* a, const float* b, unsigned char* result,
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

        compare_broadcast_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, result, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
            a_rank, b_rank, c_rank, c_elements, GtOp());
    }

    void launch_compare_scalar_eq(const float* a, float value, unsigned char* result,
                                  size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        compare_scalar_kernel<<<grid_size, block_size, 0, stream>>>(a, value, result, n, EqOp());
    }

    void launch_compare_scalar_lt(const float* a, float value, unsigned char* result,
                                  size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        compare_scalar_kernel<<<grid_size, block_size, 0, stream>>>(a, value, result, n, LtOp());
    }

    void launch_compare_scalar_gt(const float* a, float value, unsigned char* result,
                                  size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        compare_scalar_kernel<<<grid_size, block_size, 0, stream>>>(a, value, result, n, GtOp());
    }

    // ============= Logical Operations Kernels =============
    template <typename Op>
    __global__ void logical_binary_kernel(const unsigned char* a, const unsigned char* b,
                                          unsigned char* c,
                                          const size_t* a_shape, const size_t* b_shape,
                                          const size_t* c_shape,
                                          size_t a_rank, size_t b_rank, size_t c_rank,
                                          size_t c_elements, Op op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= c_elements)
            return;

        // Calculate indices
        size_t indices[10];
        size_t temp = idx;
        for (int i = c_rank - 1; i >= 0; --i) {
            indices[i] = temp % c_shape[i];
            temp /= c_shape[i];
        }

        // Map to a index
        size_t a_idx = 0;
        int a_rank_diff = c_rank - a_rank;
        size_t a_stride = 1;
        for (int i = a_rank - 1; i >= 0; --i) {
            int c_dim_idx = i + a_rank_diff;
            size_t dim_idx = (a_shape[i] == 1) ? 0 : indices[c_dim_idx];
            a_idx += dim_idx * a_stride;
            a_stride *= a_shape[i];
        }

        // Map to b index
        size_t b_idx = 0;
        int b_rank_diff = c_rank - b_rank;
        size_t b_stride = 1;
        for (int i = b_rank - 1; i >= 0; --i) {
            int c_dim_idx = i + b_rank_diff;
            size_t dim_idx = (b_shape[i] == 1) ? 0 : indices[c_dim_idx];
            b_idx += dim_idx * b_stride;
            b_stride *= b_shape[i];
        }

        c[idx] = op(a[a_idx], b[b_idx]);
    }

    struct AndOp {
        __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
            return (a && b) ? 1 : 0;
        }
    };

    struct OrOp {
        __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
            return (a || b) ? 1 : 0;
        }
    };

    struct XorOp {
        __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
            return ((a != 0) != (b != 0)) ? 1 : 0;
        }
    };

    __global__ void logical_not_kernel(const unsigned char* a, unsigned char* result, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            result[idx] = a[idx] ? 0 : 1;
        }
    }

    void launch_logical_and(const unsigned char* a, const unsigned char* b, unsigned char* result,
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

        logical_binary_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, result, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
            a_rank, b_rank, c_rank, c_elements, AndOp());
    }

    void launch_logical_or(const unsigned char* a, const unsigned char* b, unsigned char* result,
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

        logical_binary_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, result, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
            a_rank, b_rank, c_rank, c_elements, OrOp());
    }

    void launch_logical_xor(const unsigned char* a, const unsigned char* b, unsigned char* result,
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

        logical_binary_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, result, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
            a_rank, b_rank, c_rank, c_elements, XorOp());
    }

    void launch_logical_not(const unsigned char* a, unsigned char* result,
                            size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        logical_not_kernel<<<grid_size, block_size, 0, stream>>>(a, result, n);
    }

    // ============= Masking Operations Kernels =============
    __global__ void masked_select_kernel(const float* input, const unsigned char* mask,
                                         float* output, const int* scan_result, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n && mask[idx]) {
            output[scan_result[idx]] = input[idx];
        }
    }

    void launch_masked_select(const float* input, const unsigned char* mask,
                              float* output, size_t n, size_t output_size, cudaStream_t stream) {
        // Early exit for empty case
        if (n == 0 || output_size == 0) {
            return;
        }

        // Use RAII for scan result
        CudaDeviceMemory<int> d_scan_result(n);
        if (!d_scan_result.valid()) {
            return;
        }

        // Use CUB for efficient parallel prefix sum
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                      mask, d_scan_result.get(), n, stream);

        CudaDeviceMemory<uint8_t> d_temp_storage(temp_storage_bytes);
        if (!d_temp_storage.valid()) {
            return;
        }

        cub::DeviceScan::ExclusiveSum(d_temp_storage.get(), temp_storage_bytes,
                                      mask, d_scan_result.get(), n, stream);

        // Synchronize to ensure scan is complete before using results
        if (stream != 0) {
            cudaStreamSynchronize(stream);
        }

        // Now gather the selected elements
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        masked_select_kernel<<<grid_size, block_size, 0, stream>>>(
            input, mask, output, d_scan_result.get(), n);
    }

    __global__ void masked_fill_kernel(float* data, const unsigned char* mask,
                                       float value, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n && mask[idx]) {
            data[idx] = value;
        }
    }

    void launch_masked_fill(float* data, const unsigned char* mask,
                            float value, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        masked_fill_kernel<<<grid_size, block_size, 0, stream>>>(data, mask, value, n);
    }

    __global__ void masked_scatter_kernel(float* data, const unsigned char* mask,
                                          const float* src, const int* scan_result, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n && mask[idx]) {
            data[idx] = src[scan_result[idx]];
        }
    }

    void launch_masked_scatter(float* data, const unsigned char* mask,
                               const float* src, size_t n, size_t src_size, cudaStream_t stream) {
        // Early exit for empty case
        if (n == 0 || src_size == 0) {
            return;
        }

        // Use RAII for scan result
        CudaDeviceMemory<int> d_scan_result(n);
        if (!d_scan_result.valid()) {
            return;
        }

        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                      mask, d_scan_result.get(), n, stream);

        CudaDeviceMemory<uint8_t> d_temp_storage(temp_storage_bytes);
        if (!d_temp_storage.valid()) {
            return;
        }

        cub::DeviceScan::ExclusiveSum(d_temp_storage.get(), temp_storage_bytes,
                                      mask, d_scan_result.get(), n, stream);

        // Synchronize to ensure scan is complete
        if (stream != 0) {
            cudaStreamSynchronize(stream);
        }

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        masked_scatter_kernel<<<grid_size, block_size, 0, stream>>>(
            data, mask, src, d_scan_result.get(), n);
    }

    // Fixed where kernel with proper broadcasting support
    __global__ void where_broadcast_kernel(const unsigned char* condition,
                                           const float* x, const float* y, float* result,
                                           const size_t* cond_shape, const size_t* x_shape,
                                           const size_t* y_shape, const size_t* result_shape,
                                           size_t cond_rank, size_t x_rank,
                                           size_t y_rank, size_t result_rank,
                                           size_t result_elements) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= result_elements)
            return;

        // Calculate multi-dimensional indices for result
        size_t indices[10];
        size_t temp = idx;
        for (int i = result_rank - 1; i >= 0; --i) {
            indices[i] = temp % result_shape[i];
            temp /= result_shape[i];
        }

        // Map to condition index
        size_t cond_idx = 0;
        int cond_rank_diff = result_rank - cond_rank;
        size_t cond_stride = 1;
        for (int i = cond_rank - 1; i >= 0; --i) {
            int result_dim_idx = i + cond_rank_diff;
            size_t dim_idx = (cond_shape[i] == 1) ? 0 : indices[result_dim_idx];
            cond_idx += dim_idx * cond_stride;
            cond_stride *= cond_shape[i];
        }

        // Map to x index
        size_t x_idx = 0;
        int x_rank_diff = result_rank - x_rank;
        size_t x_stride = 1;
        for (int i = x_rank - 1; i >= 0; --i) {
            int result_dim_idx = i + x_rank_diff;
            size_t dim_idx = (x_shape[i] == 1) ? 0 : indices[result_dim_idx];
            x_idx += dim_idx * x_stride;
            x_stride *= x_shape[i];
        }

        // Map to y index
        size_t y_idx = 0;
        int y_rank_diff = result_rank - y_rank;
        size_t y_stride = 1;
        for (int i = y_rank - 1; i >= 0; --i) {
            int result_dim_idx = i + y_rank_diff;
            size_t dim_idx = (y_shape[i] == 1) ? 0 : indices[result_dim_idx];
            y_idx += dim_idx * y_stride;
            y_stride *= y_shape[i];
        }

        result[idx] = condition[cond_idx] ? x[x_idx] : y[y_idx];
    }

    void launch_where(const unsigned char* condition,
                      const float* x, const float* y, float* result,
                      const size_t* cond_shape, const size_t* x_shape,
                      const size_t* y_shape, const size_t* result_shape,
                      size_t cond_rank, size_t x_rank, size_t y_rank, size_t result_rank,
                      size_t result_elements, cudaStream_t stream) {

        // Allocate device memory for shapes using RAII
        CudaDeviceMemory<size_t> d_cond_shape(cond_rank);
        CudaDeviceMemory<size_t> d_x_shape(x_rank);
        CudaDeviceMemory<size_t> d_y_shape(y_rank);
        CudaDeviceMemory<size_t> d_result_shape(result_rank);

        if (!d_cond_shape.valid() || !d_x_shape.valid() ||
            !d_y_shape.valid() || !d_result_shape.valid()) {
            return;
        }

        d_cond_shape.copy_from_host(cond_shape, cond_rank);
        d_x_shape.copy_from_host(x_shape, x_rank);
        d_y_shape.copy_from_host(y_shape, y_rank);
        d_result_shape.copy_from_host(result_shape, result_rank);

        int block_size = 256;
        int grid_size = (result_elements + block_size - 1) / block_size;

        where_broadcast_kernel<<<grid_size, block_size, 0, stream>>>(
            condition, x, y, result,
            d_cond_shape.get(), d_x_shape.get(), d_y_shape.get(), d_result_shape.get(),
            cond_rank, x_rank, y_rank, result_rank,
            result_elements);
    }

    __global__ void count_nonzero_bool_kernel(const unsigned char* data, unsigned long long* count, size_t n) {
        extern __shared__ unsigned long long sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;

        unsigned long long myCount = 0;
        if (i < n) {
            myCount = data[i] ? 1 : 0;
        }

        sdata[tid] = myCount;
        __syncthreads();

        // Reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(count, sdata[0]);
        }
    }

    void launch_count_nonzero_bool(const unsigned char* data, size_t* count,
                                   size_t n, cudaStream_t stream) {
        if (n == 0) {
            *count = 0;
            return;
        }

        // Use RAII for count
        CudaDeviceMemory<unsigned long long> d_count(1);
        if (!d_count.valid()) {
            *count = 0;
            return;
        }

        cudaMemsetAsync(d_count.get(), 0, sizeof(unsigned long long), stream);

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        count_nonzero_bool_kernel<<<grid_size, block_size,
                                    block_size * sizeof(unsigned long long), stream>>>(
            data, d_count.get(), n);

        // Synchronize before copying result
        cudaStreamSynchronize(stream);

        // Copy result back to size_t
        unsigned long long h_count = 0;
        d_count.copy_to_host(&h_count, 1);
        *count = static_cast<size_t>(h_count);
    }

    __global__ void count_nonzero_float_kernel(const float* data, unsigned long long* count, size_t n) {
        extern __shared__ unsigned long long sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;

        unsigned long long myCount = 0;
        if (i < n) {
            myCount = (data[i] != 0.0f) ? 1 : 0;
        }

        sdata[tid] = myCount;
        __syncthreads();

        // Reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(count, sdata[0]);
        }
    }

    void launch_count_nonzero_float(const float* data, size_t* count,
                                    size_t n, cudaStream_t stream) {
        if (n == 0) {
            *count = 0;
            return;
        }

        // Use RAII for count
        CudaDeviceMemory<unsigned long long> d_count(1);
        if (!d_count.valid()) {
            *count = 0;
            return;
        }

        cudaMemsetAsync(d_count.get(), 0, sizeof(unsigned long long), stream);

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        count_nonzero_float_kernel<<<grid_size, block_size,
                                     block_size * sizeof(unsigned long long), stream>>>(
            data, d_count.get(), n);

        // Synchronize before copying result
        cudaStreamSynchronize(stream);

        // Copy result back to size_t
        unsigned long long h_count = 0;
        d_count.copy_to_host(&h_count, 1);
        *count = static_cast<size_t>(h_count);
    }

    // ============= Indexing Operations Kernels =============
    __global__ void index_select_kernel(const float* input, const int* indices, float* output,
                                        size_t outer_size, size_t dim_size, size_t inner_size,
                                        size_t index_size, int boundary_mode) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_output = outer_size * index_size * inner_size;

        if (idx >= total_output)
            return;

        size_t outer_idx = idx / (index_size * inner_size);
        size_t idx_pos = (idx / inner_size) % index_size;
        size_t inner_idx = idx % inner_size;

        int selected_idx = indices[idx_pos];

        // Handle boundary modes
        if (boundary_mode == 1) { // Clamp
            selected_idx = max(0, min((int)dim_size - 1, selected_idx));
        } else if (boundary_mode == 2) { // Wrap
            selected_idx = ((selected_idx % (int)dim_size) + dim_size) % dim_size;
        } else if (selected_idx < 0 || selected_idx >= dim_size) {
            // Assert mode - invalid index, set to 0
            output[idx] = 0;
            return;
        }

        size_t src_idx = outer_idx * dim_size * inner_size +
                         selected_idx * inner_size + inner_idx;
        output[idx] = input[src_idx];
    }

    void launch_index_select(const float* input, const int* indices, float* output,
                             const size_t* shape, size_t rank, int dim,
                             size_t index_size, int boundary_mode, cudaStream_t stream) {
        // Calculate dimensions
        size_t outer_size = 1;
        for (int i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        size_t dim_size = shape[dim];

        size_t inner_size = 1;
        for (size_t i = dim + 1; i < rank; ++i) {
            inner_size *= shape[i];
        }

        size_t total_output = outer_size * index_size * inner_size;

        int block_size = 256;
        int grid_size = (total_output + block_size - 1) / block_size;

        index_select_kernel<<<grid_size, block_size, 0, stream>>>(
            input, indices, output, outer_size, dim_size, inner_size,
            index_size, boundary_mode);
    }

    // Fixed gather kernel with proper index calculations
    __global__ void gather_kernel(const float* input, const int* indices, float* output,
                                  size_t outer_size, size_t dim_size, size_t inner_size,
                                  size_t index_size, int boundary_mode) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_output = outer_size * index_size * inner_size;

        if (idx >= total_output)
            return;

        size_t outer_idx = idx / (index_size * inner_size);
        size_t idx_pos = (idx / inner_size) % index_size;
        size_t inner_idx = idx % inner_size;

        // Get the index for this position from indices tensor
        int gather_idx = indices[outer_idx * index_size * inner_size + idx_pos * inner_size + inner_idx];

        // Handle boundary modes
        if (boundary_mode == 1) { // Clamp
            gather_idx = max(0, min((int)dim_size - 1, gather_idx));
        } else if (boundary_mode == 2) { // Wrap
            gather_idx = ((gather_idx % (int)dim_size) + dim_size) % dim_size;
        } else if (gather_idx < 0 || gather_idx >= dim_size) {
            // Assert mode - invalid index
            output[idx] = 0;
            return;
        }

        // Calculate source index
        size_t src_idx = outer_idx * dim_size * inner_size +
                         gather_idx * inner_size + inner_idx;
        output[idx] = input[src_idx];
    }

    void launch_gather(const float* input, const int* indices, float* output,
                       const size_t* input_shape, const size_t* indices_shape,
                       size_t rank, int dim, size_t indices_elements,
                       int boundary_mode, cudaStream_t stream) {
        // Calculate dimensions
        size_t outer_size = 1;
        for (int i = 0; i < dim; ++i) {
            outer_size *= input_shape[i];
        }

        size_t dim_size = input_shape[dim];
        size_t index_size = indices_shape[dim];

        size_t inner_size = 1;
        for (size_t i = dim + 1; i < rank; ++i) {
            inner_size *= input_shape[i];
        }

        int block_size = 256;
        int grid_size = (indices_elements + block_size - 1) / block_size;

        gather_kernel<<<grid_size, block_size, 0, stream>>>(
            input, indices, output, outer_size, dim_size, inner_size,
            index_size, boundary_mode);
    }

    __global__ void take_kernel(const float* input, const int* indices, float* output,
                                size_t input_size, size_t output_size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= output_size)
            return;

        int index = indices[idx];

        // Handle negative indices
        if (index < 0) {
            index += input_size;
        }

        // Check bounds
        if (index < 0 || index >= input_size) {
            output[idx] = 0; // Or could assert/error
            return;
        }

        output[idx] = input[index];
    }

    void launch_take(const float* input, const int* indices, float* output,
                     size_t input_size, size_t output_size, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (output_size + block_size - 1) / block_size;

        take_kernel<<<grid_size, block_size, 0, stream>>>(
            input, indices, output, input_size, output_size);
    }

    // Fixed scatter kernel
    __global__ void scatter_kernel(float* output, const int* indices, const float* input,
                                   size_t outer_size, size_t dim_size, size_t inner_size,
                                   size_t index_size, int scatter_mode) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_input = outer_size * index_size * inner_size;

        if (idx >= total_input)
            return;

        size_t outer_idx = idx / (index_size * inner_size);
        size_t idx_pos = (idx / inner_size) % index_size;
        size_t inner_idx = idx % inner_size;

        // Get the scatter index for this position in the indices array
        int scatter_idx = indices[idx_pos];

        // Check bounds
        if (scatter_idx < 0 || scatter_idx >= dim_size) {
            return; // Invalid index
        }

        size_t src_idx = idx; // Source index is just the linear index in input
        size_t dst_idx = outer_idx * dim_size * inner_size +
                         scatter_idx * inner_size + inner_idx;

        // Apply scatter mode
        switch (scatter_mode) {
        case 0: // None
            output[dst_idx] = input[src_idx];
            break;
        case 1: // Add
            atomicAdd(&output[dst_idx], input[src_idx]);
            break;
        case 2: // Multiply
            // Note: atomic multiply is not native, would need custom implementation
            // For now, just overwrite
            output[dst_idx] = input[src_idx];
            break;
        case 3: // Max
            // atomicMax only works with integers, need custom for float
            // For now, just overwrite
            output[dst_idx] = input[src_idx];
            break;
        case 4: // Min
            // atomicMin only works with integers, need custom for float
            // For now, just overwrite
            output[dst_idx] = input[src_idx];
            break;
        }
    }

    void launch_scatter(float* output, const int* indices, const float* input,
                        const size_t* output_shape, const size_t* input_shape,
                        size_t rank, int dim, size_t input_elements,
                        int scatter_mode, cudaStream_t stream) {
        // Calculate dimensions
        size_t outer_size = 1;
        for (int i = 0; i < dim; ++i) {
            outer_size *= output_shape[i];
        }

        size_t dim_size = output_shape[dim];

        size_t inner_size = 1;
        for (size_t i = dim + 1; i < rank; ++i) {
            inner_size *= output_shape[i];
        }

        size_t index_size = input_shape[dim]; // Number of indices

        int block_size = 256;
        int grid_size = (input_elements + block_size - 1) / block_size;

        scatter_kernel<<<grid_size, block_size, 0, stream>>>(
            output, indices, input, outer_size, dim_size, inner_size,
            index_size, scatter_mode);
    }

    __global__ void index_fill_kernel(float* data, const int* indices, float value,
                                      size_t outer_size, size_t dim_size, size_t inner_size,
                                      size_t num_indices) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_fills = outer_size * num_indices * inner_size;

        if (idx >= total_fills)
            return;

        size_t outer_idx = idx / (num_indices * inner_size);
        size_t idx_pos = (idx / inner_size) % num_indices;
        size_t inner_idx = idx % inner_size;

        int fill_idx = indices[idx_pos];

        // Check bounds
        if (fill_idx < 0 || fill_idx >= dim_size) {
            return;
        }

        size_t dst_idx = outer_idx * dim_size * inner_size +
                         fill_idx * inner_size + inner_idx;
        data[dst_idx] = value;
    }

    void launch_index_fill(float* data, const int* indices, float value,
                           const size_t* shape, size_t rank, int dim,
                           size_t num_indices, cudaStream_t stream) {
        // Calculate dimensions
        size_t outer_size = 1;
        for (int i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        size_t dim_size = shape[dim];

        size_t inner_size = 1;
        for (size_t i = dim + 1; i < rank; ++i) {
            inner_size *= shape[i];
        }

        size_t total_fills = outer_size * num_indices * inner_size;

        int block_size = 256;
        int grid_size = (total_fills + block_size - 1) / block_size;

        index_fill_kernel<<<grid_size, block_size, 0, stream>>>(
            data, indices, value, outer_size, dim_size, inner_size, num_indices);
    }

    __global__ void index_copy_kernel(float* dst, const int* indices, const float* src,
                                      size_t outer_size, size_t dim_size, size_t inner_size,
                                      size_t num_indices) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total_copies = outer_size * num_indices * inner_size;

        if (idx >= total_copies)
            return;

        size_t outer_idx = idx / (num_indices * inner_size);
        size_t idx_pos = (idx / inner_size) % num_indices;
        size_t inner_idx = idx % inner_size;

        int copy_idx = indices[idx_pos];

        // Check bounds
        if (copy_idx < 0 || copy_idx >= dim_size) {
            return;
        }

        size_t dst_idx = outer_idx * dim_size * inner_size +
                         copy_idx * inner_size + inner_idx;
        size_t src_idx = outer_idx * num_indices * inner_size +
                         idx_pos * inner_size + inner_idx;

        dst[dst_idx] = src[src_idx];
    }

    void launch_index_copy(float* dst, const int* indices, const float* src,
                           const size_t* shape, size_t rank, int dim,
                           size_t num_indices, cudaStream_t stream) {
        // Calculate dimensions
        size_t outer_size = 1;
        for (int i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        size_t dim_size = shape[dim];

        size_t inner_size = 1;
        for (size_t i = dim + 1; i < rank; ++i) {
            inner_size *= shape[i];
        }

        size_t total_copies = outer_size * num_indices * inner_size;

        int block_size = 256;
        int grid_size = (total_copies + block_size - 1) / block_size;

        index_copy_kernel<<<grid_size, block_size, 0, stream>>>(
            dst, indices, src, outer_size, dim_size, inner_size, num_indices);
    }

    // ============= Power Operations =============

    __global__ void pow_scalar_kernel(float* data, float exponent, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = powf(data[idx], exponent);
        }
    }

    void launch_pow_scalar(float* data, float exponent, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        pow_scalar_kernel<<<grid_size, block_size, 0, stream>>>(data, exponent, n);
    }

    template <typename Op>
    __global__ void pow_broadcast_kernel(const float* a, const float* b, float* c,
                                         const size_t* a_shape, const size_t* b_shape,
                                         const size_t* c_shape,
                                         size_t a_rank, size_t b_rank, size_t c_rank,
                                         size_t c_elements, Op op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= c_elements)
            return;

        // Calculate indices
        size_t indices[10];
        size_t temp = idx;
        for (int i = c_rank - 1; i >= 0; --i) {
            indices[i] = temp % c_shape[i];
            temp /= c_shape[i];
        }

        // Map to a index
        size_t a_idx = 0;
        int a_rank_diff = c_rank - a_rank;
        size_t a_stride = 1;
        for (int i = a_rank - 1; i >= 0; --i) {
            int c_dim_idx = i + a_rank_diff;
            size_t dim_idx = (a_shape[i] == 1) ? 0 : indices[c_dim_idx];
            a_idx += dim_idx * a_stride;
            a_stride *= a_shape[i];
        }

        // Map to b index
        size_t b_idx = 0;
        int b_rank_diff = c_rank - b_rank;
        size_t b_stride = 1;
        for (int i = b_rank - 1; i >= 0; --i) {
            int c_dim_idx = i + b_rank_diff;
            size_t dim_idx = (b_shape[i] == 1) ? 0 : indices[c_dim_idx];
            b_idx += dim_idx * b_stride;
            b_stride *= b_shape[i];
        }

        c[idx] = op(a[a_idx], b[b_idx]);
    }

    struct PowOp {
        __device__ float operator()(float a, float b) const { return powf(a, b); }
    };

    void launch_pow_tensor(const float* a, const float* b, float* c,
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

        pow_broadcast_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, c, d_a_shape.get(), d_b_shape.get(), d_c_shape.get(),
            a_rank, b_rank, c_rank, c_elements, PowOp());
    }

    // ============= Boolean Broadcasting Operations =============
    __global__ void broadcast_bool_kernel(const unsigned char* src, unsigned char* dst,
                                          const size_t* src_shape, const size_t* dst_shape,
                                          size_t src_rank, size_t dst_rank,
                                          size_t dst_elements) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= dst_elements)
            return;

        // Convert linear index to multi-dimensional indices in dst
        size_t dst_indices[10]; // Max 10 dimensions
        size_t temp = idx;
        for (int i = dst_rank - 1; i >= 0; --i) {
            dst_indices[i] = temp % dst_shape[i];
            temp /= dst_shape[i];
        }

        // Map dst indices to src indices (considering broadcasting)
        size_t src_idx = 0;
        int rank_diff = dst_rank - src_rank;
        size_t src_stride = 1;

        for (int i = src_rank - 1; i >= 0; --i) {
            int dst_dim_idx = i + rank_diff;

            // If this dimension is 1 in source, always use index 0
            // Otherwise use the corresponding destination index
            size_t dim_idx = (src_shape[i] == 1) ? 0 : dst_indices[dst_dim_idx];

            src_idx += dim_idx * src_stride;
            src_stride *= src_shape[i];
        }

        dst[idx] = src[src_idx];
    }

    void launch_broadcast_bool(const unsigned char* src, unsigned char* dst,
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

        broadcast_bool_kernel<<<grid_size, block_size, 0, stream>>>(
            src, dst, d_src_shape.get(), d_dst_shape.get(), src_rank, dst_rank, dst_elements);
    }

} // namespace gs::tensor_ops