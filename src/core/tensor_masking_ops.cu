/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include "core/cuda_memory_guard.hpp"
#include <cub/cub.cuh>

namespace gs::tensor_ops {

// ============= Import broadcast index calculator =============
// This function is defined in tensor_ops.cu
__device__ inline size_t compute_broadcast_index(
    size_t idx, const size_t* src_shape, size_t src_rank,
    const size_t* dst_shape, size_t dst_rank) {

    size_t src_idx = 0, dst_stride = 1;

    #pragma unroll 8
    for (int i = dst_rank - 1; i >= 0; --i) {
        size_t dst_coord = (idx / dst_stride) % dst_shape[i];
        int src_dim = i - (dst_rank - src_rank);

        if (src_dim >= 0) {
            size_t src_coord = (src_shape[src_dim] == 1) ? 0 : dst_coord;
            size_t src_stride = 1;
            for (int j = src_dim + 1; j < src_rank; ++j) {
                src_stride *= src_shape[j];
            }
            src_idx += src_coord * src_stride;
        }

        dst_stride *= dst_shape[i];
    }

    return src_idx;
}

// ============= Comparison Kernels =============
__global__ void compare_eq_kernel(const float* a, const float* b, unsigned char* c,
                                  const size_t* shapes, size_t info, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t a_rank = info & 0x1F;
    size_t b_rank = (info >> 5) & 0x1F;
    size_t c_rank = (info >> 10) & 0x1F;
    bool fast_path = info & 0x8000;

    if (fast_path) {
        c[idx] = (a[idx] == b[idx]) ? 1 : 0;
    } else {
        size_t a_idx = compute_broadcast_index(idx, shapes, a_rank, shapes + 20, c_rank);
        size_t b_idx = compute_broadcast_index(idx, shapes + 10, b_rank, shapes + 20, c_rank);
        c[idx] = (a[a_idx] == b[b_idx]) ? 1 : 0;
    }
}

__global__ void compare_lt_kernel(const float* a, const float* b, unsigned char* c,
                                  const size_t* shapes, size_t info, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t a_rank = info & 0x1F;
    size_t b_rank = (info >> 5) & 0x1F;
    size_t c_rank = (info >> 10) & 0x1F;
    bool fast_path = info & 0x8000;

    if (fast_path) {
        c[idx] = (a[idx] < b[idx]) ? 1 : 0;
    } else {
        size_t a_idx = compute_broadcast_index(idx, shapes, a_rank, shapes + 20, c_rank);
        size_t b_idx = compute_broadcast_index(idx, shapes + 10, b_rank, shapes + 20, c_rank);
        c[idx] = (a[a_idx] < b[b_idx]) ? 1 : 0;
    }
}

__global__ void compare_gt_kernel(const float* a, const float* b, unsigned char* c,
                                  const size_t* shapes, size_t info, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t a_rank = info & 0x1F;
    size_t b_rank = (info >> 5) & 0x1F;
    size_t c_rank = (info >> 10) & 0x1F;
    bool fast_path = info & 0x8000;

    if (fast_path) {
        c[idx] = (a[idx] > b[idx]) ? 1 : 0;
    } else {
        size_t a_idx = compute_broadcast_index(idx, shapes, a_rank, shapes + 20, c_rank);
        size_t b_idx = compute_broadcast_index(idx, shapes + 10, b_rank, shapes + 20, c_rank);
        c[idx] = (a[a_idx] > b[b_idx]) ? 1 : 0;
    }
}

// Scalar comparison kernels
__global__ void compare_scalar_eq_kernel(const float* a, float val, unsigned char* r, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) r[idx] = (a[idx] == val) ? 1 : 0;
}

__global__ void compare_scalar_lt_kernel(const float* a, float val, unsigned char* r, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) r[idx] = (a[idx] < val) ? 1 : 0;
}

__global__ void compare_scalar_gt_kernel(const float* a, float val, unsigned char* r, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) r[idx] = (a[idx] > val) ? 1 : 0;
}

// ============= Logical Operation Kernels =============
__global__ void logical_and_kernel(const unsigned char* a, const unsigned char* b,
                                   unsigned char* c, const size_t* shapes,
                                   size_t info, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t a_rank = info & 0x1F;
    size_t b_rank = (info >> 5) & 0x1F;
    size_t c_rank = (info >> 10) & 0x1F;
    bool fast_path = info & 0x8000;

    if (fast_path) {
        c[idx] = (a[idx] && b[idx]) ? 1 : 0;
    } else {
        size_t a_idx = compute_broadcast_index(idx, shapes, a_rank, shapes + 20, c_rank);
        size_t b_idx = compute_broadcast_index(idx, shapes + 10, b_rank, shapes + 20, c_rank);
        c[idx] = (a[a_idx] && b[b_idx]) ? 1 : 0;
    }
}

__global__ void logical_or_kernel(const unsigned char* a, const unsigned char* b,
                                  unsigned char* c, const size_t* shapes,
                                  size_t info, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t a_rank = info & 0x1F;
    size_t b_rank = (info >> 5) & 0x1F;
    size_t c_rank = (info >> 10) & 0x1F;
    bool fast_path = info & 0x8000;

    if (fast_path) {
        c[idx] = (a[idx] || b[idx]) ? 1 : 0;
    } else {
        size_t a_idx = compute_broadcast_index(idx, shapes, a_rank, shapes + 20, c_rank);
        size_t b_idx = compute_broadcast_index(idx, shapes + 10, b_rank, shapes + 20, c_rank);
        c[idx] = (a[a_idx] || b[b_idx]) ? 1 : 0;
    }
}

__global__ void logical_xor_kernel(const unsigned char* a, const unsigned char* b,
                                   unsigned char* c, const size_t* shapes,
                                   size_t info, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t a_rank = info & 0x1F;
    size_t b_rank = (info >> 5) & 0x1F;
    size_t c_rank = (info >> 10) & 0x1F;
    bool fast_path = info & 0x8000;

    if (fast_path) {
        c[idx] = ((a[idx] != 0) != (b[idx] != 0)) ? 1 : 0;
    } else {
        size_t a_idx = compute_broadcast_index(idx, shapes, a_rank, shapes + 20, c_rank);
        size_t b_idx = compute_broadcast_index(idx, shapes + 10, b_rank, shapes + 20, c_rank);
        c[idx] = ((a[a_idx] != 0) != (b[b_idx] != 0)) ? 1 : 0;
    }
}

__global__ void logical_not_kernel(const unsigned char* a, unsigned char* r, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) r[idx] = !a[idx];
}

// ============= Launch Functions =============
void launch_compare_eq(const float* a, const float* b, unsigned char* c,
                      const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                      size_t a_rank, size_t b_rank, size_t c_rank,
                      size_t c_elements, cudaStream_t stream) {
    static thread_local CudaDeviceMemory<size_t> shapes(30);

    size_t h_shapes[30] = {0};
    std::copy(a_shape, a_shape + a_rank, h_shapes);
    std::copy(b_shape, b_shape + b_rank, h_shapes + 10);
    std::copy(c_shape, c_shape + c_rank, h_shapes + 20);
    shapes.copy_from_host(h_shapes, 30);

    bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                     std::equal(a_shape, a_shape + a_rank, c_shape) &&
                     std::equal(b_shape, b_shape + b_rank, c_shape));
    size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

    int blocks = (c_elements + 255) / 256;
    compare_eq_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes.get(), info, c_elements);
}

void launch_compare_lt(const float* a, const float* b, unsigned char* c,
                      const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                      size_t a_rank, size_t b_rank, size_t c_rank,
                      size_t c_elements, cudaStream_t stream) {
    static thread_local CudaDeviceMemory<size_t> shapes(30);

    size_t h_shapes[30] = {0};
    std::copy(a_shape, a_shape + a_rank, h_shapes);
    std::copy(b_shape, b_shape + b_rank, h_shapes + 10);
    std::copy(c_shape, c_shape + c_rank, h_shapes + 20);
    shapes.copy_from_host(h_shapes, 30);

    bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                     std::equal(a_shape, a_shape + a_rank, c_shape) &&
                     std::equal(b_shape, b_shape + b_rank, c_shape));
    size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

    int blocks = (c_elements + 255) / 256;
    compare_lt_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes.get(), info, c_elements);
}

void launch_compare_gt(const float* a, const float* b, unsigned char* c,
                      const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                      size_t a_rank, size_t b_rank, size_t c_rank,
                      size_t c_elements, cudaStream_t stream) {
    static thread_local CudaDeviceMemory<size_t> shapes(30);

    size_t h_shapes[30] = {0};
    std::copy(a_shape, a_shape + a_rank, h_shapes);
    std::copy(b_shape, b_shape + b_rank, h_shapes + 10);
    std::copy(c_shape, c_shape + c_rank, h_shapes + 20);
    shapes.copy_from_host(h_shapes, 30);

    bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                     std::equal(a_shape, a_shape + a_rank, c_shape) &&
                     std::equal(b_shape, b_shape + b_rank, c_shape));
    size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

    int blocks = (c_elements + 255) / 256;
    compare_gt_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes.get(), info, c_elements);
}

// Scalar comparisons
void launch_compare_scalar_eq(const float* a, float val, unsigned char* r, size_t n, cudaStream_t s) {
    compare_scalar_eq_kernel<<<(n + 255) / 256, 256, 0, s>>>(a, val, r, n);
}

void launch_compare_scalar_lt(const float* a, float val, unsigned char* r, size_t n, cudaStream_t s) {
    compare_scalar_lt_kernel<<<(n + 255) / 256, 256, 0, s>>>(a, val, r, n);
}

void launch_compare_scalar_gt(const float* a, float val, unsigned char* r, size_t n, cudaStream_t s) {
    compare_scalar_gt_kernel<<<(n + 255) / 256, 256, 0, s>>>(a, val, r, n);
}

// Logical operations
void launch_logical_and(const unsigned char* a, const unsigned char* b, unsigned char* c,
                       const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                       size_t a_rank, size_t b_rank, size_t c_rank,
                       size_t c_elements, cudaStream_t stream) {
    static thread_local CudaDeviceMemory<size_t> shapes(30);

    size_t h_shapes[30] = {0};
    std::copy(a_shape, a_shape + a_rank, h_shapes);
    std::copy(b_shape, b_shape + b_rank, h_shapes + 10);
    std::copy(c_shape, c_shape + c_rank, h_shapes + 20);
    shapes.copy_from_host(h_shapes, 30);

    bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                     std::equal(a_shape, a_shape + a_rank, c_shape) &&
                     std::equal(b_shape, b_shape + b_rank, c_shape));
    size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

    int blocks = (c_elements + 255) / 256;
    logical_and_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes.get(), info, c_elements);
}

void launch_logical_or(const unsigned char* a, const unsigned char* b, unsigned char* c,
                      const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                      size_t a_rank, size_t b_rank, size_t c_rank,
                      size_t c_elements, cudaStream_t stream) {
    static thread_local CudaDeviceMemory<size_t> shapes(30);

    size_t h_shapes[30] = {0};
    std::copy(a_shape, a_shape + a_rank, h_shapes);
    std::copy(b_shape, b_shape + b_rank, h_shapes + 10);
    std::copy(c_shape, c_shape + c_rank, h_shapes + 20);
    shapes.copy_from_host(h_shapes, 30);

    bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                     std::equal(a_shape, a_shape + a_rank, c_shape) &&
                     std::equal(b_shape, b_shape + b_rank, c_shape));
    size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

    int blocks = (c_elements + 255) / 256;
    logical_or_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes.get(), info, c_elements);
}

void launch_logical_xor(const unsigned char* a, const unsigned char* b, unsigned char* c,
                       const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                       size_t a_rank, size_t b_rank, size_t c_rank,
                       size_t c_elements, cudaStream_t stream) {
    static thread_local CudaDeviceMemory<size_t> shapes(30);

    size_t h_shapes[30] = {0};
    std::copy(a_shape, a_shape + a_rank, h_shapes);
    std::copy(b_shape, b_shape + b_rank, h_shapes + 10);
    std::copy(c_shape, c_shape + c_rank, h_shapes + 20);
    shapes.copy_from_host(h_shapes, 30);

    bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                     std::equal(a_shape, a_shape + a_rank, c_shape) &&
                     std::equal(b_shape, b_shape + b_rank, c_shape));
    size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

    int blocks = (c_elements + 255) / 256;
    logical_xor_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes.get(), info, c_elements);
}

void launch_logical_not(const unsigned char* a, unsigned char* r, size_t n, cudaStream_t s) {
    logical_not_kernel<<<(n + 255) / 256, 256, 0, s>>>(a, r, n);
}

// ============= Masking Operations =============
__global__ void masked_fill_kernel(float* data, const unsigned char* mask, float val, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && mask[idx]) data[idx] = val;
}

void launch_masked_fill(float* data, const unsigned char* mask, float val, size_t n, cudaStream_t s) {
    masked_fill_kernel<<<(n + 255) / 256, 256, 0, s>>>(data, mask, val, n);
}

__global__ void masked_select_compact_kernel(const float* input, const unsigned char* mask,
                                            float* output, const int* scan, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && mask[idx]) {
        output[scan[idx]] = input[idx];
    }
}

void launch_masked_select(const float* input, const unsigned char* mask,
                         float* output, size_t n, size_t output_size, cudaStream_t stream) {
    if (n == 0 || output_size == 0) return;

    CudaDeviceMemory<int> scan_result(n);
    if (!scan_result.valid()) return;

    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, mask, scan_result.get(), n, stream);

    CudaDeviceMemory<uint8_t> temp_storage(temp_bytes);
    if (!temp_storage.valid()) return;

    cub::DeviceScan::ExclusiveSum(temp_storage.get(), temp_bytes,
                                  mask, scan_result.get(), n, stream);

    if (stream != 0) cudaStreamSynchronize(stream);

    int blocks = (n + 255) / 256;
    masked_select_compact_kernel<<<blocks, 256, 0, stream>>>(
        input, mask, output, scan_result.get(), n);
}

__global__ void masked_scatter_compact_kernel(float* data, const unsigned char* mask,
                                             const float* src, const int* scan, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && mask[idx]) {
        data[idx] = src[scan[idx]];
    }
}

void launch_masked_scatter(float* data, const unsigned char* mask,
                          const float* src, size_t n, size_t src_size, cudaStream_t stream) {
    if (n == 0 || src_size == 0) return;

    CudaDeviceMemory<int> scan_result(n);
    if (!scan_result.valid()) return;

    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, mask, scan_result.get(), n, stream);

    CudaDeviceMemory<uint8_t> temp_storage(temp_bytes);
    if (!temp_storage.valid()) return;

    cub::DeviceScan::ExclusiveSum(temp_storage.get(), temp_bytes,
                                  mask, scan_result.get(), n, stream);

    if (stream != 0) cudaStreamSynchronize(stream);

    int blocks = (n + 255) / 256;
    masked_scatter_compact_kernel<<<blocks, 256, 0, stream>>>(
        data, mask, src, scan_result.get(), n);
}

// ============= Where Operation =============
__global__ void where_kernel(const unsigned char* cond, const float* x, const float* y,
                            float* r, const size_t* shapes,
                            size_t cr, size_t xr, size_t yr, size_t rr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t c_idx = compute_broadcast_index(idx, shapes, cr, shapes + 30, rr);
    size_t x_idx = compute_broadcast_index(idx, shapes + 10, xr, shapes + 30, rr);
    size_t y_idx = compute_broadcast_index(idx, shapes + 20, yr, shapes + 30, rr);

    r[idx] = cond[c_idx] ? x[x_idx] : y[y_idx];
}

void launch_where(const unsigned char* cond, const float* x, const float* y, float* r,
                 const size_t* cond_shape, const size_t* x_shape,
                 const size_t* y_shape, const size_t* r_shape,
                 size_t cond_rank, size_t x_rank, size_t y_rank, size_t r_rank,
                 size_t total, cudaStream_t stream) {

    static thread_local CudaDeviceMemory<size_t> shapes(40);

    size_t h_shapes[40] = {0};
    std::copy(cond_shape, cond_shape + cond_rank, h_shapes);
    std::copy(x_shape, x_shape + x_rank, h_shapes + 10);
    std::copy(y_shape, y_shape + y_rank, h_shapes + 20);
    std::copy(r_shape, r_shape + r_rank, h_shapes + 30);
    shapes.copy_from_host(h_shapes, 40);

    where_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        cond, x, y, r, shapes.get(), cond_rank, x_rank, y_rank, r_rank, total);
}

// ============= Count Nonzero =============
template<typename T>
__global__ void count_nonzero_kernel(const T* data, unsigned long long* cnt, size_t n) {
    extern __shared__ unsigned long long sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n && data[i] != T(0)) ? 1 : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(cnt, sdata[0]);
}

template<typename T>
void count_nonzero_impl(const T* data, size_t* count, size_t n, cudaStream_t stream) {
    static thread_local CudaDeviceMemory<unsigned long long> d_count(1);

    cudaMemsetAsync(d_count.get(), 0, sizeof(unsigned long long), stream);

    int blocks = (n + 255) / 256;
    count_nonzero_kernel<T><<<blocks, 256, 256 * sizeof(unsigned long long), stream>>>(
        data, d_count.get(), n);

    cudaStreamSynchronize(stream);

    unsigned long long h_count;
    d_count.copy_to_host(&h_count, 1);
    *count = static_cast<size_t>(h_count);
}

void launch_count_nonzero_bool(const unsigned char* data, size_t* count,
                              size_t n, cudaStream_t stream) {
    count_nonzero_impl(data, count, n, stream);
}

void launch_count_nonzero_float(const float* data, size_t* count,
                               size_t n, cudaStream_t stream) {
    count_nonzero_impl(data, count, n, stream);
}

// ============= Index Operations =============
__global__ void index_select_kernel(const float* in, const int* idx, float* out,
                                   size_t outer, size_t dim_size, size_t inner,
                                   size_t idx_size, int boundary) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = outer * idx_size * inner;

    if (tid >= total) return;

    size_t o = tid / (idx_size * inner);
    size_t i = (tid / inner) % idx_size;
    size_t j = tid % inner;

    int sel = idx[i];

    if (boundary == 1) sel = max(0, min((int)dim_size - 1, sel));
    else if (boundary == 2) sel = ((sel % (int)dim_size) + dim_size) % dim_size;
    else if (sel < 0 || sel >= dim_size) { out[tid] = 0; return; }

    out[tid] = in[o * dim_size * inner + sel * inner + j];
}

void launch_index_select(const float* in, const int* idx, float* out,
                        const size_t* shape, size_t rank, int dim,
                        size_t idx_size, int boundary, cudaStream_t stream) {
    size_t outer = 1, inner = 1;
    for (int i = 0; i < dim; ++i) outer *= shape[i];
    for (size_t i = dim + 1; i < rank; ++i) inner *= shape[i];

    size_t total = outer * idx_size * inner;
    index_select_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        in, idx, out, outer, shape[dim], inner, idx_size, boundary);
}

// Fixed gather kernel
__global__ void gather_kernel(const float* in, const int* idx, float* out,
                              const size_t* in_shape, const size_t* idx_shape,
                              size_t in_rank, size_t idx_rank, int dim, size_t total, int boundary) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    // Calculate strides for input tensor
    size_t in_strides[10];  // Max rank
    in_strides[in_rank - 1] = 1;
    for (int i = in_rank - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
    }

    // Calculate strides for output/indices tensor
    size_t out_strides[10];  // Max rank
    out_strides[idx_rank - 1] = 1;
    for (int i = idx_rank - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * idx_shape[i + 1];
    }

    // Get coordinates in output/indices tensor
    size_t out_coords[10] = {0};
    size_t temp = tid;
    for (size_t d = 0; d < idx_rank; ++d) {
        out_coords[d] = temp / out_strides[d];
        temp %= out_strides[d];
    }

    // Get the gather index
    int gather_idx = idx[tid];

    // Apply boundary mode
    if (boundary == 1) { // Clamp
        gather_idx = max(0, min((int)in_shape[dim] - 1, gather_idx));
    } else if (boundary == 2) { // Wrap
        gather_idx = ((gather_idx % (int)in_shape[dim]) + in_shape[dim]) % in_shape[dim];
    } else if (gather_idx < 0 || gather_idx >= in_shape[dim]) { // Assert
        out[tid] = 0;
        return;
    }

    // Build source coordinates
    // Use output coordinates for all dimensions except dim
    // For dim, use the gathered index
    size_t src_idx = 0;
    for (size_t d = 0; d < in_rank; ++d) {
        size_t coord;
        if (d == dim) {
            coord = gather_idx;
        } else if (d < idx_rank) {
            coord = out_coords[d];
        } else {
            coord = 0;
        }

        // Bounds check
        if (coord >= in_shape[d]) {
            out[tid] = 0;
            return;
        }

        src_idx += coord * in_strides[d];
    }

    out[tid] = in[src_idx];
}

void launch_gather(const float* in, const int* idx, float* out,
                  const size_t* in_shape, const size_t* idx_shape,
                  size_t rank, int dim, size_t total, int boundary, cudaStream_t stream) {
    // Need to pass both shapes to the kernel
    CudaDeviceMemory<size_t> d_in_shape(10);
    CudaDeviceMemory<size_t> d_idx_shape(10);

    size_t h_in_shape[10] = {0};
    size_t h_idx_shape[10] = {0};

    // Calculate idx_rank from total and idx_shape
    // The total is the number of elements in indices tensor
    size_t idx_rank = rank; // Default to same rank as input

    // Try to infer idx_rank from the shapes if possible
    // For the gather operation, indices typically has same rank as input
    // But we can calculate it from the total elements
    size_t idx_elements = 1;
    for (size_t i = 0; i < rank; ++i) {
        if (idx_shape[i] > 0) {
            h_idx_shape[i] = idx_shape[i];
            idx_elements *= idx_shape[i];
        } else {
            break;
        }
    }

    // Find actual rank of indices
    idx_rank = 0;
    size_t check_elements = 1;
    for (size_t i = 0; i < 10; ++i) {
        if (idx_shape[i] > 0) {
            check_elements *= idx_shape[i];
            idx_rank++;
            if (check_elements == total) break;
        } else {
            break;
        }
    }

    if (idx_rank == 0) idx_rank = 1; // At least 1D

    // Copy shapes
    for (size_t i = 0; i < rank; ++i) {
        h_in_shape[i] = in_shape[i];
    }

    d_in_shape.copy_from_host(h_in_shape, 10);
    d_idx_shape.copy_from_host(h_idx_shape, 10);

    int blocks = (total + 255) / 256;
    gather_kernel<<<blocks, 256, 0, stream>>>(
        in, idx, out, d_in_shape.get(), d_idx_shape.get(),
        rank, idx_rank, dim, total, boundary);
}

__global__ void take_kernel(const float* in, const int* idx, float* out,
                           size_t in_size, size_t out_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= out_size) return;

    int pos = idx[tid];
    if (pos < 0) pos += in_size;
    out[tid] = (pos >= 0 && pos < in_size) ? in[pos] : 0;
}

void launch_take(const float* in, const int* idx, float* out,
                size_t in_size, size_t out_size, cudaStream_t stream) {
    take_kernel<<<(out_size + 255) / 256, 256, 0, stream>>>(in, idx, out, in_size, out_size);
}

// Scatter operations
__global__ void scatter_kernel(float* out, const int* idx, const float* in,
                              size_t outer, size_t dim_sz, size_t inner,
                              size_t idx_sz, int mode) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = outer * idx_sz * inner;
    if (tid >= n) return;

    size_t outer_idx = tid / (idx_sz * inner);
    size_t idx_pos = (tid / inner) % idx_sz;
    size_t inner_idx = tid % inner;

    int scatter_idx = idx[idx_pos];
    if (scatter_idx < 0 || scatter_idx >= dim_sz) return;

    size_t dst_idx = outer_idx * dim_sz * inner + scatter_idx * inner + inner_idx;

    if (mode == 1) { // Add
        atomicAdd(&out[dst_idx], in[tid]);
    } else { // Assign
        out[dst_idx] = in[tid];
    }
}

void launch_scatter(float* out, const int* idx, const float* in,
                   const size_t* out_shape, const size_t* in_shape,
                   size_t rank, int dim, size_t total, int mode, cudaStream_t stream) {
    size_t outer = 1, inner = 1;
    for (int i = 0; i < dim; ++i) outer *= out_shape[i];
    for (size_t i = dim + 1; i < rank; ++i) inner *= out_shape[i];

    scatter_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        out, idx, in, outer, out_shape[dim], inner, in_shape[dim], mode);
}

void launch_index_fill(float* data, const int* idx, float val,
                      const size_t* shape, size_t rank, int dim,
                      size_t n_idx, cudaStream_t stream) {
    CudaDeviceMemory<float> val_buffer(n_idx);
    cudaMemsetAsync(val_buffer.get(), *(int*)&val, n_idx * sizeof(float), stream);

    size_t in_shape[10] = {0};
    std::copy(shape, shape + rank, in_shape);
    in_shape[dim] = n_idx;

    launch_scatter(data, idx, val_buffer.get(), shape, in_shape, rank, dim,
                  n_idx, 0, stream);
}

void launch_index_copy(float* dst, const int* idx, const float* src,
                      const size_t* shape, size_t rank, int dim,
                      size_t n_idx, cudaStream_t stream) {
    size_t in_shape[10] = {0};
    std::copy(shape, shape + rank, in_shape);
    in_shape[dim] = n_idx;

    launch_scatter(dst, idx, src, shape, in_shape, rank, dim, n_idx, 0, stream);
}

void launch_index_add(float* dst, const int* idx, const float* src,
                     const size_t* shape, size_t rank, int dim,
                     size_t n_idx, cudaStream_t stream) {
    size_t in_shape[10] = {0};
    std::copy(shape, shape + rank, in_shape);
    in_shape[dim] = n_idx;

    launch_scatter(dst, idx, src, shape, in_shape, rank, dim, n_idx, 1, stream);
}

__global__ void index_put_kernel(float* data, const int* idx, const float* vals,
                                size_t data_size, size_t idx_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= idx_size) return;

    int pos = idx[tid];
    if (pos < 0) pos += data_size;
    if (pos >= 0 && pos < data_size) data[pos] = vals[tid];
}

void launch_index_put(float* data, const int* idx, const float* vals,
                     size_t data_size, size_t idx_size, cudaStream_t stream) {
    index_put_kernel<<<(idx_size + 255) / 256, 256, 0, stream>>>(
        data, idx, vals, data_size, idx_size);
}

// Nonzero operations
__global__ void nonzero_kernel(const float* data, int64_t* indices,
                              const int* scan, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && data[idx] != 0.0f) {
        indices[scan[idx]] = idx;
    }
}

__global__ void nonzero_bool_kernel(const unsigned char* data, int64_t* indices,
                                   const int* scan, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && data[idx]) {
        indices[scan[idx]] = idx;
    }
}

__global__ void create_nonzero_mask_kernel(const float* data, unsigned char* mask, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) mask[idx] = (data[idx] != 0.0f) ? 1 : 0;
}

void launch_nonzero(const float* data, int64_t* indices,
                   size_t n, size_t output_size, cudaStream_t stream) {
    if (n == 0 || output_size == 0) return;

    CudaDeviceMemory<unsigned char> mask(n);
    create_nonzero_mask_kernel<<<(n + 255) / 256, 256, 0, stream>>>(data, mask.get(), n);

    CudaDeviceMemory<int> scan_result(n);
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, mask.get(), scan_result.get(), n, stream);

    CudaDeviceMemory<uint8_t> temp_storage(temp_bytes);
    cub::DeviceScan::ExclusiveSum(temp_storage.get(), temp_bytes,
                                  mask.get(), scan_result.get(), n, stream);

    nonzero_kernel<<<(n + 255) / 256, 256, 0, stream>>>(data, indices, scan_result.get(), n);
}

void launch_nonzero_bool(const unsigned char* data, int64_t* indices,
                         size_t n, size_t output_size, cudaStream_t stream) {
    if (n == 0 || output_size == 0) return;

    CudaDeviceMemory<int> scan_result(n);
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, data, scan_result.get(), n, stream);

    CudaDeviceMemory<uint8_t> temp_storage(temp_bytes);
    cub::DeviceScan::ExclusiveSum(temp_storage.get(), temp_bytes,
                                  data, scan_result.get(), n, stream);

    nonzero_bool_kernel<<<(n + 255) / 256, 256, 0, stream>>>(data, indices, scan_result.get(), n);
}
} // namespace gs::tensor_ops