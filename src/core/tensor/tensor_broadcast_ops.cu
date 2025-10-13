/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace gs::tensor_ops {

    // ============================================================================
    // HELPER: Thrust execution policy with stream support
    // ============================================================================

    template<typename Func>
    void run_with_thrust_policy(cudaStream_t stream, Func&& func) {
        if (stream) {
            func(thrust::cuda::par.on(stream));
        } else {
            func(thrust::cuda::par);
        }
    }

    // ============================================================================
    // BROADCASTING INDEX FUNCTOR (for single-array broadcast)
    // ============================================================================

    template<int MaxRank = 8>
    struct broadcast_index_functor {
        int src_rank, dst_rank;
        int src_shape[MaxRank];
        int dst_shape[MaxRank];
        int src_strides[MaxRank];
        int dst_strides[MaxRank];

        broadcast_index_functor(const std::vector<size_t>& src_shape_vec,
                                const std::vector<size_t>& dst_shape_vec)
            : src_rank(src_shape_vec.size()),
              dst_rank(dst_shape_vec.size()) {

            for (int i = 0; i < src_rank; ++i) {
                src_shape[i] = static_cast<int>(src_shape_vec[i]);
            }
            for (int i = 0; i < dst_rank; ++i) {
                dst_shape[i] = static_cast<int>(dst_shape_vec[i]);
            }

            // Compute row-major strides
            if (src_rank > 0) {
                src_strides[src_rank - 1] = 1;
                for (int i = src_rank - 2; i >= 0; --i) {
                    src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
                }
            }

            if (dst_rank > 0) {
                dst_strides[dst_rank - 1] = 1;
                for (int i = dst_rank - 2; i >= 0; --i) {
                    dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1];
                }
            }
        }

        __device__ size_t operator()(size_t dst_linear_idx) const {
            size_t src_idx = 0;
            size_t remaining = dst_linear_idx;

            for (int i = 0; i < dst_rank; ++i) {
                int dst_coord = remaining / dst_strides[i];
                remaining %= dst_strides[i];

                int offset = dst_rank - src_rank;
                if (i >= offset) {
                    int src_dim = i - offset;
                    int src_coord = (src_shape[src_dim] == 1) ? 0 : dst_coord;
                    src_idx += src_coord * src_strides[src_dim];
                }
            }

            return src_idx;
        }
    };

    // ============================================================================
    // SINGLE-ARRAY BROADCASTING (Generic)
    // ============================================================================

    template<typename T>
    void launch_broadcast_generic(const T* src, T* dst,
                                   const size_t* src_shape, const size_t* dst_shape,
                                   size_t src_rank, size_t dst_rank,
                                   size_t dst_elements, cudaStream_t stream) {
        if (dst_elements == 0) return;

        std::vector<size_t> src_vec(src_shape, src_shape + src_rank);
        std::vector<size_t> dst_vec(dst_shape, dst_shape + dst_rank);

        auto src_ptr = thrust::device_pointer_cast(src);
        auto dst_ptr = thrust::device_pointer_cast(dst);

        broadcast_index_functor<> index_mapper(src_vec, dst_vec);

        auto counting = thrust::make_counting_iterator<size_t>(0);
        auto src_index_iter = thrust::make_transform_iterator(counting, index_mapper);
        auto permuted_src = thrust::make_permutation_iterator(src_ptr, src_index_iter);

        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::copy(policy, permuted_src, permuted_src + dst_elements, dst_ptr);
        });
    }

    void launch_broadcast(const float* src, float* dst,
                          const size_t* src_shape, const size_t* dst_shape,
                          size_t src_rank, size_t dst_rank,
                          size_t dst_elements, cudaStream_t stream) {
        launch_broadcast_generic(src, dst, src_shape, dst_shape, src_rank, dst_rank, dst_elements, stream);
    }

    void launch_broadcast_bool(const unsigned char* src, unsigned char* dst,
                               const size_t* src_shape, const size_t* dst_shape,
                               size_t src_rank, size_t dst_rank,
                               size_t dst_elements, cudaStream_t stream) {
        launch_broadcast_generic(src, dst, src_shape, dst_shape, src_rank, dst_rank, dst_elements, stream);
    }

    // ============================================================================
    // BINARY BROADCAST KERNEL
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

    // ============================================================================
    // BINARY BROADCAST HOST LAUNCHER TEMPLATE
    // ============================================================================

    template<typename T, typename OutputT, typename BinaryOp>
    void launch_broadcast_binary(const T* a, const T* b, OutputT* c,
                                 const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                                 size_t a_rank, size_t b_rank, size_t c_rank,
                                 size_t c_elements, BinaryOp op, cudaStream_t stream) {
        if (c_elements == 0) return;

        // Copy shapes to device
        int *d_a_shape, *d_b_shape, *d_c_shape;
        cudaMalloc(&d_a_shape, a_rank * sizeof(int));
        cudaMalloc(&d_b_shape, b_rank * sizeof(int));
        cudaMalloc(&d_c_shape, c_rank * sizeof(int));

        std::vector<int> a_vec(a_shape, a_shape + a_rank);
        std::vector<int> b_vec(b_shape, b_shape + b_rank);
        std::vector<int> c_vec(c_shape, c_shape + c_rank);

        cudaMemcpy(d_a_shape, a_vec.data(), a_rank * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_shape, b_vec.data(), b_rank * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c_shape, c_vec.data(), c_rank * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel
        int block_size = 256;
        int grid_size = (c_elements + block_size - 1) / block_size;

        broadcast_binary_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, c, d_a_shape, d_b_shape, d_c_shape,
            a_rank, b_rank, c_rank, c_elements, op);

        cudaFree(d_a_shape);
        cudaFree(d_b_shape);
        cudaFree(d_c_shape);
    }

    // ============================================================================
    // EXPLICIT INSTANTIATIONS - Required for template linking
    // ============================================================================

    // Float32 arithmetic
    template void launch_broadcast_binary(const float*, const float*, float*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, add_op<float>, cudaStream_t);
    template void launch_broadcast_binary(const float*, const float*, float*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, sub_op<float>, cudaStream_t);
    template void launch_broadcast_binary(const float*, const float*, float*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, mul_op<float>, cudaStream_t);
    template void launch_broadcast_binary(const float*, const float*, float*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, div_op<float>, cudaStream_t);
    template void launch_broadcast_binary(const float*, const float*, float*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, pow_op<float>, cudaStream_t);

    // Float32 comparison
    template void launch_broadcast_binary(const float*, const float*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, eq_op<float>, cudaStream_t);
    template void launch_broadcast_binary(const float*, const float*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, ne_op<float>, cudaStream_t);
    template void launch_broadcast_binary(const float*, const float*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, lt_op<float>, cudaStream_t);
    template void launch_broadcast_binary(const float*, const float*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, le_op<float>, cudaStream_t);
    template void launch_broadcast_binary(const float*, const float*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, gt_op<float>, cudaStream_t);
    template void launch_broadcast_binary(const float*, const float*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, ge_op<float>, cudaStream_t);

    // Int32 arithmetic
    template void launch_broadcast_binary(const int*, const int*, int*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, add_op<int>, cudaStream_t);
    template void launch_broadcast_binary(const int*, const int*, int*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, sub_op<int>, cudaStream_t);
    template void launch_broadcast_binary(const int*, const int*, int*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, mul_op<int>, cudaStream_t);
    template void launch_broadcast_binary(const int*, const int*, int*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, div_op<int>, cudaStream_t);

    // Int32 comparison
    template void launch_broadcast_binary(const int*, const int*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, eq_op<int>, cudaStream_t);
    template void launch_broadcast_binary(const int*, const int*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, ne_op<int>, cudaStream_t);
    template void launch_broadcast_binary(const int*, const int*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, lt_op<int>, cudaStream_t);
    template void launch_broadcast_binary(const int*, const int*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, le_op<int>, cudaStream_t);
    template void launch_broadcast_binary(const int*, const int*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, gt_op<int>, cudaStream_t);
    template void launch_broadcast_binary(const int*, const int*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, ge_op<int>, cudaStream_t);

    // Bool logical
    template void launch_broadcast_binary(const unsigned char*, const unsigned char*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, logical_and_op, cudaStream_t);
    template void launch_broadcast_binary(const unsigned char*, const unsigned char*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, logical_or_op, cudaStream_t);
    template void launch_broadcast_binary(const unsigned char*, const unsigned char*, unsigned char*, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, logical_xor_op, cudaStream_t);

} // namespace gs::tensor_ops