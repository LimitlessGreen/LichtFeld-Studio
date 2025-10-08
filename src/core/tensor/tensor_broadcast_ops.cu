/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Thrust headers
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace gs::tensor_ops {

    // ============= Broadcasting Index Mapping =============

    // Device function for computing broadcast index
    // Maps output linear index to input linear index based on broadcast rules
    __device__ inline size_t map_broadcast_index(size_t linear_idx,
                                                 const size_t* out_shape,
                                                 const size_t* in_shape,
                                                 size_t out_rank,
                                                 size_t in_rank) {
        size_t in_idx = 0;
        size_t out_stride = 1;

        // Work backwards from last dimension
        for (int i = out_rank - 1; i >= 0; --i) {
            size_t out_coord = (linear_idx / out_stride) % out_shape[i];

            // Map to input dimension (accounting for rank difference)
            int in_dim = i - (out_rank - in_rank);
            if (in_dim >= 0) {
                // If input dim is 1, broadcast (use coord 0), else use output coord
                size_t in_coord = (in_shape[in_dim] == 1) ? 0 : out_coord;

                // Compute stride for this input dimension
                size_t in_stride = 1;
                for (size_t j = in_dim + 1; j < in_rank; ++j) {
                    in_stride *= in_shape[j];
                }

                in_idx += in_coord * in_stride;
            }

            out_stride *= out_shape[i];
        }

        return in_idx;
    }

    // ============= Broadcast Functors =============

    // Functor for mapping output indices to input indices
    template <typename T>
    struct broadcast_index_functor {
        const size_t* src_shape;
        const size_t* dst_shape;
        size_t src_rank;
        size_t dst_rank;

        broadcast_index_functor(const size_t* src, const size_t* dst,
                                size_t src_r, size_t dst_r)
            : src_shape(src),
              dst_shape(dst),
              src_rank(src_r),
              dst_rank(dst_r) {}

        __device__ size_t operator()(size_t idx) const {
            return map_broadcast_index(idx, dst_shape, src_shape, dst_rank, src_rank);
        }
    };

    // Functor for binary broadcast operations
    template <typename T, typename BinaryOp>
    struct broadcast_binary_functor {
        const size_t* a_shape;
        const size_t* b_shape;
        const size_t* c_shape;
        size_t a_rank;
        size_t b_rank;
        size_t c_rank;
        BinaryOp op;

        broadcast_binary_functor(const size_t* a, const size_t* b, const size_t* c,
                                 size_t ar, size_t br, size_t cr, BinaryOp op_)
            : a_shape(a),
              b_shape(b),
              c_shape(c),
              a_rank(ar),
              b_rank(br),
              c_rank(cr),
              op(op_) {}

        __device__ T operator()(size_t idx) const {
            size_t a_idx = map_broadcast_index(idx, c_shape, a_shape, c_rank, a_rank);
            size_t b_idx = map_broadcast_index(idx, c_shape, b_shape, c_rank, b_rank);
            return op(a_idx, b_idx);
        }
    };

    // ============= Binary Operation Functors =============

    template <typename T>
    struct add_op {
        const T* a;
        const T* b;

        __device__ T operator()(size_t a_idx, size_t b_idx) const {
            return a[a_idx] + b[b_idx];
        }
    };

    template <typename T>
    struct sub_op {
        const T* a;
        const T* b;

        __device__ T operator()(size_t a_idx, size_t b_idx) const {
            return a[a_idx] - b[b_idx];
        }
    };

    template <typename T>
    struct mul_op {
        const T* a;
        const T* b;

        __device__ T operator()(size_t a_idx, size_t b_idx) const {
            return a[a_idx] * b[b_idx];
        }
    };

    template <typename T>
    struct div_op {
        const T* a;
        const T* b;

        __device__ T operator()(size_t a_idx, size_t b_idx) const {
            return a[a_idx] / (b[b_idx] + T(1e-8));
        }
    };

    template <typename T>
    struct pow_op {
        const T* a;
        const T* b;

        __device__ T operator()(size_t a_idx, size_t b_idx) const {
            return powf(a[a_idx], b[b_idx]);
        }
    };

    // ============= Public API Functions =============

    void launch_broadcast(const float* src, float* dst,
                          const size_t* src_shape, const size_t* dst_shape,
                          size_t src_rank, size_t dst_rank,
                          size_t dst_elements, cudaStream_t stream) {
        if (dst_elements == 0)
            return;

        // Copy shapes to device using thrust::device_vector
        thrust::device_vector<size_t> d_src_shape(src_shape, src_shape + src_rank);
        thrust::device_vector<size_t> d_dst_shape(dst_shape, dst_shape + dst_rank);

        // Create device pointers
        auto src_ptr = thrust::device_pointer_cast(src);
        auto dst_ptr = thrust::device_pointer_cast(dst);

        // Create counting iterator for output indices
        auto counting_iter = thrust::make_counting_iterator<size_t>(0);

        // Create index mapping functor
        broadcast_index_functor<float> index_mapper(
            thrust::raw_pointer_cast(d_src_shape.data()),
            thrust::raw_pointer_cast(d_dst_shape.data()),
            src_rank, dst_rank);

        // Create transform iterator that maps output indices to input indices
        auto src_index_iter = thrust::make_transform_iterator(counting_iter, index_mapper);

        // Create permutation iterator that reads from source using mapped indices
        auto permuted_src = thrust::make_permutation_iterator(src_ptr, src_index_iter);

        // Copy with broadcasting
        thrust::copy(
            thrust::cuda::par.on(stream),
            permuted_src,
            permuted_src + dst_elements,
            dst_ptr);
    }

    void launch_broadcast_bool(const unsigned char* src, unsigned char* dst,
                               const size_t* src_shape, const size_t* dst_shape,
                               size_t src_rank, size_t dst_rank,
                               size_t dst_elements, cudaStream_t stream) {
        if (dst_elements == 0)
            return;

        // Copy shapes to device
        thrust::device_vector<size_t> d_src_shape(src_shape, src_shape + src_rank);
        thrust::device_vector<size_t> d_dst_shape(dst_shape, dst_shape + dst_rank);

        // Create device pointers
        auto src_ptr = thrust::device_pointer_cast(src);
        auto dst_ptr = thrust::device_pointer_cast(dst);

        // Create counting iterator
        auto counting_iter = thrust::make_counting_iterator<size_t>(0);

        // Create index mapping functor
        broadcast_index_functor<unsigned char> index_mapper(
            thrust::raw_pointer_cast(d_src_shape.data()),
            thrust::raw_pointer_cast(d_dst_shape.data()),
            src_rank, dst_rank);

        // Create transform and permutation iterators
        auto src_index_iter = thrust::make_transform_iterator(counting_iter, index_mapper);
        auto permuted_src = thrust::make_permutation_iterator(src_ptr, src_index_iter);

        // Copy with broadcasting
        thrust::copy(
            thrust::cuda::par.on(stream),
            permuted_src,
            permuted_src + dst_elements,
            dst_ptr);
    }

    // ============= Broadcasting Binary Operations =============

    // Helper template function for broadcast binary operations
    template <typename T, typename OpFunctor>
    void launch_broadcast_binary_impl(const T* a, const T* b, T* c,
                                      const size_t* a_shape, const size_t* b_shape,
                                      const size_t* c_shape,
                                      size_t a_rank, size_t b_rank, size_t c_rank,
                                      size_t c_elements, cudaStream_t stream,
                                      OpFunctor op_functor) {
        if (c_elements == 0)
            return;

        // Copy shapes to device
        thrust::device_vector<size_t> d_a_shape(a_shape, a_shape + a_rank);
        thrust::device_vector<size_t> d_b_shape(b_shape, b_shape + b_rank);
        thrust::device_vector<size_t> d_c_shape(c_shape, c_shape + c_rank);

        // Create device pointers
        auto c_ptr = thrust::device_pointer_cast(c);

        // Create counting iterator for output indices
        auto counting_iter = thrust::make_counting_iterator<size_t>(0);

        // Create broadcast binary functor
        broadcast_binary_functor<T, OpFunctor> binary_func(
            thrust::raw_pointer_cast(d_a_shape.data()),
            thrust::raw_pointer_cast(d_b_shape.data()),
            thrust::raw_pointer_cast(d_c_shape.data()),
            a_rank, b_rank, c_rank,
            op_functor);

        // Transform output indices to results
        thrust::transform(
            thrust::cuda::par.on(stream),
            counting_iter,
            counting_iter + c_elements,
            c_ptr,
            binary_func);
    }

    void launch_broadcast_add(const float* a, const float* b, float* c,
                              const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                              size_t a_rank, size_t b_rank, size_t c_rank,
                              size_t c_elements, cudaStream_t stream) {
        launch_broadcast_binary_impl(a, b, c, a_shape, b_shape, c_shape,
                                     a_rank, b_rank, c_rank, c_elements, stream,
                                     add_op<float>{a, b});
    }

    void launch_broadcast_sub(const float* a, const float* b, float* c,
                              const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                              size_t a_rank, size_t b_rank, size_t c_rank,
                              size_t c_elements, cudaStream_t stream) {
        launch_broadcast_binary_impl(a, b, c, a_shape, b_shape, c_shape,
                                     a_rank, b_rank, c_rank, c_elements, stream,
                                     sub_op<float>{a, b});
    }

    void launch_broadcast_mul(const float* a, const float* b, float* c,
                              const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                              size_t a_rank, size_t b_rank, size_t c_rank,
                              size_t c_elements, cudaStream_t stream) {
        launch_broadcast_binary_impl(a, b, c, a_shape, b_shape, c_shape,
                                     a_rank, b_rank, c_rank, c_elements, stream,
                                     mul_op<float>{a, b});
    }

    void launch_broadcast_div(const float* a, const float* b, float* c,
                              const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                              size_t a_rank, size_t b_rank, size_t c_rank,
                              size_t c_elements, cudaStream_t stream) {
        launch_broadcast_binary_impl(a, b, c, a_shape, b_shape, c_shape,
                                     a_rank, b_rank, c_rank, c_elements, stream,
                                     div_op<float>{a, b});
    }

    void launch_pow_tensor(const float* a, const float* b, float* c,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream) {
        launch_broadcast_binary_impl(a, b, c, a_shape, b_shape, c_shape,
                                     a_rank, b_rank, c_rank, c_elements, stream,
                                     pow_op<float>{a, b});
    }

} // namespace gs::tensor_ops