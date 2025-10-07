/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace gs::tensor_ops {

    // ============= Clamp Scalar Operations =============
    void launch_clamp_scalar(float* data, float min_val, float max_val, size_t n, cudaStream_t stream);
    void launch_clamp_scalar_int(int* data, int min_val, int max_val, size_t n, cudaStream_t stream);

    // ============= Unified Operations =============
    void launch_unary_op(const void* input, void* output,
                         size_t n, UnaryOp op,
                         DataType dtype, cudaStream_t stream);

    void launch_unary_op_inplace(void* data, size_t n,
                                 UnaryOp op, DataType dtype,
                                 cudaStream_t stream);

    void launch_binary_op(const void* a, const void* b, void* c,
                          size_t n, BinaryOp op,
                          DataType a_dtype, DataType b_dtype, DataType c_dtype,
                          cudaStream_t stream);

    void launch_binary_scalar(const void* data, float scalar, void* result,
                              size_t n, BinaryOp op,
                              DataType src_dtype, DataType dst_dtype,
                              cudaStream_t stream);

    void launch_binary_op_inplace(void* a, const void* b, size_t n,
                                  BinaryOp op, cudaStream_t stream);

    void launch_binary_scalar_inplace(void* data, float scalar, size_t n,
                                      BinaryOp op, cudaStream_t stream);

    void launch_reduce_op(const void* input, void* output,
                          const size_t* shape, size_t rank,
                          const int* axes, size_t num_axes,
                          bool keepdim, ReduceOp op,
                          DataType dtype, cudaStream_t stream);

    void launch_ternary_op(const void* a, const void* b, const void* c, void* output,
                           size_t n, TernaryOp op,
                           DataType dtype, cudaStream_t stream);

    // ============= Load Operations =============
    void launch_load_op(void* output, const size_t* shape, size_t rank,
                        LoadOp op, const void* args,
                        DataType dtype, cudaStream_t stream);

    // ============= Type Conversions =============
    void launch_bool_to_float(const unsigned char* src, float* dst, size_t n, cudaStream_t stream);
    void launch_float_to_bool(const float* src, unsigned char* dst, size_t n, cudaStream_t stream);
    void launch_float_to_int(const float* src, int* dst, size_t n, cudaStream_t stream);
    void launch_int_to_float(const int* src, float* dst, size_t n, cudaStream_t stream);

    // ============= Broadcasting =============
    void launch_broadcast(const float* src, float* dst,
                          const size_t* src_shape, const size_t* dst_shape,
                          size_t src_rank, size_t dst_rank,
                          size_t dst_elements, cudaStream_t stream);

    void launch_broadcast_bool(const unsigned char* src, unsigned char* dst,
                               const size_t* src_shape, const size_t* dst_shape,
                               size_t src_rank, size_t dst_rank,
                               size_t dst_elements, cudaStream_t stream);

    // ============= Matrix Operations =============
    void launch_matmul(const float* a, const float* b, float* c,
                       size_t m, size_t n, size_t k,
                       cudaStream_t stream);

    void launch_batch_matmul(const float* a, const float* b, float* c,
                             size_t batch_size, size_t m, size_t n, size_t k,
                             cudaStream_t stream);

    void launch_transpose(const float* input, float* output,
                          size_t rows, size_t cols,
                          cudaStream_t stream);

    void launch_dot_product(const float* a, const float* b, float* result,
                            size_t n, cudaStream_t stream);

    // ============= Random Operations =============
    void launch_uniform(float* data, size_t n, float low, float high,
                        unsigned long long seed, cudaStream_t stream);

    void launch_normal(float* data, size_t n, float mean, float std,
                       unsigned long long seed, cudaStream_t stream);

    void launch_bernoulli(float* data, size_t n, float p,
                          unsigned long long seed, cudaStream_t stream);

    void launch_randint(int* data, size_t n, int low, int high,
                        unsigned long long seed, cudaStream_t stream);

    void launch_multinomial(const float* weights, int* samples,
                            unsigned long n, unsigned long num_samples, bool replacement,
                            unsigned long long seed, cudaStream_t stream);

    // ============= Matrix Creation Operations =============
    void launch_eye(float* data, size_t m, size_t n, cudaStream_t stream);
    void launch_diag(const float* diagonal, float* matrix, size_t n, cudaStream_t stream);
    void launch_extract_diag(const float* matrix, float* diagonal, size_t n, cudaStream_t stream);

    // ============= Masking Operations =============
    void launch_masked_select(const float* input, const unsigned char* mask,
                              float* output, size_t n, size_t output_size, cudaStream_t stream);

    void launch_masked_fill(float* data, const unsigned char* mask,
                            float value, size_t n, cudaStream_t stream);

    void launch_masked_scatter(float* data, const unsigned char* mask,
                               const float* src, size_t n, size_t src_size, cudaStream_t stream);

    void launch_where(const unsigned char* condition,
                      const float* x, const float* y, float* result,
                      const size_t* cond_shape, const size_t* x_shape,
                      const size_t* y_shape, const size_t* result_shape,
                      size_t cond_rank, size_t x_rank, size_t y_rank, size_t result_rank,
                      size_t result_elements, cudaStream_t stream);

    void launch_count_nonzero_bool(const unsigned char* data, size_t* count,
                                   size_t n, cudaStream_t stream);

    void launch_count_nonzero_float(const float* data, size_t* count,
                                    size_t n, cudaStream_t stream);

    // ============= Indexing Operations =============
    void launch_index_select(const float* input, const int* indices, float* output,
                             const size_t* shape, size_t rank, int dim,
                             size_t index_size, int boundary_mode, cudaStream_t stream);

    void launch_gather(const float* input, const int* indices, float* output,
                       const size_t* input_shape, const size_t* index_shape,
                       size_t rank, int dim, size_t total_elements,
                       int boundary_mode, cudaStream_t stream);

    void launch_take(const float* input, const int* indices, float* output,
                     size_t input_size, size_t index_size, cudaStream_t stream);

    void launch_scatter(float* output, const int* indices, const float* src,
                        const size_t* output_shape, const size_t* index_shape,
                        size_t rank, int dim, size_t total_elements,
                        int scatter_mode, cudaStream_t stream);

    void launch_index_fill(float* data, const int* indices, float value,
                           const size_t* shape, size_t rank, int dim,
                           size_t index_size, cudaStream_t stream);

    void launch_index_copy(float* data, const int* indices, const float* src,
                           const size_t* shape, size_t rank, int dim,
                           size_t index_size, cudaStream_t stream);

    void launch_index_add(float* data, const int* indices, const float* src,
                          const size_t* shape, size_t rank, int dim,
                          size_t index_size, cudaStream_t stream);

    void launch_index_put(float* data, const int* indices, const float* values,
                          size_t data_size, size_t index_size, cudaStream_t stream);

    void launch_nonzero(const float* data, int64_t* indices,
                        size_t n, size_t output_size, cudaStream_t stream);

    void launch_nonzero_bool(const unsigned char* data, int64_t* indices,
                             size_t n, size_t output_size, cudaStream_t stream);

    // ============= Cumulative Sum Operation =============
    void launch_cumsum(void* data, const size_t* shape, size_t rank,
                       int dim, DataType dtype, cudaStream_t stream);

} // namespace gs::tensor_ops