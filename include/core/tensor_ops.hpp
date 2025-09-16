/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace gs::tensor_ops {

    // Scalar operations
    void launch_scalar_add(float* data, float scalar, size_t n, cudaStream_t stream);
    void launch_scalar_sub(float* data, float scalar, size_t n, cudaStream_t stream);
    void launch_scalar_mul(float* data, float scalar, size_t n, cudaStream_t stream);
    void launch_scalar_div(float* data, float scalar, size_t n, cudaStream_t stream);

    // Element-wise operations (non-broadcast)
    void launch_element_add(const float* a, const float* b, float* c, size_t n, cudaStream_t stream);
    void launch_element_sub(const float* a, const float* b, float* c, size_t n, cudaStream_t stream);
    void launch_element_mul(const float* a, const float* b, float* c, size_t n, cudaStream_t stream);
    void launch_element_div(const float* a, const float* b, float* c, size_t n, cudaStream_t stream);

    // In-place element-wise operations
    void launch_element_add_inplace(float* a, const float* b, size_t n, cudaStream_t stream);
    void launch_element_sub_inplace(float* a, const float* b, size_t n, cudaStream_t stream);
    void launch_element_mul_inplace(float* a, const float* b, size_t n, cudaStream_t stream);
    void launch_element_div_inplace(float* a, const float* b, size_t n, cudaStream_t stream);

    // Broadcasting operations
    void launch_broadcast(const float* src, float* dst,
                          const size_t* src_shape, const size_t* dst_shape,
                          size_t src_rank, size_t dst_rank,
                          size_t dst_elements, cudaStream_t stream);

    void launch_broadcast_add(const float* a, const float* b, float* c,
                              const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                              size_t a_rank, size_t b_rank, size_t c_rank,
                              size_t c_elements, cudaStream_t stream);

    void launch_broadcast_sub(const float* a, const float* b, float* c,
                              const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                              size_t a_rank, size_t b_rank, size_t c_rank,
                              size_t c_elements, cudaStream_t stream);

    void launch_broadcast_mul(const float* a, const float* b, float* c,
                              const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                              size_t a_rank, size_t b_rank, size_t c_rank,
                              size_t c_elements, cudaStream_t stream);

    void launch_broadcast_div(const float* a, const float* b, float* c,
                              const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                              size_t a_rank, size_t b_rank, size_t c_rank,
                              size_t c_elements, cudaStream_t stream);

    // Math functions
    void launch_abs(float* data, size_t n, cudaStream_t stream);
    void launch_sqrt(float* data, size_t n, cudaStream_t stream);
    void launch_exp(float* data, size_t n, cudaStream_t stream);
    void launch_log(float* data, size_t n, cudaStream_t stream);
    void launch_sigmoid(float* data, size_t n, cudaStream_t stream);
    void launch_relu(float* data, size_t n, cudaStream_t stream);
    void launch_clamp(float* data, float min_val, float max_val, size_t n, cudaStream_t stream);
    void launch_pow_scalar(float* data, float exponent, size_t n, cudaStream_t stream);
    void launch_pow_tensor(const float* a, const float* b, float* c,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream);

    // Reduction operations
    void launch_reduce_sum(const float* data, float* result, size_t n, cudaStream_t stream);
    void launch_reduce_mean(const float* data, float* result, size_t n, cudaStream_t stream);
    void launch_reduce_min(const float* data, float* result, size_t n, cudaStream_t stream);
    void launch_reduce_max(const float* data, float* result, size_t n, cudaStream_t stream);

    // Matrix operations
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

    // Random operations
    void launch_uniform(float* data, size_t n, float low, float high,
                        unsigned long long seed, cudaStream_t stream);

    void launch_normal(float* data, size_t n, float mean, float std,
                       unsigned long long seed, cudaStream_t stream);

    void launch_bernoulli(float* data, size_t n, float p,
                          unsigned long long seed, cudaStream_t stream);

    void launch_randint(int* data, size_t n, int low, int high,
                        unsigned long long seed, cudaStream_t stream);

    // Matrix creation operations
    void launch_eye(float* data, size_t m, size_t n, cudaStream_t stream);
    void launch_diag(const float* diagonal, float* matrix, size_t n, cudaStream_t stream);
    void launch_extract_diag(const float* matrix, float* diagonal, size_t n, cudaStream_t stream);

    // ============= NEW: Comparison Operations =============
    void launch_compare_eq(const float* a, const float* b, unsigned char* result,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream);

    void launch_compare_lt(const float* a, const float* b, unsigned char* result,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream);

    void launch_compare_gt(const float* a, const float* b, unsigned char* result,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream);

    void launch_compare_scalar_eq(const float* a, float value, unsigned char* result,
                                  size_t n, cudaStream_t stream);

    void launch_compare_scalar_lt(const float* a, float value, unsigned char* result,
                                  size_t n, cudaStream_t stream);

    void launch_compare_scalar_gt(const float* a, float value, unsigned char* result,
                                  size_t n, cudaStream_t stream);

    // ============= NEW: Logical Operations =============
    void launch_logical_and(const unsigned char* a, const unsigned char* b, unsigned char* result,
                            const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                            size_t a_rank, size_t b_rank, size_t c_rank,
                            size_t c_elements, cudaStream_t stream);

    void launch_logical_or(const unsigned char* a, const unsigned char* b, unsigned char* result,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream);

    void launch_logical_xor(const unsigned char* a, const unsigned char* b, unsigned char* result,
                            const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                            size_t a_rank, size_t b_rank, size_t c_rank,
                            size_t c_elements, cudaStream_t stream);

    void launch_logical_not(const unsigned char* a, unsigned char* result,
                            size_t n, cudaStream_t stream);

    // ============= NEW: Masking Operations =============
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

    // ============= NEW: Indexing Operations =============
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

} // namespace gs::tensor_ops
