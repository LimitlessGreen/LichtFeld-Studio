/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace gs::tensor_ops {

    // Scalar operations
    void launch_scalar_mul(float* data, float scalar, size_t n, cudaStream_t stream);
    void launch_scalar_div(float* data, float scalar, size_t n, cudaStream_t stream);

    // Element-wise operations
    void launch_element_add(const float* a, const float* b, float* c, size_t n, cudaStream_t stream);
    void launch_element_sub(const float* a, const float* b, float* c, size_t n, cudaStream_t stream);
    void launch_element_mul(const float* a, const float* b, float* c, size_t n, cudaStream_t stream);
    void launch_element_div(const float* a, const float* b, float* c, size_t n, cudaStream_t stream);

    // In-place element-wise operations
    void launch_element_add_inplace(float* a, const float* b, size_t n, cudaStream_t stream);
    void launch_element_sub_inplace(float* a, const float* b, size_t n, cudaStream_t stream);
    void launch_element_mul_inplace(float* a, const float* b, size_t n, cudaStream_t stream);
    void launch_element_div_inplace(float* a, const float* b, size_t n, cudaStream_t stream);

    // Math functions
    void launch_abs(float* data, size_t n, cudaStream_t stream);
    void launch_sqrt(float* data, size_t n, cudaStream_t stream);
    void launch_exp(float* data, size_t n, cudaStream_t stream);
    void launch_log(float* data, size_t n, cudaStream_t stream);
    void launch_sigmoid(float* data, size_t n, cudaStream_t stream);
    void launch_clamp(float* data, float min_val, float max_val, size_t n, cudaStream_t stream);

    // Reduction operations
    void launch_reduce_sum(const float* data, float* result, size_t n, cudaStream_t stream);
    void launch_reduce_mean(const float* data, float* result, size_t n, cudaStream_t stream);
    void launch_reduce_min(const float* data, float* result, size_t n, cudaStream_t stream);
    void launch_reduce_max(const float* data, float* result, size_t n, cudaStream_t stream);

} // namespace gs::tensor_ops
