/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
*
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace gs::training {

    void launch_background_blend(
    float* output,
    const float* base_bg,
    const float* sine_bg,
    float mix_weight,
    cudaStream_t stream = 0);

    void launch_add_scale_regularization(
        float* grad_scales,
        const float* scales_raw,
        float reg_weight,
        int n,
        cudaStream_t stream = 0);

    void launch_add_opacity_regularization(
        float* grad_opacity,
        const float* opacity_raw,
        float reg_weight,
        int n,
        cudaStream_t stream = 0);

    void launch_zero_tensor(float* data, int n, cudaStream_t stream = 0);

    void launch_fill_tensor(float* data, float value, int n, cudaStream_t stream = 0);

    void launch_compute_l1_loss_forward(
        const float* pred,
        const float* gt,
        float* diff,
        float* abs_diff,
        float* l1_grad,
        int n,
        cudaStream_t stream = 0);

    void launch_compute_mean(
        const float* data,
        float* result,
        int n,
        cudaStream_t stream = 0);

    void launch_combine_gradients(
        float* output,
        const float* grad1,
        float weight1,
        const float* grad2,
        float weight2,
        int n,
        cudaStream_t stream = 0);

    void launch_copy(
        float* dst,
        const float* src,
        int n,
        cudaStream_t stream = 0);

    void launch_compute_sine_background(
    float* output,
    int step,
    int periodR = 37,
    int periodG = 41,
    int periodB = 43,
    float jitter_amp = 0.03f,
    cudaStream_t stream = 0);

} // namespace gs::training