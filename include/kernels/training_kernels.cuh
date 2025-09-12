/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
*
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace gs::training {

    // Existing declarations...
    void launch_compute_sine_background(
        float* output,
        int step,
        int periodR, int periodG, int periodB,
        float jitter_amp,
        cudaStream_t stream);

    void launch_background_blend(
        float* output,
        const float* base_bg,
        const float* sine_bg,
        float mix_weight,
        cudaStream_t stream);

    void launch_add_scale_regularization(
        float* grad_scales,
        const float* scales_raw,
        float reg_weight,
        int n,
        cudaStream_t stream);

    void launch_add_opacity_regularization(
        float* grad_opacity,
        const float* opacity_raw,
        float reg_weight,
        int n,
        cudaStream_t stream);

    void launch_zero_tensor(float* data, int n, cudaStream_t stream);

    void launch_fill_tensor(float* data, float value, int n, cudaStream_t stream);

    void launch_compute_l1_loss_forward(
        const float* pred,
        const float* gt,
        float* diff,
        float* abs_diff,
        float* l1_grad,
        int n,
        cudaStream_t stream);

    void launch_compute_mean(
        const float* data,
        float* result,
        int n,
        cudaStream_t stream);

    void launch_combine_gradients(
        float* output,
        const float* grad1,
        float weight1,
        const float* grad2,
        float weight2,
        int n,
        cudaStream_t stream);

    void launch_copy(
        float* dst,
        const float* src,
        int n,
        cudaStream_t stream);

    // NEW: Add background blending with image
    void launch_blend_image_with_background(
        float* image,
        const float* alpha,
        const float* bg_color,
        int width,
        int height,
        cudaStream_t stream);

} // namespace gs::training