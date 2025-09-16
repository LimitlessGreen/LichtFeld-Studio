/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cuda_runtime.h>

namespace gs::training {

    // Raw SSIM forward pass - writes directly to pre-allocated buffers
    void launch_ssim_forward(
        const float* img1,    // [B, C, H, W] rendered image
        const float* img2,    // [B, C, H, W] ground truth image
        float* ssim_map,      // [B, C, H, W] output SSIM values
        float* dm_dmu1,       // [B, C, H, W] partial derivatives (optional, can be nullptr)
        float* dm_dsigma1_sq, // [B, C, H, W] partial derivatives (optional, can be nullptr)
        float* dm_dsigma12,   // [B, C, H, W] partial derivatives (optional, can be nullptr)
        int batch,
        int channels,
        int height,
        int width,
        float C1,
        float C2,
        cudaStream_t stream = 0);

    // Raw SSIM backward pass - writes gradient w.r.t img1
    void launch_ssim_backward(
        const float* img1,          // [B, C, H, W] rendered image
        const float* img2,          // [B, C, H, W] ground truth image
        const float* dL_dmap,       // [B, C, H, W] gradient from loss
        const float* dm_dmu1,       // [B, C, H, W] partial derivatives from forward
        const float* dm_dsigma1_sq, // [B, C, H, W] partial derivatives from forward
        const float* dm_dsigma12,   // [B, C, H, W] partial derivatives from forward
        float* dL_dimg1,            // [B, C, H, W] output gradient
        int batch,
        int channels,
        int height,
        int width,
        float C1,
        float C2,
        cudaStream_t stream = 0);

    // Compute mean of SSIM map with optional cropping
    void launch_ssim_reduce_mean(
        const float* ssim_map, // [B, C, H, W] SSIM values
        float* mean_value,     // [1] output mean
        int batch,
        int channels,
        int height,
        int width,
        int crop_border, // How many pixels to crop from each side (0 = no crop, 5 = typical)
        cudaStream_t stream = 0);

    // Fill gradient map for SSIM backward (handles cropping)
    void launch_ssim_fill_gradient(
        float* dL_dmap,       // [B, C, H, W] gradient map to fill
        float gradient_value, // Value to fill (typically -1/N where N is number of elements)
        int batch,
        int channels,
        int height,
        int width,
        int crop_border, // How many pixels to crop from each side
        cudaStream_t stream = 0);

} // namespace gs::training