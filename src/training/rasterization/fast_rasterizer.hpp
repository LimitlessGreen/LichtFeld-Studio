/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_new.hpp"
#include "core/splat_data_new.hpp"
#include "rasterizer.hpp"

namespace gs::training {
    // Forward declaration
    class TrainingMemory;

    /**
     * @brief Forward rendering using LibTorch-free tensor library
     *
     * This is the LibTorch-free rasterization path that uses:
     * - CameraNew (uses gs::Tensor instead of torch::Tensor)
     * - SplatDataNew (uses gs::Tensor instead of torch::Tensor)
     * - Raw CUDA pointers extracted from gs::Tensor
     *
     * @param viewpoint_camera Camera with gs::Tensor-based transforms
     * @param gaussian_model Splat data with gs::Tensor attributes
     * @param bg_color Background color (3 floats on GPU)
     * @param cuda_memory Training memory buffers
     * @return RenderOutput with image, alpha, and forward context
     */
    RenderOutput fast_rasterize(
        CameraNew& viewpoint_camera,
        SplatDataNew& gaussian_model,
        float* bg_color,
        TrainingMemory& cuda_memory);

    /**
     * @brief Backward pass using LibTorch-free tensor library
     *
     * Writes gradients directly to gs::Tensor gradient buffers in SplatDataNew.
     *
     * @param grad_image Gradient w.r.t. output image [3, H, W] (row-major)
     * @param grad_alpha Gradient w.r.t. output alpha [H, W]
     * @param render_output Forward pass output (contains context)
     * @param gaussian_model Splat data (gradients written here)
     * @param viewpoint_camera Camera parameters
     */
    void fast_rasterize_backward(
        const float* grad_image,
        const float* grad_alpha,
        const RenderOutput& render_output,
        SplatDataNew& gaussian_model,
        CameraNew& viewpoint_camera);
} // namespace gs::training