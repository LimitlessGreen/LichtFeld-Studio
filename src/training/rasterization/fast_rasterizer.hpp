/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "rasterizer.hpp"

namespace gs::training {
    // Forward declaration
    class TrainingMemory;

    // Forward rendering without any torch dependency
    RenderOutput fast_rasterize(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        float* bg_color, // Raw pointer to 3 floats
        TrainingMemory& cuda_memory);

    // Backward pass with direct gradient writing
    void fast_rasterize_backward(
        const float* grad_image,
        const float* grad_alpha,
        const RenderOutput& render_output,
        SplatData& gaussian_model,
        Camera& viewpoint_camera);
} // namespace gs::training