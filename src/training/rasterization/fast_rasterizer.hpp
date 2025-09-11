/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
*
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "rasterization_api.h"
#include "rasterizer.hpp"

namespace gs::training {
    // Forward rendering without autograd
    RenderOutput fast_rasterize(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color);

    // Backward pass with direct gradient writing
    void fast_rasterize_backward(
        const torch::Tensor& grad_image,
        const torch::Tensor& grad_alpha,
        const RenderOutput& render_output,
        SplatData& gaussian_model,
        Camera& viewpoint_camera);
} // namespace gs::training