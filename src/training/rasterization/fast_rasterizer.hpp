/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "rasterization_api.h"
#include "rasterizer.hpp"

namespace gs::training {
    // Forward pass context - holds intermediate buffers needed for backward
    struct FastRasterizeContext {
        torch::Tensor image;
        torch::Tensor alpha;
        torch::Tensor bg_color; // Saved for alpha gradient computation

        // Gaussian parameters (saved to avoid re-fetching in backward)
        torch::Tensor means;
        torch::Tensor raw_scales;
        torch::Tensor raw_rotations;
        torch::Tensor shN;
        torch::Tensor w2c;
        torch::Tensor cam_position;

        // Forward context (contains buffer pointers, frame_id, etc.)
        fast_gs::rasterization::ForwardContext forward_ctx;

        int active_sh_bases;
        int total_bases_sh_rest;
        int width;
        int height;
        float focal_x;
        float focal_y;
        float center_x;
        float center_y;
        float near_plane;
        float far_plane;
    };

    // Explicit forward pass - returns render output and context for backward
    std::pair<RenderOutput, FastRasterizeContext> fast_rasterize_forward(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color);

    // Explicit backward pass - computes gradients and accumulates them manually
    void fast_rasterize_backward(
        const FastRasterizeContext& ctx,
        const torch::Tensor& grad_image,
        SplatData& gaussian_model);

    // Convenience wrapper for inference (no backward needed)
    inline RenderOutput fast_rasterize(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color) {
        auto [output, ctx] = fast_rasterize_forward(viewpoint_camera, gaussian_model, bg_color);
        return output;
    }
} // namespace gs::training
