/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "geometry/bounding_box.hpp"
#include "Ops.h"
#include <torch/torch.h>

namespace gs::training {
    struct RenderOutput {
        torch::Tensor image;      // [..., channels, H, W]
        torch::Tensor alpha;      // [..., C, H, W, 1]
        torch::Tensor depth;      // [..., C, H, W, 1] - accumulated or expected depth
        torch::Tensor means2d;    // [..., C, N, 2]
        torch::Tensor depths;     // [..., N] - per-gaussian depths
        torch::Tensor radii;      // [..., N]
        torch::Tensor visibility; // [..., N]
        int width;
        int height;
    };

    enum class RenderMode {
        RGB = 0,
        D = 1,
        ED = 2,
        RGB_D = 3,
        RGB_ED = 4
    };

    inline RenderMode render_mode_from_string(const std::string& mode) {
        if (mode == "RGB")
            return RenderMode::RGB;
        else if (mode == "D")
            return RenderMode::D;
        else if (mode == "ED")
            return RenderMode::ED;
        else if (mode == "RGB_D")
            return RenderMode::RGB_D;
        else if (mode == "RGB_ED")
            return RenderMode::RGB_ED;
        else
            throw std::runtime_error("Invalid render mode: " + mode);
    }

    // Alias for backward compatibility with old code
    inline RenderMode stringToRenderMode(const std::string& mode) {
        return render_mode_from_string(mode);
    }

    // Helper function to check if render mode includes depth
    inline bool renderModeHasDepth(RenderMode mode) {
        return mode != RenderMode::RGB;
    }

    // Helper function to check if render mode includes RGB
    inline bool renderModeHasRGB(RenderMode mode) {
        return mode == RenderMode::RGB ||
               mode == RenderMode::RGB_D ||
               mode == RenderMode::RGB_ED;
    }

    // Context structure to save forward pass information for backward
    struct RasterizeContext {
        // Raw parameters (for chain rule)
        torch::Tensor means_raw;
        torch::Tensor rotations_raw;
        torch::Tensor scales_raw;
        torch::Tensor opacities_raw;

        // Activated parameters (for gsplat backward)
        torch::Tensor means;
        torch::Tensor rotations;
        torch::Tensor scales;
        torch::Tensor opacities;
        torch::Tensor sh_coeffs;

        // Camera parameters
        torch::Tensor bg_color;
        torch::Tensor viewmat;
        torch::Tensor K;
        std::optional<torch::Tensor> radial_dist;
        std::optional<torch::Tensor> tangential_dist;

        // Forward outputs (for gsplat backward)
        torch::Tensor rendered_image;  // Save for clamp backward
        torch::Tensor rendered_alpha;
        torch::Tensor last_ids;
        torch::Tensor tile_offsets;
        torch::Tensor flatten_ids;
        torch::Tensor colors;
        torch::Tensor radii;
        torch::Tensor means2d;
        torch::Tensor depths;
        torch::Tensor compensations;

        // Rendering parameters
        int sh_degree;
        int width;
        int height;
        int tile_size;
        gsplat::CameraModelType camera_model;
        float eps2d;
        float near_plane;
        float far_plane;
        float radius_clip;
        float scaling_modifier;
        bool calc_compensations;
        int render_mode;
    };

    // Explicit forward pass - returns render output and context for backward
    std::pair<RenderOutput, RasterizeContext> rasterize_forward(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier = 1.0,
        bool packed = false,
        bool antialiased = false,
        RenderMode render_mode = RenderMode::RGB,
        const gs::geometry::BoundingBox* = nullptr);

    // Explicit backward pass - computes gradients and accumulates them manually
    void rasterize_backward(
        const RasterizeContext& ctx,
        const torch::Tensor& grad_image,
        SplatData& gaussian_model);

    // Wrapper function to use gsplat backend for rendering (backward-compatible API)
    RenderOutput rasterize(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier = 1.0,
        bool packed = false,
        bool antialiased = false,
        RenderMode render_mode = RenderMode::RGB,
        const gs::geometry::BoundingBox* = nullptr);
} // namespace gs::training
