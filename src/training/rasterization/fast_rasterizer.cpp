/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fast_rasterizer.hpp"

namespace gs::training {
    using torch::indexing::None;
    using torch::indexing::Slice;

    std::pair<RenderOutput, FastRasterizeContext> fast_rasterize_forward(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color) {
        // Get camera parameters
        const int width = static_cast<int>(viewpoint_camera.image_width());
        const int height = static_cast<int>(viewpoint_camera.image_height());
        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        // Get Gaussian parameters
        auto means = gaussian_model.means();
        auto raw_opacities = gaussian_model.opacity_raw();
        auto raw_scales = gaussian_model.scaling_raw();
        auto raw_rotations = gaussian_model.rotation_raw();
        auto sh0 = gaussian_model.sh0();
        auto shN = gaussian_model.shN();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        auto w2c = viewpoint_camera.world_view_transform();
        auto cam_position = viewpoint_camera.cam_position();

        // Call forward wrapper directly (no autograd)
        auto outputs = fast_gs::rasterization::forward_wrapper(
            means,
            raw_scales,
            raw_rotations,
            raw_opacities,
            sh0,
            shN,
            w2c,
            cam_position,
            active_sh_bases,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near_plane,
            far_plane);

        auto image = std::get<0>(outputs);
        auto alpha = std::get<1>(outputs);
        auto per_primitive_buffers = std::get<2>(outputs);
        auto per_tile_buffers = std::get<3>(outputs);
        auto per_instance_buffers = std::get<4>(outputs);
        auto per_bucket_buffers = std::get<5>(outputs);
        int n_visible_primitives = std::get<6>(outputs);
        int n_instances = std::get<7>(outputs);
        int n_buckets = std::get<8>(outputs);
        int primitive_primitive_indices_selector = std::get<9>(outputs);
        int instance_primitive_indices_selector = std::get<10>(outputs);

        // Prepare render output
        RenderOutput render_output;
        render_output.image = image + (1.0f - alpha) * bg_color.unsqueeze(-1).unsqueeze(-1);
        render_output.alpha = alpha;

        // Prepare context for backward
        FastRasterizeContext ctx;
        ctx.image = image;
        ctx.alpha = alpha;
        ctx.bg_color = bg_color;  // Save bg_color for alpha gradient

        // Save parameters (avoid re-fetching in backward)
        ctx.means = means;
        ctx.raw_scales = raw_scales;
        ctx.raw_rotations = raw_rotations;
        ctx.shN = shN;

        ctx.per_primitive_buffers = per_primitive_buffers;
        ctx.per_tile_buffers = per_tile_buffers;
        ctx.per_instance_buffers = per_instance_buffers;
        ctx.per_bucket_buffers = per_bucket_buffers;
        ctx.w2c = w2c;
        ctx.cam_position = cam_position;
        ctx.n_visible_primitives = n_visible_primitives;
        ctx.n_instances = n_instances;
        ctx.n_buckets = n_buckets;
        ctx.primitive_primitive_indices_selector = primitive_primitive_indices_selector;
        ctx.instance_primitive_indices_selector = instance_primitive_indices_selector;
        ctx.active_sh_bases = active_sh_bases;
        ctx.width = width;
        ctx.height = height;
        ctx.focal_x = fx;
        ctx.focal_y = fy;
        ctx.center_x = cx;
        ctx.center_y = cy;
        ctx.near_plane = near_plane;
        ctx.far_plane = far_plane;

        return {render_output, ctx};
    }

    void fast_rasterize_backward(
        const FastRasterizeContext& ctx,
        const torch::Tensor& grad_image,
        SplatData& gaussian_model) {

        // Compute gradient w.r.t. alpha from background blending
        // Forward: output_image = image + (1 - alpha) * bg_color
        // where bg_color is [3], alpha is [H, W], output_image is [3, H, W]
        //
        // Backward:
        // ∂L/∂image_raw = ∂L/∂output_image (grad_image)
        // ∂L/∂alpha = -sum_over_channels(∂L/∂output_image * bg_color)
        //
        // grad_image shape: [3, H, W] or [H, W, 3]
        // bg_color shape: [3]
        // alpha shape: [H, W]

        torch::Tensor grad_alpha;

        // Determine the layout of grad_image
        if (grad_image.size(0) == 3) {
            // Layout: [3, H, W]
            // ∂L/∂alpha[h,w] = -sum_c(grad_image[c,h,w] * bg_color[c])
            auto bg_expanded = ctx.bg_color.view({3, 1, 1});  // [3, 1, 1]
            grad_alpha = -(grad_image * bg_expanded).sum(0);  // [H, W]
        } else if (grad_image.size(2) == 3) {
            // Layout: [H, W, 3]
            // ∂L/∂alpha[h,w] = -sum_c(grad_image[h,w,c] * bg_color[c])
            auto bg_expanded = ctx.bg_color.view({1, 1, 3});  // [1, 1, 3]
            grad_alpha = -(grad_image * bg_expanded).sum(2);  // [H, W]
        } else {
            throw std::runtime_error("Unexpected grad_image shape in fast_rasterize_backward");
        }

        // Call backward wrapper directly (parameters already saved in context)
        auto grad_outputs = fast_gs::rasterization::backward_wrapper(
            gaussian_model._densification_info,
            grad_image,
            grad_alpha,
            ctx.image,
            ctx.alpha,
            ctx.means,
            ctx.raw_scales,
            ctx.raw_rotations,
            ctx.shN,
            ctx.per_primitive_buffers,
            ctx.per_tile_buffers,
            ctx.per_instance_buffers,
            ctx.per_bucket_buffers,
            ctx.w2c,
            ctx.cam_position,
            ctx.active_sh_bases,
            ctx.width,
            ctx.height,
            ctx.focal_x,
            ctx.focal_y,
            ctx.center_x,
            ctx.center_y,
            ctx.near_plane,
            ctx.far_plane,
            ctx.n_visible_primitives,
            ctx.n_instances,
            ctx.n_buckets,
            ctx.primitive_primitive_indices_selector,
            ctx.instance_primitive_indices_selector);

        auto grad_means = std::get<0>(grad_outputs);
        auto grad_scales_raw = std::get<1>(grad_outputs);
        auto grad_rotations_raw = std::get<2>(grad_outputs);
        auto grad_opacities_raw = std::get<3>(grad_outputs);
        auto grad_sh_coefficients_0 = std::get<4>(grad_outputs);
        auto grad_sh_coefficients_rest = std::get<5>(grad_outputs);
        // std::get<6>(grad_outputs) is grad_w2c - ignore for now

        // Manually accumulate gradients into the parameter tensors
        // NOTE: Gradients should already be defined and zeroed by optimizer.zero_grad()
        // If undefined (first iteration), PyTorch will allocate on first assignment
        if (!gaussian_model.means().grad().defined()) {
            gaussian_model.means().mutable_grad() = grad_means;
        } else {
            gaussian_model.means().mutable_grad().add_(grad_means);
        }

        if (!gaussian_model.scaling_raw().grad().defined()) {
            gaussian_model.scaling_raw().mutable_grad() = grad_scales_raw;
        } else {
            gaussian_model.scaling_raw().mutable_grad().add_(grad_scales_raw);
        }

        if (!gaussian_model.rotation_raw().grad().defined()) {
            gaussian_model.rotation_raw().mutable_grad() = grad_rotations_raw;
        } else {
            gaussian_model.rotation_raw().mutable_grad().add_(grad_rotations_raw);
        }

        if (!gaussian_model.opacity_raw().grad().defined()) {
            gaussian_model.opacity_raw().mutable_grad() = grad_opacities_raw;
        } else {
            gaussian_model.opacity_raw().mutable_grad().add_(grad_opacities_raw);
        }

        if (!gaussian_model.sh0().grad().defined()) {
            gaussian_model.sh0().mutable_grad() = grad_sh_coefficients_0;
        } else {
            gaussian_model.sh0().mutable_grad().add_(grad_sh_coefficients_0);
        }

        if (!gaussian_model.shN().grad().defined()) {
            gaussian_model.shN().mutable_grad() = grad_sh_coefficients_rest;
        } else {
            gaussian_model.shN().mutable_grad().add_(grad_sh_coefficients_rest);
        }
    }
} // namespace gs::training
