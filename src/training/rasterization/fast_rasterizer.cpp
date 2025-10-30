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

        const int n_primitives = means.size(0);
        const int total_bases_sh_rest = shN.size(1);

        // Allocate output tensors
        const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
        torch::Tensor image = torch::empty({3, height, width}, float_options);
        torch::Tensor alpha = torch::empty({1, height, width}, float_options);

        // Call forward_raw with raw pointers (no PyTorch wrappers)
        auto forward_ctx = fast_gs::rasterization::forward_raw(
            reinterpret_cast<const float*>(means.data_ptr<float>()),
            reinterpret_cast<const float*>(raw_scales.data_ptr<float>()),
            reinterpret_cast<const float*>(raw_rotations.data_ptr<float>()),
            raw_opacities.data_ptr<float>(),
            reinterpret_cast<const float*>(sh0.data_ptr<float>()),
            reinterpret_cast<const float*>(shN.data_ptr<float>()),
            w2c.contiguous().data_ptr<float>(),
            cam_position.contiguous().data_ptr<float>(),
            image.data_ptr<float>(),
            alpha.data_ptr<float>(),
            n_primitives,
            active_sh_bases,
            total_bases_sh_rest,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near_plane,
            far_plane);

        // Prepare render output
        RenderOutput render_output;
        render_output.image = image + (1.0f - alpha) * bg_color.unsqueeze(-1).unsqueeze(-1);
        render_output.alpha = alpha;

        // Prepare context for backward
        FastRasterizeContext ctx;
        ctx.image = image;
        ctx.alpha = alpha;
        ctx.bg_color = bg_color; // Save bg_color for alpha gradient

        // Save parameters (avoid re-fetching in backward)
        ctx.means = means;
        ctx.raw_scales = raw_scales;
        ctx.raw_rotations = raw_rotations;
        ctx.shN = shN;
        ctx.w2c = w2c;
        ctx.cam_position = cam_position;

        // Store forward context (contains buffer pointers, frame_id, etc.)
        ctx.forward_ctx = forward_ctx;

        ctx.active_sh_bases = active_sh_bases;
        ctx.total_bases_sh_rest = total_bases_sh_rest;
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
            auto bg_expanded = ctx.bg_color.view({3, 1, 1}); // [3, 1, 1]
            grad_alpha = -(grad_image * bg_expanded).sum(0); // [H, W]
        } else if (grad_image.size(2) == 3) {
            // Layout: [H, W, 3]
            // ∂L/∂alpha[h,w] = -sum_c(grad_image[h,w,c] * bg_color[c])
            auto bg_expanded = ctx.bg_color.view({1, 1, 3}); // [1, 1, 3]
            grad_alpha = -(grad_image * bg_expanded).sum(2); // [H, W]
        } else {
            throw std::runtime_error("Unexpected grad_image shape in fast_rasterize_backward");
        }

        const int n_primitives = ctx.means.size(0);

        // Allocate gradient tensors
        const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
        torch::Tensor grad_means = torch::zeros({n_primitives, 3}, float_options);
        torch::Tensor grad_scales_raw = torch::zeros({n_primitives, 3}, float_options);
        torch::Tensor grad_rotations_raw = torch::zeros({n_primitives, 4}, float_options);
        torch::Tensor grad_opacities_raw = torch::zeros({n_primitives, 1}, float_options);
        torch::Tensor grad_sh_coefficients_0 = torch::zeros({n_primitives, 1, 3}, float_options);
        torch::Tensor grad_sh_coefficients_rest = torch::zeros({n_primitives, ctx.total_bases_sh_rest, 3}, float_options);
        torch::Tensor grad_w2c = torch::zeros_like(ctx.w2c, float_options);

        // Call backward_raw with raw pointers
        const bool update_densification_info = gaussian_model._densification_info.size(0) > 0;
        auto backward_result = fast_gs::rasterization::backward_raw(
            update_densification_info ? gaussian_model._densification_info.data_ptr<float>() : nullptr,
            grad_image.data_ptr<float>(),
            grad_alpha.data_ptr<float>(),
            ctx.image.data_ptr<float>(),
            ctx.alpha.data_ptr<float>(),
            reinterpret_cast<const float*>(ctx.means.data_ptr<float>()),
            reinterpret_cast<const float*>(ctx.raw_scales.data_ptr<float>()),
            reinterpret_cast<const float*>(ctx.raw_rotations.data_ptr<float>()),
            reinterpret_cast<const float*>(ctx.shN.data_ptr<float>()),
            ctx.w2c.contiguous().data_ptr<float>(),
            ctx.cam_position.contiguous().data_ptr<float>(),
            ctx.forward_ctx,
            reinterpret_cast<float*>(grad_means.data_ptr<float>()),
            reinterpret_cast<float*>(grad_scales_raw.data_ptr<float>()),
            reinterpret_cast<float*>(grad_rotations_raw.data_ptr<float>()),
            grad_opacities_raw.data_ptr<float>(),
            reinterpret_cast<float*>(grad_sh_coefficients_0.data_ptr<float>()),
            reinterpret_cast<float*>(grad_sh_coefficients_rest.data_ptr<float>()),
            ctx.w2c.requires_grad() ? reinterpret_cast<float*>(grad_w2c.data_ptr<float>()) : nullptr,
            n_primitives,
            ctx.active_sh_bases,
            ctx.total_bases_sh_rest,
            ctx.width,
            ctx.height,
            ctx.focal_x,
            ctx.focal_y,
            ctx.center_x,
            ctx.center_y);

        if (!backward_result.success) {
            throw std::runtime_error(std::string("Backward failed: ") + backward_result.error_message);
        }

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
