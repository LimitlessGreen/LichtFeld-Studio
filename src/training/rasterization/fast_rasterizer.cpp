/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fast_rasterizer.hpp"
#include "rasterization_api.h"
#include <torch/torch.h>

namespace gs::training {
    using torch::indexing::None;
    using torch::indexing::Slice;

    // Check the actual size of ForwardContext
    static_assert(sizeof(fast_gs::rasterization::ForwardContext) <= RenderOutput::FORWARD_CONTEXT_SIZE,
                  "ForwardContext size exceeds allocated storage. Update FORWARD_CONTEXT_SIZE in rasterizer.hpp");

    RenderOutput fast_rasterize(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color) {

        // Ensure we're not building autograd graph
        torch::NoGradGuard no_grad;

        // Get camera parameters
        const int width = static_cast<int>(viewpoint_camera.image_width());
        const int height = static_cast<int>(viewpoint_camera.image_height());
        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        // Get Gaussian parameters - no requires_grad needed
        auto means = gaussian_model.means();
        auto raw_opacities = gaussian_model.opacity_raw();
        auto raw_scales = gaussian_model.scaling_raw();
        auto raw_rotations = gaussian_model.rotation_raw();
        auto sh0 = gaussian_model.sh0();
        auto shN = gaussian_model.shN();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);
        const int n_primitives = means.size(0);
        const int total_bases_sh_rest = shN.size(1);

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        // Get camera position and world view transform
        torch::Tensor cam_position_tensor = viewpoint_camera.cam_position();
        auto w2c = viewpoint_camera.world_view_transform();

        // Ensure all tensors are contiguous
        means = means.contiguous();
        raw_scales = raw_scales.contiguous();
        raw_rotations = raw_rotations.contiguous();
        raw_opacities = raw_opacities.contiguous();
        sh0 = sh0.contiguous();
        shN = shN.contiguous();
        w2c = w2c.contiguous();

        // Allocate output tensors
        const torch::TensorOptions float_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCUDA)
            .requires_grad(false);

        torch::Tensor image = torch::empty({3, height, width}, float_options);
        torch::Tensor alpha = torch::empty({1, height, width}, float_options);

        // Call raw CUDA implementation directly
        fast_gs::rasterization::ForwardContext forward_ctx = fast_gs::rasterization::forward_raw(
            means.data_ptr<float>(),
            raw_scales.data_ptr<float>(),
            raw_rotations.data_ptr<float>(),
            raw_opacities.data_ptr<float>(),
            sh0.data_ptr<float>(),
            shN.data_ptr<float>(),
            w2c.data_ptr<float>(),
            cam_position_tensor.data_ptr<float>(),
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

        // Blend with background
        image = image + (1.0f - alpha) * bg_color.unsqueeze(-1).unsqueeze(-1);

        RenderOutput output;
        output.image = image;
        output.alpha = alpha;
        output.set_context(forward_ctx);  // Store the context
        output.width = width;
        output.height = height;

        return output;
    }

    void fast_rasterize_backward(
        const torch::Tensor& grad_image,
        const torch::Tensor& grad_alpha,
        const RenderOutput& render_output,
        SplatData& gaussian_model,
        Camera& viewpoint_camera) {

        // Ensure we're not building autograd graph
        torch::NoGradGuard no_grad;

        if (!render_output.has_context) {
            throw std::runtime_error("RenderOutput does not have a valid forward context for backward pass");
        }

        // Get the context
        const auto* forward_ctx = render_output.get_context<fast_gs::rasterization::ForwardContext>();

        // Get camera parameters
        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();
        torch::Tensor cam_position_tensor = viewpoint_camera.cam_position();
        auto w2c = viewpoint_camera.world_view_transform();

        const int n_primitives = gaussian_model.means().size(0);
        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);
        const int total_bases_sh_rest = gaussian_model.shN().size(1);

        // Ensure model has gradients allocated
        gaussian_model.ensure_grad_allocated();

        // Get mutable gradient pointers
        auto grad_means = gaussian_model.means().mutable_grad();
        auto grad_scales_raw = gaussian_model.scaling_raw().mutable_grad();
        auto grad_rotations_raw = gaussian_model.rotation_raw().mutable_grad();
        auto grad_opacities_raw = gaussian_model.opacity_raw().mutable_grad();
        auto grad_sh0 = gaussian_model.sh0().mutable_grad();
        auto grad_shN = gaussian_model.shN().mutable_grad();

        // Camera gradients if needed
        torch::Tensor grad_w2c;
        if (w2c.requires_grad()) {
            grad_w2c = torch::zeros_like(w2c);
        }

        // Ensure gradients are contiguous
        grad_means = grad_means.contiguous();
        grad_scales_raw = grad_scales_raw.contiguous();
        grad_rotations_raw = grad_rotations_raw.contiguous();
        grad_opacities_raw = grad_opacities_raw.contiguous();
        grad_sh0 = grad_sh0.contiguous();
        grad_shN = grad_shN.contiguous();

        // Call raw CUDA backward
        fast_gs::rasterization::BackwardOutputs outputs = fast_gs::rasterization::backward_raw(
            gaussian_model._densification_info.defined() ?
                gaussian_model._densification_info.data_ptr<float>() : nullptr,
            grad_image.contiguous().data_ptr<float>(),
            grad_alpha.contiguous().data_ptr<float>(),
            render_output.image.data_ptr<float>(),
            render_output.alpha.data_ptr<float>(),
            gaussian_model.means().contiguous().data_ptr<float>(),
            gaussian_model.scaling_raw().contiguous().data_ptr<float>(),
            gaussian_model.rotation_raw().contiguous().data_ptr<float>(),
            gaussian_model.shN().contiguous().data_ptr<float>(),
            w2c.contiguous().data_ptr<float>(),
            cam_position_tensor.data_ptr<float>(),
            *forward_ctx,  // Dereference the pointer to get the context
            grad_means.data_ptr<float>(),
            grad_scales_raw.data_ptr<float>(),
            grad_rotations_raw.data_ptr<float>(),
            grad_opacities_raw.data_ptr<float>(),
            grad_sh0.data_ptr<float>(),
            grad_shN.data_ptr<float>(),
            grad_w2c.defined() ? grad_w2c.data_ptr<float>() : nullptr,
            n_primitives,
            active_sh_bases,
            total_bases_sh_rest,
            render_output.width,
            render_output.height,
            fx,
            fy,
            cx,
            cy);

        if (!outputs.success) {
            throw std::runtime_error(std::string("Backward pass failed: ") +
                                   (outputs.error_message ? outputs.error_message : "Unknown error"));
        }
    }
} // namespace gs::training