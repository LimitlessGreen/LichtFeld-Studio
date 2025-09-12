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

        // Get raw Gaussian pointers - NO TORCH OPERATIONS!
        float* means_ptr = gaussian_model.means_cuda_ptr();
        float* raw_opacities_ptr = gaussian_model.opacity_raw_cuda_ptr();
        float* raw_scales_ptr = gaussian_model.scaling_raw_cuda_ptr();
        float* raw_rotations_ptr = gaussian_model.rotation_raw_cuda_ptr();
        float* sh0_ptr = gaussian_model.sh0_cuda_ptr();
        float* shN_ptr = gaussian_model.shN_cuda_ptr();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);
        const int n_primitives = gaussian_model.size();
        const int total_bases_sh_rest = gaussian_model.shN().size(1);

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        // Get raw camera pointers - NO TORCH OPERATIONS!
        const float* w2c_ptr = viewpoint_camera.world_view_transform_cuda_ptr();
        const float* cam_position_ptr = viewpoint_camera.cam_position_cuda_ptr();

        // Allocate output tensors
        const torch::TensorOptions float_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCUDA)
            .requires_grad(false);

        torch::Tensor image = torch::empty({3, height, width}, float_options);
        torch::Tensor alpha = torch::empty({1, height, width}, float_options);

        // Call raw CUDA implementation directly with raw pointers
        fast_gs::rasterization::ForwardContext forward_ctx = fast_gs::rasterization::forward_raw(
            means_ptr,              // Raw pointer from SplatData
            raw_scales_ptr,         // Raw pointer from SplatData
            raw_rotations_ptr,      // Raw pointer from SplatData
            raw_opacities_ptr,      // Raw pointer from SplatData
            sh0_ptr,                // Raw pointer from SplatData
            shN_ptr,                // Raw pointer from SplatData
            w2c_ptr,                // Raw pointer from Camera
            cam_position_ptr,       // Raw pointer from Camera
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

        // Get raw camera pointers - NO TORCH OPERATIONS!
        const float* w2c_ptr = viewpoint_camera.world_view_transform_cuda_ptr();
        const float* cam_position_ptr = viewpoint_camera.cam_position_cuda_ptr();

        const int n_primitives = gaussian_model.size();
        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);
        const int total_bases_sh_rest = gaussian_model.shN().size(1);

        // Get raw gradient pointers - these point to the same memory that tensor.grad() uses!
        float* grad_means_ptr = gaussian_model.means_grad_cuda_ptr();
        float* grad_scales_raw_ptr = gaussian_model.scaling_grad_cuda_ptr();
        float* grad_rotations_raw_ptr = gaussian_model.rotation_grad_cuda_ptr();
        float* grad_opacities_raw_ptr = gaussian_model.opacity_grad_cuda_ptr();
        float* grad_sh0_ptr = gaussian_model.sh0_grad_cuda_ptr();
        float* grad_shN_ptr = gaussian_model.shN_grad_cuda_ptr();

        // We don't support camera gradients since camera is torch-free
        float* grad_w2c_ptr = nullptr;

        // Get raw data pointers for forward data
        float* means_ptr = gaussian_model.means_cuda_ptr();
        float* scales_raw_ptr = gaussian_model.scaling_raw_cuda_ptr();
        float* rotations_raw_ptr = gaussian_model.rotation_raw_cuda_ptr();
        float* shN_ptr = gaussian_model.shN_cuda_ptr();

        // Call raw CUDA backward with raw pointers
        fast_gs::rasterization::BackwardOutputs outputs = fast_gs::rasterization::backward_raw(
            gaussian_model._densification_info.defined() ?
                gaussian_model._densification_info.data_ptr<float>() : nullptr,
            grad_image.contiguous().data_ptr<float>(),
            grad_alpha.contiguous().data_ptr<float>(),
            render_output.image.data_ptr<float>(),
            render_output.alpha.data_ptr<float>(),
            means_ptr,              // Raw pointer from SplatData
            scales_raw_ptr,         // Raw pointer from SplatData
            rotations_raw_ptr,      // Raw pointer from SplatData
            shN_ptr,                // Raw pointer from SplatData
            w2c_ptr,                // Raw pointer from Camera
            cam_position_ptr,       // Raw pointer from Camera
            *forward_ctx,           // Dereference the pointer to get the context
            grad_means_ptr,         // Raw gradient pointer - same memory as tensor.grad()!
            grad_scales_raw_ptr,    // Raw gradient pointer - same memory as tensor.grad()!
            grad_rotations_raw_ptr, // Raw gradient pointer - same memory as tensor.grad()!
            grad_opacities_raw_ptr, // Raw gradient pointer - same memory as tensor.grad()!
            grad_sh0_ptr,           // Raw gradient pointer - same memory as tensor.grad()!
            grad_shN_ptr,           // Raw gradient pointer - same memory as tensor.grad()!
            grad_w2c_ptr,           // nullptr since we don't support camera gradients
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

        // No need to sync gradients! The optimizer's tensors already have .grad() pointing
        // to the same memory we just wrote to via grad_means_ptr etc.
    }
} // namespace gs::training