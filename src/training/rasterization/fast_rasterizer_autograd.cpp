/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fast_rasterizer_autograd.hpp"
#include <torch/torch.h>
#include <cstring>

namespace gs::training {
    // FastGSRasterize implementation
    torch::autograd::tensor_list FastGSRasterize::forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& means,                // [N, 3]
        const torch::Tensor& scales_raw,           // [N, 3]
        const torch::Tensor& rotations_raw,        // [N, 4]
        const torch::Tensor& opacities_raw,        // [N, 1]
        const torch::Tensor& sh_coefficients_0,    // [N, 1, 3]
        const torch::Tensor& sh_coefficients_rest, // [N, B-1, 3]
        const torch::Tensor& w2c,                  // [4, 4]
        torch::Tensor& densification_info,         // [2, N] or empty tensor
        const fast_gs::rasterization::FastGSSettings& settings) {

        // Validate inputs
        TORCH_CHECK(means.is_cuda() && means.is_contiguous(), "means must be CUDA contiguous");
        TORCH_CHECK(scales_raw.is_cuda() && scales_raw.is_contiguous(), "scales_raw must be CUDA contiguous");
        TORCH_CHECK(rotations_raw.is_cuda() && rotations_raw.is_contiguous(), "rotations_raw must be CUDA contiguous");
        TORCH_CHECK(opacities_raw.is_cuda() && opacities_raw.is_contiguous(), "opacities_raw must be CUDA contiguous");
        TORCH_CHECK(sh_coefficients_0.is_cuda() && sh_coefficients_0.is_contiguous(), "sh_coefficients_0 must be CUDA contiguous");
        TORCH_CHECK(sh_coefficients_rest.is_cuda() && sh_coefficients_rest.is_contiguous(), "sh_coefficients_rest must be CUDA contiguous");
        TORCH_CHECK(w2c.is_cuda() && w2c.is_contiguous(), "w2c must be CUDA contiguous");

        const int n_primitives = means.size(0);
        const int total_bases_sh_rest = sh_coefficients_rest.size(1);

        // Allocate output tensors
        const torch::TensorOptions float_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCUDA)
            .requires_grad(false);

        torch::Tensor image = torch::empty({3, settings.height, settings.width}, float_options);
        torch::Tensor alpha = torch::empty({1, settings.height, settings.width}, float_options);

        // Call raw CUDA implementation
        fast_gs::rasterization::ForwardContext forward_ctx = fast_gs::rasterization::forward_raw(
            means.data_ptr<float>(),
            scales_raw.data_ptr<float>(),
            rotations_raw.data_ptr<float>(),
            opacities_raw.data_ptr<float>(),
            sh_coefficients_0.data_ptr<float>(),
            sh_coefficients_rest.data_ptr<float>(),
            w2c.data_ptr<float>(),
            settings.cam_position_ptr,
            image.data_ptr<float>(),
            alpha.data_ptr<float>(),
            n_primitives,
            settings.active_sh_bases,
            total_bases_sh_rest,
            settings.width,
            settings.height,
            settings.focal_x,
            settings.focal_y,
            settings.center_x,
            settings.center_y,
            settings.near_plane,
            settings.far_plane);

        // Store context for backward - serialize it into a byte tensor
        const size_t context_size = sizeof(fast_gs::rasterization::ForwardContext);
        torch::Tensor context_tensor = torch::empty({static_cast<long long>(context_size)},
                                                    torch::TensorOptions()
                                                        .dtype(torch::kUInt8)
                                                        .device(torch::kCPU));

        // Copy context to tensor (on CPU to persist)
        std::memcpy(context_tensor.data_ptr(), &forward_ctx, context_size);

        // Mark non-differentiable tensors
        ctx->mark_non_differentiable({context_tensor, densification_info});

        // Save for backward
        ctx->save_for_backward({image,
                                alpha,
                                means,
                                scales_raw,
                                rotations_raw,
                                sh_coefficients_rest,
                                w2c,
                                densification_info,
                                context_tensor});

        // Save camera position tensor separately since it was passed as pointer
        ctx->saved_data["cam_position"] = torch::from_blob(
            const_cast<float*>(settings.cam_position_ptr),
            {3},
            float_options).clone();
        ctx->saved_data["active_sh_bases"] = settings.active_sh_bases;
        ctx->saved_data["width"] = settings.width;
        ctx->saved_data["height"] = settings.height;
        ctx->saved_data["focal_x"] = settings.focal_x;
        ctx->saved_data["focal_y"] = settings.focal_y;
        ctx->saved_data["center_x"] = settings.center_x;
        ctx->saved_data["center_y"] = settings.center_y;
        ctx->saved_data["near_plane"] = settings.near_plane;
        ctx->saved_data["far_plane"] = settings.far_plane;

        return {image, alpha};
    }

    torch::autograd::tensor_list FastGSRasterize::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
        auto grad_image = grad_outputs[0];
        auto grad_alpha = grad_outputs[1];

        auto saved = ctx->get_saved_variables();
        const torch::Tensor& image = saved[0];
        const torch::Tensor& alpha = saved[1];
        const torch::Tensor& means = saved[2];
        const torch::Tensor& scales_raw = saved[3];
        const torch::Tensor& rotations_raw = saved[4];
        const torch::Tensor& sh_coefficients_rest = saved[5];
        const torch::Tensor& w2c = saved[6];
        torch::Tensor& densification_info = saved[7];
        const torch::Tensor& context_tensor = saved[8];

        // Retrieve context from tensor
        fast_gs::rasterization::ForwardContext forward_ctx;
        std::memcpy(&forward_ctx, context_tensor.data_ptr(), sizeof(forward_ctx));

        const int n_primitives = means.size(0);
        const int total_bases_sh_rest = sh_coefficients_rest.size(1);

        // Allocate gradient tensors
        const torch::TensorOptions float_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCUDA)
            .requires_grad(false);

        torch::Tensor grad_means = torch::zeros({n_primitives, 3}, float_options);
        torch::Tensor grad_scales_raw = torch::zeros({n_primitives, 3}, float_options);
        torch::Tensor grad_rotations_raw = torch::zeros({n_primitives, 4}, float_options);
        torch::Tensor grad_opacities_raw = torch::zeros({n_primitives, 1}, float_options);
        torch::Tensor grad_sh_coefficients_0 = torch::zeros({n_primitives, 1, 3}, float_options);
        torch::Tensor grad_sh_coefficients_rest = torch::zeros({n_primitives, total_bases_sh_rest, 3}, float_options);
        torch::Tensor grad_w2c = torch::Tensor();
        if (w2c.requires_grad()) {
            grad_w2c = torch::zeros_like(w2c, float_options);
        }

        // Get cam_position
        torch::Tensor cam_position = ctx->saved_data["cam_position"].toTensor();

        // Call raw CUDA backward
        fast_gs::rasterization::BackwardOutputs outputs = fast_gs::rasterization::backward_raw(
            densification_info.numel() > 0 ? densification_info.data_ptr<float>() : nullptr,
            grad_image.data_ptr<float>(),
            grad_alpha.data_ptr<float>(),
            image.data_ptr<float>(),
            alpha.data_ptr<float>(),
            means.data_ptr<float>(),
            scales_raw.data_ptr<float>(),
            rotations_raw.data_ptr<float>(),
            sh_coefficients_rest.data_ptr<float>(),
            w2c.data_ptr<float>(),
            cam_position.data_ptr<float>(),
            forward_ctx,
            grad_means.data_ptr<float>(),
            grad_scales_raw.data_ptr<float>(),
            grad_rotations_raw.data_ptr<float>(),
            grad_opacities_raw.data_ptr<float>(),
            grad_sh_coefficients_0.data_ptr<float>(),
            grad_sh_coefficients_rest.data_ptr<float>(),
            w2c.requires_grad() ? grad_w2c.data_ptr<float>() : nullptr,
            n_primitives,
            ctx->saved_data["active_sh_bases"].toInt(),
            total_bases_sh_rest,
            ctx->saved_data["width"].toInt(),
            ctx->saved_data["height"].toInt(),
            static_cast<float>(ctx->saved_data["focal_x"].toDouble()),
            static_cast<float>(ctx->saved_data["focal_y"].toDouble()),
            static_cast<float>(ctx->saved_data["center_x"].toDouble()),
            static_cast<float>(ctx->saved_data["center_y"].toDouble()));

        if (!outputs.success) {
            TORCH_CHECK(false, "Backward pass failed: ",
                       outputs.error_message ? outputs.error_message : "Unknown error");
        }

        return {
            grad_means,
            grad_scales_raw,
            grad_rotations_raw,
            grad_opacities_raw,
            grad_sh_coefficients_0,
            grad_sh_coefficients_rest,
            grad_w2c,
            torch::Tensor(), // densification_info (no gradient)
            torch::Tensor(), // settings (no gradient)
        };
    }
} // namespace gs::training