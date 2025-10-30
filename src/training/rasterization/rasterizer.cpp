/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rasterizer.hpp"
#include "core/logger.hpp"
#include <torch/torch.h>

namespace gs::training {
    using torch::indexing::None;
    using torch::indexing::Slice;

    // Explicit forward pass - returns render output and context for backward
    std::pair<RenderOutput, RasterizeContext> rasterize_forward(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier,
        bool packed,
        bool antialiased,
        RenderMode render_mode,
        const gs::geometry::BoundingBox* bounding_box) {

        // Ensure we don't use packed mode
        TORCH_CHECK(!packed, "Packed mode is not supported in this implementation");

        // Get camera parameters
        const int image_height = static_cast<int>(viewpoint_camera.image_height());
        const int image_width = static_cast<int>(viewpoint_camera.image_width());

        // Prepare viewmat and K
        auto viewmat = viewpoint_camera.world_view_transform().to(torch::kCUDA);
        TORCH_CHECK(viewmat.dim() == 3 && viewmat.size(0) == 1 && viewmat.size(1) == 4 && viewmat.size(2) == 4,
                    "viewmat must be [1, 4, 4], got ", viewmat.sizes());
        TORCH_CHECK(viewmat.is_cuda(), "viewmat must be on CUDA");

        const auto K = viewpoint_camera.K().to(torch::kCUDA);
        TORCH_CHECK(K.is_cuda(), "K must be on CUDA");

        // Initialize context
        RasterizeContext ctx;

        // Get RAW parameters (no activations)
        ctx.means_raw = gaussian_model.get_means();
        ctx.opacities_raw = gaussian_model.opacity_raw();
        ctx.scales_raw = gaussian_model.scaling_raw();
        ctx.rotations_raw = gaussian_model.rotation_raw();
        ctx.sh_coeffs = gaussian_model.get_shs();
        ctx.sh_degree = gaussian_model.get_active_sh_degree();

        // Validate parameters
        const int N = static_cast<int>(ctx.means_raw.size(0));
        TORCH_CHECK(ctx.means_raw.dim() == 2 && ctx.means_raw.size(1) == 3,
                    "means must be [N, 3], got ", ctx.means_raw.sizes());
        TORCH_CHECK(ctx.opacities_raw.dim() == 2 && ctx.opacities_raw.size(0) == N && ctx.opacities_raw.size(1) == 1,
                    "opacities_raw must be [N, 1], got ", ctx.opacities_raw.sizes());
        TORCH_CHECK(ctx.scales_raw.dim() == 2 && ctx.scales_raw.size(0) == N && ctx.scales_raw.size(1) == 3,
                    "scales_raw must be [N, 3], got ", ctx.scales_raw.sizes());
        TORCH_CHECK(ctx.rotations_raw.dim() == 2 && ctx.rotations_raw.size(0) == N && ctx.rotations_raw.size(1) == 4,
                    "rotations_raw must be [N, 4], got ", ctx.rotations_raw.sizes());

        // Device checks
        TORCH_CHECK(ctx.means_raw.is_cuda(), "means must be on CUDA");
        TORCH_CHECK(ctx.opacities_raw.is_cuda(), "opacities_raw must be on CUDA");
        TORCH_CHECK(ctx.scales_raw.is_cuda(), "scales_raw must be on CUDA");
        TORCH_CHECK(ctx.rotations_raw.is_cuda(), "rotations_raw must be on CUDA");
        TORCH_CHECK(ctx.sh_coeffs.is_cuda(), "sh_coeffs must be on CUDA");

        // Handle background color
        if (!bg_color.defined() || bg_color.numel() == 0) {
            ctx.bg_color = torch::tensor({0.0f, 0.0f, 0.0f}, ctx.means_raw.options()).view({1, 3});
        } else {
            ctx.bg_color = bg_color.view({1, -1}).to(torch::kCUDA);
            TORCH_CHECK(ctx.bg_color.size(0) == 1 && ctx.bg_color.size(1) == 3,
                        "bg_color must be reshapeable to [1, 3], got ", ctx.bg_color.sizes());
            TORCH_CHECK(ctx.bg_color.is_cuda(), "bg_color must be on CUDA");
        }

        // Rendering parameters
        ctx.eps2d = 0.3f;
        ctx.near_plane = 0.01f;
        ctx.far_plane = 1e10f; // Match FastGS
        ctx.radius_clip = 0.0f;
        ctx.tile_size = 16;
        ctx.scaling_modifier = scaling_modifier;
        ctx.calc_compensations = antialiased;
        ctx.render_mode = static_cast<int>(render_mode);
        ctx.camera_model = viewpoint_camera.camera_model_type();
        ctx.width = image_width;
        ctx.height = image_height;
        ctx.viewmat = viewmat;
        ctx.K = K;

        // Handle distortion
        if (viewpoint_camera.radial_distortion().numel() > 0) {
            auto radial_distortion_val = viewpoint_camera.radial_distortion().to(torch::kCUDA);
            TORCH_CHECK(radial_distortion_val.dim() == 1, "radial_distortion must be 1D");
            if (radial_distortion_val.size(-1) < 4) {
                radial_distortion_val = torch::nn::functional::pad(
                    radial_distortion_val,
                    torch::nn::functional::PadFuncOptions({0, 4 - radial_distortion_val.size(-1)}).mode(torch::kConstant).value(0));
            }
            ctx.radial_dist = radial_distortion_val;
        }
        if (viewpoint_camera.tangential_distortion().numel() > 0) {
            auto tangential_distortion_val = viewpoint_camera.tangential_distortion().to(torch::kCUDA);
            TORCH_CHECK(tangential_distortion_val.dim() == 1, "tangential_distortion must be 1D");
            if (tangential_distortion_val.size(-1) < 2) {
                tangential_distortion_val = torch::nn::functional::pad(
                    tangential_distortion_val,
                    torch::nn::functional::PadFuncOptions({0, 2 - tangential_distortion_val.size(-1)}).mode(torch::kConstant).value(0));
            }
            ctx.tangential_dist = tangential_distortion_val;
        }

        // Apply activation functions manually (no autograd)
        {
            torch::NoGradGuard no_grad;
            ctx.means = ctx.means_raw;                                     // No activation
            ctx.opacities = torch::sigmoid(ctx.opacities_raw).squeeze(-1); // sigmoid + squeeze
            ctx.scales = torch::exp(ctx.scales_raw);                       // exp
            ctx.rotations = torch::nn::functional::normalize(ctx.rotations_raw,
                                                             torch::nn::functional::NormalizeFuncOptions().dim(-1)); // normalize
        }

        // Call gsplat forward
        auto fwd_results = gsplat::rasterize_from_world_with_sh_fwd(
            ctx.means.contiguous(),
            ctx.rotations.contiguous(),
            ctx.scales.contiguous(),
            ctx.opacities.contiguous(),
            ctx.sh_coeffs.contiguous(),
            static_cast<uint32_t>(ctx.sh_degree),
            ctx.bg_color,
            std::nullopt, // masks
            ctx.width, ctx.height, ctx.tile_size,
            ctx.viewmat,
            std::nullopt, // viewmats1
            ctx.K,
            ctx.camera_model,
            ctx.eps2d, ctx.near_plane, ctx.far_plane, ctx.radius_clip,
            ctx.scaling_modifier,
            ctx.calc_compensations,
            ctx.render_mode,
            UnscentedTransformParameters{},
            ShutterType::GLOBAL,
            ctx.radial_dist,
            ctx.tangential_dist,
            std::nullopt // thin_prism
        );

        // Extract results
        auto rendered_image = std::get<0>(fwd_results); // [1, H, W, channels]
        ctx.rendered_image = rendered_image;            // Save for backward (needed for clamp backward)
        ctx.rendered_alpha = std::get<1>(fwd_results);
        ctx.radii = std::get<2>(fwd_results);
        ctx.means2d = std::get<3>(fwd_results);
        ctx.depths = std::get<4>(fwd_results);
        auto v_means = std::get<5>(fwd_results);
        ctx.tile_offsets = std::get<6>(fwd_results);
        ctx.flatten_ids = std::get<7>(fwd_results);
        ctx.last_ids = std::get<8>(fwd_results);
        ctx.compensations = std::get<9>(fwd_results);
        ctx.colors = v_means; // Reusing v_means for colors

        // Post-process based on render mode
        torch::Tensor final_image, final_depth;

        switch (render_mode) {
        case RenderMode::RGB:
            final_image = rendered_image;
            final_depth = torch::Tensor();
            break;

        case RenderMode::D:
            final_depth = rendered_image;
            final_image = torch::Tensor();
            break;

        case RenderMode::ED:
            final_depth = rendered_image / ctx.rendered_alpha.clamp_min(1e-10);
            final_image = torch::Tensor();
            break;

        case RenderMode::RGB_D:
            final_image = rendered_image.index({Slice(), Slice(), Slice(), Slice(None, -1)});
            final_depth = rendered_image.index({Slice(), Slice(), Slice(), Slice(-1, None)});
            break;

        case RenderMode::RGB_ED:
            final_image = rendered_image.index({Slice(), Slice(), Slice(), Slice(None, -1)});
            auto accum_depth = rendered_image.index({Slice(), Slice(), Slice(), Slice(-1, None)});
            final_depth = accum_depth / ctx.rendered_alpha.clamp_min(1e-10);
            break;
        }

        // Prepare output
        RenderOutput result;

        if (final_image.defined() && final_image.numel() > 0) {
            result.image = torch::clamp(final_image.squeeze(0).permute({2, 0, 1}), 0.0f, 1.0f);
        } else {
            result.image = torch::Tensor();
        }

        result.alpha = ctx.rendered_alpha.squeeze(0).permute({2, 0, 1});

        if (final_depth.defined() && final_depth.numel() > 0) {
            result.depth = final_depth.squeeze(0).permute({2, 0, 1});
        } else {
            result.depth = torch::Tensor();
        }

        result.means2d = ctx.means2d;
        result.depths = ctx.depths.squeeze(0);
        result.radii = std::get<0>(ctx.radii.squeeze(0).max(-1));
        result.visibility = (result.radii > 0);
        result.width = image_width;
        result.height = image_height;

        // Device checks
        if (result.image.defined() && result.image.numel() > 0) {
            TORCH_CHECK(result.image.is_cuda(), "result.image must be on CUDA");
        }
        TORCH_CHECK(result.alpha.is_cuda(), "result.alpha must be on CUDA");
        if (result.depth.defined() && result.depth.numel() > 0) {
            TORCH_CHECK(result.depth.is_cuda(), "result.depth must be on CUDA");
        }

        return {result, ctx};
    }

    // Explicit backward pass - computes gradients and accumulates them manually
    void rasterize_backward(
        const RasterizeContext& ctx,
        const torch::Tensor& grad_image,
        SplatData& gaussian_model) {
        // Step 1: Reverse the post-processing (permute, squeeze, clamp)
        // grad_image is [3, H, W] (or [1, H, W] for depth-only modes)
        // Need to convert back to [1, H, W, C] format for gsplat

        torch::Tensor grad_rendered_image;
        if (grad_image.defined() && grad_image.numel() > 0) {
            // Reverse the post-processing operations in backward order:
            // Forward was: clamp(final_image.squeeze(0).permute({2, 0, 1}), 0, 1)
            // Backward: first undo clamp, then undo permute, then undo squeeze

            // Step 1: Undo clamp - need the UNCLAMPED forward output
            // Clamp backward: grad flows through only where input was in [0,1]
            // We need to reconstruct the unclamped output
            // Forward: result.image = clamp(final_image.squeeze(0).permute({2, 0, 1}), 0, 1)
            // We need final_image before clamp to know where to mask
            auto unclamped = ctx.rendered_image.squeeze(0).permute({2, 0, 1}); // [3, H, W]
            auto clamp_mask = (unclamped >= 0.0f) & (unclamped <= 1.0f);       // [3, H, W]

            auto grad_after_clamp = grad_image * clamp_mask.to(grad_image.dtype());

            // Step 2: Reverse permute and unsqueeze
            auto grad_permuted = grad_after_clamp.permute({1, 2, 0}).unsqueeze(0); // [1, H, W, 3]
            grad_rendered_image = grad_permuted.contiguous();
        } else {
            grad_rendered_image = torch::zeros({1, ctx.height, ctx.width, 3}, ctx.means.options());
        }

        auto grad_rendered_alpha = torch::zeros_like(ctx.rendered_alpha);

        // Step 2: Call gsplat backward to get gradients on activated parameters
        auto grads = gsplat::rasterize_from_world_with_sh_bwd(
            ctx.means, ctx.rotations, ctx.scales, ctx.opacities, ctx.sh_coeffs,
            static_cast<uint32_t>(ctx.sh_degree),
            ctx.bg_color,
            std::nullopt, // masks
            ctx.width, ctx.height, ctx.tile_size,
            ctx.viewmat,
            std::nullopt, // viewmats1
            ctx.K,
            ctx.camera_model,
            ctx.eps2d, ctx.near_plane, ctx.far_plane, ctx.radius_clip,
            ctx.scaling_modifier,
            ctx.calc_compensations,
            ctx.render_mode,
            UnscentedTransformParameters{},
            ShutterType::GLOBAL,
            ctx.radial_dist,
            ctx.tangential_dist,
            std::nullopt, // thin_prism
            ctx.rendered_alpha, ctx.last_ids, ctx.tile_offsets, ctx.flatten_ids,
            ctx.colors, ctx.radii, ctx.means2d, ctx.depths, ctx.compensations,
            grad_rendered_image.contiguous(),
            grad_rendered_alpha.contiguous());

        auto v_means = std::get<0>(grads);
        auto v_quats = std::get<1>(grads);
        auto v_scales = std::get<2>(grads);
        auto v_opacities = std::get<3>(grads);
        auto v_sh_coeffs = std::get<4>(grads);

        // Step 3: Manually apply chain rule to get gradients on RAW parameters

        // Means: no activation, gradient passes through
        auto grad_means_raw = v_means.contiguous();

        // Opacity: sigmoid + squeeze
        // Forward: activated = sigmoid(raw).squeeze(-1)
        // Backward: grad_raw = grad_activated.unsqueeze(-1) * sigmoid(raw) * (1 - sigmoid(raw))
        auto sigmoid_val = torch::sigmoid(ctx.opacities_raw);
        auto sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
        auto grad_opacities_raw = (v_opacities.reshape(ctx.opacities.sizes()).unsqueeze(-1) * sigmoid_deriv).contiguous();

        // Scaling: exp
        // Forward: activated = exp(raw)
        // Backward: grad_raw = grad_activated * exp(raw)
        auto grad_scales_raw = (v_scales * ctx.scales).contiguous();

        // Rotation: normalize
        // Forward: activated = raw / ||raw||
        // Backward: grad_raw = (grad_activated - (grad_activated · activated) * activated) / ||raw||
        auto norm = ctx.rotations_raw.norm(2, -1, true);
        auto dot_product = (v_quats * ctx.rotations).sum(-1, true);
        auto grad_rotations_raw = ((v_quats - dot_product * ctx.rotations) / norm).contiguous();

        // SH coeffs: just split gradient (no chain rule needed)
        const int sh0_size = gaussian_model.sh0().size(1);
        const int shN_size = gaussian_model.shN().size(1);
        auto grad_sh0 = v_sh_coeffs.narrow(1, 0, sh0_size).contiguous();
        torch::Tensor grad_shN;
        if (shN_size > 0) {
            grad_shN = v_sh_coeffs.narrow(1, sh0_size, shN_size).contiguous();
        } else {
            grad_shN = torch::zeros({ctx.means_raw.size(0), 0, 3}, ctx.means_raw.options());
        }

        // Step 4: Accumulate gradients into parameters
        // Get REFERENCES (not copies!) to parameter tensors
        auto& means = gaussian_model.means(); // Use reference getter!
        auto& opacity_raw = gaussian_model.opacity_raw();
        auto& scaling_raw = gaussian_model.scaling_raw();
        auto& rotation_raw = gaussian_model.rotation_raw();
        auto& sh0 = gaussian_model.sh0();
        auto& shN = gaussian_model.shN();

        // Accumulate gradients (match FastGS pattern exactly)
        if (!means.grad().defined()) {
            means.mutable_grad() = grad_means_raw;
        } else {
            means.mutable_grad().add_(grad_means_raw);
        }

        if (!opacity_raw.grad().defined()) {
            opacity_raw.mutable_grad() = grad_opacities_raw;
        } else {
            opacity_raw.mutable_grad().add_(grad_opacities_raw);
        }

        if (!scaling_raw.grad().defined()) {
            scaling_raw.mutable_grad() = grad_scales_raw;
        } else {
            scaling_raw.mutable_grad().add_(grad_scales_raw);
        }

        if (!rotation_raw.grad().defined()) {
            rotation_raw.mutable_grad() = grad_rotations_raw;
        } else {
            rotation_raw.mutable_grad().add_(grad_rotations_raw);
        }

        if (!sh0.grad().defined()) {
            sh0.mutable_grad() = grad_sh0;
        } else {
            sh0.mutable_grad().add_(grad_sh0);
        }

        if (!shN.grad().defined()) {
            shN.mutable_grad() = torch::zeros_like(shN);
        }
        if (shN_size > 0) {
            shN.mutable_grad().add_(grad_shN);
        }
    }

    // Wrapper function for compatibility (calls forward only, for backward-compatible API)
    RenderOutput rasterize(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier,
        bool packed,
        bool antialiased,
        RenderMode render_mode,
        const gs::geometry::BoundingBox* bounding_box) {

        // Cast away const for forward pass (parameters need to be mutable for gradient accumulation)
        SplatData& mutable_model = const_cast<SplatData&>(gaussian_model);
        auto [output, ctx] = rasterize_forward(viewpoint_camera, mutable_model, bg_color,
                                               scaling_modifier, packed, antialiased, render_mode, bounding_box);
        return output;
    }

} // namespace gs::training
