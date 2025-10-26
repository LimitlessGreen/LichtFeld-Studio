/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "validation.hpp"
#include "Ops.h"
#include <torch/torch.h>
#include "core/logger.hpp"

namespace gs::training::validation {
    using torch::indexing::None;
    using torch::indexing::Slice;

    // Manual forward/backward implementation without torch::autograd
    // Chain rule is manually applied for all activation functions
    class GsplatRasterizeFunction : public torch::autograd::Function<GsplatRasterizeFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor means_raw,          // Raw parameter (no activation)
            torch::Tensor rotations_raw,      // Raw parameter (will be normalized)
            torch::Tensor scales_raw,         // Raw parameter (will be exp)
            torch::Tensor opacities_raw,      // Raw parameter (will be sigmoid + squeeze)
            torch::Tensor sh_coeffs,          // Already activated (cat of sh0, shN)
            int sh_degree,
            torch::Tensor bg_color,
            torch::Tensor viewmat,
            torch::Tensor K,
            int width,
            int height,
            int tile_size,
            gsplat::CameraModelType camera_model,
            float eps2d,
            float near_plane,
            float far_plane,
            float radius_clip,
            float scaling_modifier,
            bool calc_compensations,
            int render_mode,
            UnscentedTransformParameters ut_params,
            std::optional<torch::Tensor> radial_dist,
            std::optional<torch::Tensor> tangential_dist
        ) {
            // Apply activation functions manually (no autograd)
            torch::Tensor means, opacities, scales, rotations;
            {
                torch::NoGradGuard no_grad;
                means = means_raw;  // No activation
                opacities = torch::sigmoid(opacities_raw).squeeze(-1);  // sigmoid + squeeze
                scales = torch::exp(scales_raw);  // exp
                rotations = torch::nn::functional::normalize(rotations_raw,
                    torch::nn::functional::NormalizeFuncOptions().dim(-1));  // normalize
            }
            auto results = gsplat::rasterize_from_world_with_sh_fwd(
                means.contiguous(),
                rotations.contiguous(),
                scales.contiguous(),
                opacities.contiguous(),
                sh_coeffs.contiguous(),
                static_cast<uint32_t>(sh_degree),
                bg_color,
                std::nullopt,  // masks
                width, height, tile_size,
                viewmat,
                std::nullopt,  // viewmats1
                K,
                camera_model,
                eps2d, near_plane, far_plane, radius_clip,
                scaling_modifier,
                calc_compensations,
                render_mode,
                ut_params,
                ShutterType::GLOBAL,
                radial_dist,
                tangential_dist,
                std::nullopt  // thin_prism
            );

            // Save RAW parameters and activated parameters for backward
            ctx->save_for_backward({means_raw, rotations_raw, scales_raw, opacities_raw,
                                    means, rotations, scales, opacities, sh_coeffs,
                                    bg_color, viewmat, K,
                                    radial_dist.has_value() ? *radial_dist : torch::Tensor(),
                                    tangential_dist.has_value() ? *tangential_dist : torch::Tensor(),
                                    std::get<1>(results), std::get<8>(results), std::get<6>(results),
                                    std::get<7>(results), std::get<5>(results), std::get<2>(results),
                                    std::get<3>(results), std::get<4>(results), std::get<9>(results)});

            ctx->saved_data["sh_degree"] = sh_degree;
            ctx->saved_data["width"] = width;
            ctx->saved_data["height"] = height;
            ctx->saved_data["tile_size"] = tile_size;
            ctx->saved_data["camera_model"] = static_cast<int>(camera_model);
            ctx->saved_data["eps2d"] = eps2d;
            ctx->saved_data["near_plane"] = near_plane;
            ctx->saved_data["far_plane"] = far_plane;
            ctx->saved_data["radius_clip"] = radius_clip;
            ctx->saved_data["scaling_modifier"] = scaling_modifier;
            ctx->saved_data["calc_compensations"] = calc_compensations;
            ctx->saved_data["render_mode"] = render_mode;

            return {std::get<0>(results), std::get<1>(results), std::get<2>(results),
                    std::get<3>(results), std::get<4>(results)};
        }

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs
        ) {
            auto saved = ctx->get_saved_variables();
            // First 4 are raw parameters
            const auto& means_raw = saved[0];
            const auto& rotations_raw = saved[1];
            const auto& scales_raw = saved[2];
            const auto& opacities_raw = saved[3];
            // Next 5 are activated parameters
            const auto& means = saved[4];
            const auto& rotations = saved[5];
            const auto& scales = saved[6];
            const auto& opacities = saved[7];
            const auto& sh_coeffs = saved[8];
            // Camera and other parameters
            const auto& bg_color = saved[9];
            const auto& viewmat = saved[10];
            const auto& K = saved[11];
            const std::optional<torch::Tensor> radial_dist = saved[12].numel() > 0 ? std::optional(saved[12]) : std::nullopt;
            const std::optional<torch::Tensor> tangential_dist = saved[13].numel() > 0 ? std::optional(saved[13]) : std::nullopt;
            // Forward outputs
            const auto& rendered_alpha = saved[14];
            const auto& last_ids = saved[15];
            const auto& tile_offsets = saved[16];
            const auto& flatten_ids = saved[17];
            const auto& colors = saved[18];
            const auto& radii = saved[19];
            const auto& means2d = saved[20];
            const auto& depths = saved[21];
            const auto& compensations = saved[22];

            int sh_degree = ctx->saved_data["sh_degree"].toInt();
            int width = ctx->saved_data["width"].toInt();
            int height = ctx->saved_data["height"].toInt();
            int tile_size = ctx->saved_data["tile_size"].toInt();
            auto camera_model = static_cast<gsplat::CameraModelType>(ctx->saved_data["camera_model"].toInt());
            float eps2d = static_cast<float>(ctx->saved_data["eps2d"].toDouble());
            float near_plane = static_cast<float>(ctx->saved_data["near_plane"].toDouble());
            float far_plane = static_cast<float>(ctx->saved_data["far_plane"].toDouble());
            float radius_clip = static_cast<float>(ctx->saved_data["radius_clip"].toDouble());
            float scaling_modifier = static_cast<float>(ctx->saved_data["scaling_modifier"].toDouble());
            bool calc_compensations = ctx->saved_data["calc_compensations"].toBool();
            int render_mode = ctx->saved_data["render_mode"].toInt();

            // Step 1: Call gsplat backward to get gradients on activated parameters
            LOG_INFO("validation backward: grad_outputs[0] shape = [{}, {}, {}, {}], norm = {:.6f}",
                     grad_outputs[0].size(0), grad_outputs[0].size(1), grad_outputs[0].size(2), grad_outputs[0].size(3),
                     grad_outputs[0].norm().item<float>());

            auto grads = gsplat::rasterize_from_world_with_sh_bwd(
                means, rotations, scales, opacities, sh_coeffs,
                static_cast<uint32_t>(sh_degree),
                bg_color,
                std::nullopt,  // masks
                width, height, tile_size,
                viewmat,
                std::nullopt,  // viewmats1
                K,
                camera_model,
                eps2d, near_plane, far_plane, radius_clip,
                scaling_modifier,
                calc_compensations,
                render_mode,
                UnscentedTransformParameters{},
                ShutterType::GLOBAL,
                radial_dist,
                tangential_dist,
                std::nullopt,  // thin_prism
                rendered_alpha, last_ids, tile_offsets, flatten_ids,
                colors, radii, means2d, depths, compensations,
                grad_outputs[0].contiguous(),
                grad_outputs[1].contiguous()
            );

            auto v_means = std::get<0>(grads);
            auto v_quats = std::get<1>(grads);
            auto v_scales = std::get<2>(grads);

            LOG_INFO("validation gsplat backward outputs: v_means norm = {:.6f}", v_means.norm().item<float>());
            auto v_opacities = std::get<3>(grads);
            auto v_sh_coeffs = std::get<4>(grads);

            // Step 2: Manually apply chain rule to get gradients on RAW parameters
            torch::Tensor grad_means_raw, grad_opacities_raw, grad_scales_raw, grad_rotations_raw;

            // Means: no activation, gradient passes through
            grad_means_raw = v_means;

            // Opacity: sigmoid + squeeze
            // Forward: activated = sigmoid(raw).squeeze(-1)
            // Backward: grad_raw = grad_activated.unsqueeze(-1) * sigmoid(raw) * (1 - sigmoid(raw))
            auto sigmoid_val = torch::sigmoid(opacities_raw);
            auto sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
            grad_opacities_raw = v_opacities.reshape(opacities.sizes()).unsqueeze(-1) * sigmoid_deriv;

            // Scaling: exp
            // Forward: activated = exp(raw)
            // Backward: grad_raw = grad_activated * exp(raw)
            grad_scales_raw = v_scales * scales;  // scales = exp(scales_raw)

            // Rotation: normalize
            // Forward: activated = raw / ||raw||
            // Backward: grad_raw = (grad_activated - (grad_activated · activated) * activated) / ||raw||
            auto norm = rotations_raw.norm(2, -1, true);
            auto dot_product = (v_quats * rotations).sum(-1, true);
            grad_rotations_raw = (v_quats - dot_product * rotations) / norm;

            // SH coeffs: no chain rule needed, already activated

            return {grad_means_raw, grad_rotations_raw, grad_scales_raw, grad_opacities_raw, v_sh_coeffs,
                    torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
                    torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
                    torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
                    torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
                    torch::Tensor(), torch::Tensor()};
        }
    };

    // Main render function (VALIDATION VERSION - WITH AUTOGRAD)
    RenderOutput rasterize_validation(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier,
        bool packed,
        bool antialiased,
        RenderMode render_mode,
        const gs::geometry::BoundingBox* bounding_box) {
        // Ensure we don't use packed mode (not supported in this implementation)
        TORCH_CHECK(!packed, "Packed mode is not supported in this implementation");

        // Get camera parameters
        const int image_height = static_cast<int>(viewpoint_camera.image_height());
        const int image_width = static_cast<int>(viewpoint_camera.image_width());

        // Prepare viewmat and K
        auto viewmat = viewpoint_camera.world_view_transform().to(torch::kCUDA);
        TORCH_CHECK(viewmat.dim() == 3 && viewmat.size(0) == 1 && viewmat.size(1) == 4 && viewmat.size(2) == 4,
                    "viewmat must be [1, 4, 4] after transpose and unsqueeze, got ", viewmat.sizes());
        TORCH_CHECK(viewmat.is_cuda(), "viewmat must be on CUDA");

        const auto K = viewpoint_camera.K().to(torch::kCUDA);
        TORCH_CHECK(K.is_cuda(), "K must be on CUDA");

        // Get RAW Gaussian parameters (no activations applied yet)
        auto means3D = gaussian_model.get_means();  // No activation
        auto opacities_raw = gaussian_model.opacity_raw();  // Raw (will be sigmoid + squeeze)
        auto scales_raw = gaussian_model.scaling_raw();  // Raw (will be exp)
        auto rotations_raw = gaussian_model.rotation_raw();  // Raw (will be normalized)
        auto sh_coeffs = gaussian_model.get_shs();  // Already activated (cat of sh0, shN)
        const int sh_degree = gaussian_model.get_active_sh_degree();

        // Validate RAW Gaussian parameters
        const int N = static_cast<int>(means3D.size(0));
        TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3,
                    "means3D must be [N, 3], got ", means3D.sizes());
        TORCH_CHECK(opacities_raw.dim() == 2 && opacities_raw.size(0) == N && opacities_raw.size(1) == 1,
                    "opacities_raw must be [N, 1], got ", opacities_raw.sizes());
        TORCH_CHECK(scales_raw.dim() == 2 && scales_raw.size(0) == N && scales_raw.size(1) == 3,
                    "scales_raw must be [N, 3], got ", scales_raw.sizes());
        TORCH_CHECK(rotations_raw.dim() == 2 && rotations_raw.size(0) == N && rotations_raw.size(1) == 4,
                    "rotations_raw must be [N, 4], got ", rotations_raw.sizes());
        TORCH_CHECK(sh_coeffs.dim() == 3 && sh_coeffs.size(0) == N && sh_coeffs.size(2) == 3,
                    "sh_coeffs must be [N, K, 3], got ", sh_coeffs.sizes());

        // Check if we have enough SH coefficients for the requested degree
        const int required_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
        TORCH_CHECK(sh_coeffs.size(1) >= required_sh_coeffs,
                    "Not enough SH coefficients. Expected at least ", required_sh_coeffs,
                    " but got ", sh_coeffs.size(1));

        // Device checks for Gaussian parameters
        TORCH_CHECK(means3D.is_cuda(), "means3D must be on CUDA");
        TORCH_CHECK(opacities_raw.is_cuda(), "opacities_raw must be on CUDA");
        TORCH_CHECK(scales_raw.is_cuda(), "scales_raw must be on CUDA");
        TORCH_CHECK(rotations_raw.is_cuda(), "rotations_raw must be on CUDA");
        TORCH_CHECK(sh_coeffs.is_cuda(), "sh_coeffs must be on CUDA");

        // Handle background color - can be undefined
        torch::Tensor prepared_bg_color;
        if (!bg_color.defined() || bg_color.numel() == 0) {
            // Keep it undefined
            prepared_bg_color = torch::Tensor();
        } else {
            prepared_bg_color = bg_color.view({1, -1}).to(torch::kCUDA);
            TORCH_CHECK(prepared_bg_color.size(0) == 1 && prepared_bg_color.size(1) == 3,
                        "bg_color must be reshapeable to [1, 3], got ", prepared_bg_color.sizes());
            TORCH_CHECK(prepared_bg_color.is_cuda(), "bg_color must be on CUDA");
        }

        const float eps2d = 0.3f;
        const float near_plane = 0.01f;
        const float far_plane = 10000.0f;
        const float radius_clip = 0.0f;
        const int tile_size = 16;
        const bool calc_compensations = antialiased;

        std::optional<torch::Tensor> radial_distortion;
        if (viewpoint_camera.radial_distortion().numel() > 0) {
            auto radial_distortion_val = viewpoint_camera.radial_distortion().to(torch::kCUDA);
            TORCH_CHECK(radial_distortion_val.dim() == 1, "radial_distortion must be 1D, got ", radial_distortion_val.sizes());
            if (radial_distortion_val.size(-1) < 4) {
                // Pad to 4 coefficients if less are provided
                radial_distortion_val = torch::nn::functional::pad(
                    radial_distortion_val,
                    torch::nn::functional::PadFuncOptions({0, 4 - radial_distortion_val.size(-1)}).mode(torch::kConstant).value(0));
            }
            radial_distortion = radial_distortion_val;
        }
        std::optional<torch::Tensor> tangential_distortion;
        if (viewpoint_camera.tangential_distortion().numel() > 0) {
            auto tangential_distortion_val = viewpoint_camera.tangential_distortion().to(torch::kCUDA);
            TORCH_CHECK(tangential_distortion_val.dim() == 1, "tangential_distortion must be 1D, got ", tangential_distortion_val.sizes());
            if (tangential_distortion_val.size(-1) < 2) {
                // Pad to 2 coefficients if less are provided
                tangential_distortion_val = torch::nn::functional::pad(
                    tangential_distortion_val,
                    torch::nn::functional::PadFuncOptions({0, 2 - tangential_distortion_val.size(-1)}).mode(torch::kConstant).value(0));
            }
            tangential_distortion = tangential_distortion_val;
        }

        // Call manual forward/backward (no PyTorch autograd)
        auto ut_params = UnscentedTransformParameters{};

        auto fused_outputs = GsplatRasterizeFunction::apply(
            means3D,
            rotations_raw,
            scales_raw,
            opacities_raw,
            sh_coeffs,
            sh_degree,
            prepared_bg_color.defined() ? prepared_bg_color : torch::tensor({0.0f, 0.0f, 0.0f}, means3D.options()).view({1, 3}),
            viewmat,
            K,
            image_width,
            image_height,
            tile_size,
            viewpoint_camera.camera_model_type(),
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            scaling_modifier,
            calc_compensations,
            static_cast<int>(render_mode),
            ut_params,
            radial_distortion,
            tangential_distortion
        );

        auto rendered_image = fused_outputs[0];  // [1, H, W, channels]
        auto rendered_alpha = fused_outputs[1];  // [1, H, W, 1]
        auto radii = fused_outputs[2];           // [C, N, 2]
        auto means2d_with_grad = fused_outputs[3]; // [C, N, 2]
        auto depths = fused_outputs[4];          // [C, N]

        means2d_with_grad = means2d_with_grad.contiguous();
        means2d_with_grad.set_requires_grad(true);
        means2d_with_grad.retain_grad();

        // Step 7: Post-process based on render mode
        torch::Tensor final_image, final_depth;

        switch (render_mode) {
        case RenderMode::RGB:
            final_image = rendered_image;
            final_depth = torch::Tensor(); // Empty
            break;

        case RenderMode::D:
            final_depth = rendered_image;  // It's actually depth
            final_image = torch::Tensor(); // Empty
            break;

        case RenderMode::ED:
            // Normalize accumulated depth by alpha to get expected depth
            final_depth = rendered_image / rendered_alpha.clamp_min(1e-10);
            final_image = torch::Tensor(); // Empty
            break;

        case RenderMode::RGB_D:
            final_image = rendered_image.index({Slice(), Slice(), Slice(), Slice(None, -1)});
            final_depth = rendered_image.index({Slice(), Slice(), Slice(), Slice(-1, None)});
            break;

        case RenderMode::RGB_ED:
            final_image = rendered_image.index({Slice(), Slice(), Slice(), Slice(None, -1)});
            auto accum_depth = rendered_image.index({Slice(), Slice(), Slice(), Slice(-1, None)});
            final_depth = accum_depth / rendered_alpha.clamp_min(1e-10);
            break;
        }

        // Prepare output
        RenderOutput result;

        // Handle image output
        if (final_image.defined() && final_image.numel() > 0) {
            // Register hook to log the gradient PyTorch computes for the output
            final_image.register_hook([](const torch::Tensor& grad) -> torch::Tensor {
                float grad_norm = grad.norm().item<float>();
                LOG_INFO("PyTorch hook on final_image BEFORE clamp/permute: grad shape = [{}, {}, {}, {}], norm = {:.6f}",
                         grad.size(0), grad.size(1), grad.size(2), grad.size(3), grad_norm);
                return grad;
            });

            result.image = torch::clamp(final_image.squeeze(0).permute({2, 0, 1}), 0.0f, 1.0f);

            // Also register hook on the final result to see what gradient the loss gives
            result.image.register_hook([](const torch::Tensor& grad) -> torch::Tensor {
                float grad_norm = grad.norm().item<float>();
                LOG_INFO("PyTorch hook on result.image AFTER clamp/permute: grad shape = [{}, {}, {}], norm = {:.6f}",
                         grad.size(0), grad.size(1), grad.size(2), grad_norm);
                return grad;
            });
        } else {
            result.image = torch::Tensor();
        }

        // Handle alpha output - always present
        result.alpha = rendered_alpha.squeeze(0).permute({2, 0, 1});

        // Handle depth output
        if (final_depth.defined() && final_depth.numel() > 0) {
            result.depth = final_depth.squeeze(0).permute({2, 0, 1});
        } else {
            result.depth = torch::Tensor();
        }

        result.means2d = means2d_with_grad;
        result.depths = depths.squeeze(0);
        result.radii = std::get<0>(radii.squeeze(0).max(-1));
        result.visibility = (result.radii > 0);
        result.width = image_width;
        result.height = image_height;

        // Final device checks for outputs
        if (result.image.defined() && result.image.numel() > 0) {
            TORCH_CHECK(result.image.is_cuda(), "result.image must be on CUDA");
        }
        TORCH_CHECK(result.alpha.is_cuda(), "result.alpha must be on CUDA");
        if (result.depth.defined() && result.depth.numel() > 0) {
            TORCH_CHECK(result.depth.is_cuda(), "result.depth must be on CUDA");
        }
        TORCH_CHECK(result.means2d.is_cuda(), "result.means2d must be on CUDA");
        TORCH_CHECK(result.depths.is_cuda(), "result.depths must be on CUDA");
        TORCH_CHECK(result.radii.is_cuda(), "result.radii must be on CUDA");
        TORCH_CHECK(result.visibility.is_cuda(), "result.visibility must be on CUDA");

        return result;
    }
} // namespace gs::training::validation
