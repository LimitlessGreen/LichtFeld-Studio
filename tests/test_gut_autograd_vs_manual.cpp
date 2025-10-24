/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "rasterization/rasterizer.hpp"
#include "Ops.h"

// This test compares:
// 1. Current implementation (uses torch::autograd::backward through activations)
// 2. Manual chain rule implementation (what we need for libtorch-free)
//
// Both should produce IDENTICAL gradients on the raw Gaussian parameters

class AutogradVsManualTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_points = 50;
        image_width = 128;
        image_height = 128;

        // Create identical initial parameters
        initial_means = torch::randn({n_points, 3}, torch::kFloat32).cuda() * 0.1f;
        initial_sh0 = torch::rand({n_points, 1, 3}, torch::kFloat32).cuda();
        initial_shN = torch::zeros({n_points, 0, 3}, torch::kFloat32).cuda();
        initial_scaling = torch::ones({n_points, 3}, torch::kFloat32).cuda() * -3.0f;
        initial_rotation = torch::zeros({n_points, 4}, torch::kFloat32).cuda();
        initial_rotation.index_put_({torch::indexing::Slice(), 0}, 1.0f);
        initial_opacity = torch::ones({n_points, 1}, torch::kFloat32).cuda() * 0.5f;

        // Create camera
        auto R = torch::eye(3, torch::kFloat32).cuda();
        auto T = torch::tensor({0.0f, 0.0f, 2.0f}, torch::kFloat32).cuda();
        auto radial_dist = torch::zeros({4}, torch::kFloat32).cuda();
        auto tangential_dist = torch::zeros({2}, torch::kFloat32).cuda();

        camera = std::make_unique<gs::Camera>(
            R, T, 300.0f, 300.0f, 64.0f, 64.0f,
            radial_dist, tangential_dist,
            gsplat::CameraModelType::PINHOLE,
            "test", "", image_width, image_height, 0
        );

        bg_color = torch::zeros({3}, torch::kFloat32).cuda();
    }

    int n_points;
    int image_width;
    int image_height;
    torch::Tensor initial_means;
    torch::Tensor initial_sh0;
    torch::Tensor initial_shN;
    torch::Tensor initial_scaling;
    torch::Tensor initial_rotation;
    torch::Tensor initial_opacity;
    std::unique_ptr<gs::Camera> camera;
    torch::Tensor bg_color;

    // Helper to apply manual chain rule for activation functions
    void applyManualChainRule(
        const torch::Tensor& v_means,
        const torch::Tensor& v_opacities,
        const torch::Tensor& v_scales,
        const torch::Tensor& v_quats,
        const torch::Tensor& v_sh_coeffs,
        const torch::Tensor& means_raw,
        const torch::Tensor& opacity_raw,
        const torch::Tensor& scaling_raw,
        const torch::Tensor& rotation_raw,
        const torch::Tensor& sh0,
        const torch::Tensor& shN,
        torch::Tensor& grad_means_out,
        torch::Tensor& grad_opacity_out,
        torch::Tensor& grad_scaling_out,
        torch::Tensor& grad_rotation_out,
        torch::Tensor& grad_sh0_out)
    {
        // Means: no activation, gradient passes through
        grad_means_out = v_means;

        // Opacity: sigmoid
        // Forward: activated = sigmoid(raw).squeeze(-1)
        // Backward: grad_raw = grad_activated.unsqueeze(-1) * sigmoid(raw) * (1 - sigmoid(raw))
        auto sigmoid_val = torch::sigmoid(opacity_raw);
        auto sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
        grad_opacity_out = v_opacities.unsqueeze(-1) * sigmoid_deriv;

        // Scaling: exp
        // Forward: activated = exp(raw)
        // Backward: grad_raw = grad_activated * exp(raw)
        auto scaling_activated = torch::exp(scaling_raw);
        grad_scaling_out = v_scales * scaling_activated;

        // Rotation: normalize
        // Forward: activated = raw / ||raw||
        // Backward: grad_raw = (grad_activated - (grad_activated · activated) * activated) / ||raw||
        auto norm = rotation_raw.norm(2, -1, true);
        auto rotation_activated = rotation_raw / norm;
        auto dot_product = (v_quats * rotation_activated).sum(-1, true);
        grad_rotation_out = (v_quats - dot_product * rotation_activated) / norm;

        // SH: cat(sh0, shN)
        // Forward: activated = cat(sh0, shN)
        // Backward: just split the gradient
        int sh0_size = sh0.size(1);
        grad_sh0_out = v_sh_coeffs.narrow(1, 0, sh0_size);
    }
};

TEST_F(AutogradVsManualTest, CompareGradients) {
    const float tolerance = 1e-4f;

    torch::Tensor grad_means_auto, grad_opacity_auto, grad_scaling_auto, grad_rotation_auto, grad_sh0_auto;
    torch::Tensor grad_means_manual, grad_opacity_manual, grad_scaling_manual, grad_rotation_manual, grad_sh0_manual;

    // ========== PATH 1: AUTOGRAD (current implementation) ==========
    {
        auto means_auto = initial_means.clone();
        auto sh0_auto = initial_sh0.clone();
        auto shN_auto = initial_shN.clone();
        auto scaling_auto = initial_scaling.clone();
        auto rotation_auto = initial_rotation.clone();
        auto opacity_auto = initial_opacity.clone();

        means_auto.set_requires_grad(true);
        sh0_auto.set_requires_grad(true);
        shN_auto.set_requires_grad(true);
        scaling_auto.set_requires_grad(true);
        rotation_auto.set_requires_grad(true);
        opacity_auto.set_requires_grad(true);

        auto splat_auto = gs::SplatData(0, means_auto, sh0_auto, shN_auto,
                                         scaling_auto, rotation_auto, opacity_auto, 1.0f);

        // Use current rasterizer (which uses torch::autograd::backward)
        auto output_auto = gs::training::rasterize(
            *camera, splat_auto, bg_color, 1.0f, false, false,
            gs::training::RenderMode::RGB, nullptr
        );

        // Compute loss and backward
        auto loss_auto = output_auto.image.sum();
        loss_auto.backward();

        // Save autograd gradients
        grad_means_auto = means_auto.grad().clone();
        grad_opacity_auto = opacity_auto.grad().clone();
        grad_scaling_auto = scaling_auto.grad().clone();
        grad_rotation_auto = rotation_auto.grad().clone();
        grad_sh0_auto = sh0_auto.grad().clone();
    }

    // ========== PATH 2: MANUAL (what we need for libtorch-free) ==========
    {
        auto means_manual = initial_means.clone();
        auto sh0_manual = initial_sh0.clone();
        auto shN_manual = initial_shN.clone();
        auto scaling_manual = initial_scaling.clone();
        auto rotation_manual = initial_rotation.clone();
        auto opacity_manual = initial_opacity.clone();

        // Don't need requires_grad for manual path
        means_manual.set_requires_grad(false);
        sh0_manual.set_requires_grad(false);
        shN_manual.set_requires_grad(false);
        scaling_manual.set_requires_grad(false);
        rotation_manual.set_requires_grad(false);
        opacity_manual.set_requires_grad(false);

        // Step 1: Apply activation functions (no autograd)
        torch::Tensor means3D, opacities, scales, rotations, sh_coeffs;
        {
            torch::NoGradGuard no_grad;
            means3D = means_manual;
            opacities = torch::sigmoid(opacity_manual).squeeze(-1);
            scales = torch::exp(scaling_manual);
            rotations = torch::nn::functional::normalize(rotation_manual,
                torch::nn::functional::NormalizeFuncOptions().dim(-1));
            sh_coeffs = torch::cat({sh0_manual, shN_manual}, 1);
        }

        // Step 2: Prepare camera matrices
        auto viewmat = camera->world_view_transform().to(torch::kCUDA);
        auto K = camera->K().to(torch::kCUDA);

        // Step 3: Call gsplat forward manually
        auto results = gsplat::rasterize_from_world_with_sh_fwd(
            means3D.contiguous(),
            rotations.contiguous(),
            scales.contiguous(),
            opacities.contiguous(),
            sh_coeffs.contiguous(),
            0,  // sh_degree
            bg_color.view({1, 3}),
            std::nullopt,
            image_width, image_height, 16,
            viewmat, std::nullopt, K,
            gsplat::CameraModelType::PINHOLE,
            0.3f, 0.01f, 10000.0f, 0.0f, 1.0f, false, 0,
            UnscentedTransformParameters{},
            ShutterType::GLOBAL,
            std::nullopt, std::nullopt, std::nullopt
        );

        auto rendered_image = std::get<0>(results);
        auto rendered_alpha = std::get<1>(results);
        auto radii = std::get<2>(results);
        auto means2d = std::get<3>(results);
        auto depths = std::get<4>(results);
        auto colors = std::get<5>(results);
        auto tile_offsets = std::get<6>(results);
        auto flatten_ids = std::get<7>(results);
        auto last_ids = std::get<8>(results);
        auto compensations = std::get<9>(results);

        // Step 4: Post-process to match rasterizer output format
        auto final_image = torch::clamp(rendered_image.squeeze(0).permute({2, 0, 1}), 0.0f, 1.0f);

        // Step 5: Compute gradient w.r.t. rendered image (reverse operations)
        auto grad_final_image = torch::ones_like(final_image);
        auto grad_rendered_image = grad_final_image.permute({1, 2, 0}).unsqueeze(0);

        // Apply clamp gradient
        auto mask = (rendered_image >= 0.0f) & (rendered_image <= 1.0f);
        grad_rendered_image = grad_rendered_image * mask.to(torch::kFloat32);

        auto grad_rendered_alpha = torch::zeros_like(rendered_alpha);

        // Step 6: Call gsplat backward manually
        auto grads = gsplat::rasterize_from_world_with_sh_bwd(
            means3D.contiguous(),
            rotations.contiguous(),
            scales.contiguous(),
            opacities.contiguous(),
            sh_coeffs.contiguous(),
            0,  // sh_degree
            bg_color.view({1, 3}),
            std::nullopt,
            image_width, image_height, 16,
            viewmat, std::nullopt, K,
            gsplat::CameraModelType::PINHOLE,
            0.3f, 0.01f, 10000.0f, 0.0f, 1.0f, false, 0,
            UnscentedTransformParameters{},
            ShutterType::GLOBAL,
            std::nullopt, std::nullopt, std::nullopt,
            // Saved forward outputs
            rendered_alpha, last_ids, tile_offsets, flatten_ids,
            colors, radii, means2d, depths, compensations,
            // Gradients
            grad_rendered_image,
            grad_rendered_alpha
        );

        auto v_means = std::get<0>(grads);
        auto v_quats = std::get<1>(grads);
        auto v_scales = std::get<2>(grads);
        auto v_opacities = std::get<3>(grads);
        auto v_sh_coeffs = std::get<4>(grads);

        // Step 7: Apply manual chain rule to get gradients on raw parameters
        applyManualChainRule(
            v_means, v_opacities.reshape(opacities.sizes()), v_scales, v_quats, v_sh_coeffs,
            means_manual, opacity_manual, scaling_manual, rotation_manual, sh0_manual, shN_manual,
            grad_means_manual, grad_opacity_manual, grad_scaling_manual, grad_rotation_manual, grad_sh0_manual
        );
    }

    // ========== COMPARE GRADIENTS ==========
    ASSERT_TRUE(grad_means_auto.defined() && grad_means_manual.defined());
    ASSERT_TRUE(grad_opacity_auto.defined() && grad_opacity_manual.defined());
    ASSERT_TRUE(grad_scaling_auto.defined() && grad_scaling_manual.defined());
    ASSERT_TRUE(grad_rotation_auto.defined() && grad_rotation_manual.defined());
    ASSERT_TRUE(grad_sh0_auto.defined() && grad_sh0_manual.defined());

    // Compare means gradients
    auto means_diff = (grad_means_auto - grad_means_manual).abs().max().item().toFloat();
    auto means_scale = grad_means_auto.abs().max().item().toFloat();
    EXPECT_LT(means_diff / (means_scale + 1e-8f), tolerance)
        << "Means gradient mismatch: max_diff=" << means_diff << ", scale=" << means_scale;

    // Compare opacity gradients
    auto opacity_diff = (grad_opacity_auto - grad_opacity_manual).abs().max().item().toFloat();
    auto opacity_scale = grad_opacity_auto.abs().max().item().toFloat();
    EXPECT_LT(opacity_diff / (opacity_scale + 1e-8f), tolerance)
        << "Opacity gradient mismatch: max_diff=" << opacity_diff << ", scale=" << opacity_scale;

    // Compare scaling gradients
    auto scaling_diff = (grad_scaling_auto - grad_scaling_manual).abs().max().item().toFloat();
    auto scaling_scale = grad_scaling_auto.abs().max().item().toFloat();
    EXPECT_LT(scaling_diff / (scaling_scale + 1e-8f), tolerance)
        << "Scaling gradient mismatch: max_diff=" << scaling_diff << ", scale=" << scaling_scale;

    // Compare rotation gradients
    auto rotation_diff = (grad_rotation_auto - grad_rotation_manual).abs().max().item().toFloat();
    auto rotation_scale = grad_rotation_auto.abs().max().item().toFloat();
    EXPECT_LT(rotation_diff / (rotation_scale + 1e-8f), tolerance)
        << "Rotation gradient mismatch: max_diff=" << rotation_diff << ", scale=" << rotation_scale;

    // Compare SH gradients
    auto sh0_diff = (grad_sh0_auto - grad_sh0_manual).abs().max().item().toFloat();
    auto sh0_scale = grad_sh0_auto.abs().max().item().toFloat();
    EXPECT_LT(sh0_diff / (sh0_scale + 1e-8f), tolerance)
        << "SH0 gradient mismatch: max_diff=" << sh0_diff << ", scale=" << sh0_scale;

    // Print summary
    std::cout << "\nGradient comparison (Autograd vs Manual Chain Rule):\n";
    std::cout << "  Means:    max_diff=" << means_diff << ", rel_err=" << (means_diff / (means_scale + 1e-8f)) << "\n";
    std::cout << "  Opacity:  max_diff=" << opacity_diff << ", rel_err=" << (opacity_diff / (opacity_scale + 1e-8f)) << "\n";
    std::cout << "  Scaling:  max_diff=" << scaling_diff << ", rel_err=" << (scaling_diff / (scaling_scale + 1e-8f)) << "\n";
    std::cout << "  Rotation: max_diff=" << rotation_diff << ", rel_err=" << (rotation_diff / (rotation_scale + 1e-8f)) << "\n";
    std::cout << "  SH0:      max_diff=" << sh0_diff << ", rel_err=" << (sh0_diff / (sh0_scale + 1e-8f)) << "\n";
}
