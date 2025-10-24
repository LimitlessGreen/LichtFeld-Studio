/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/parameters.hpp"
#include "rasterization/rasterizer.hpp"
#include "loader/loader.hpp"
#include "Ops.h"  // For gsplat fwd/bwd functions

class GutManualVsAutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_points = 50;
        image_width = 128;
        image_height = 128;

        // Create initial synthetic parameters
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

        camera = std::make_shared<gs::Camera>(
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
    std::shared_ptr<gs::Camera> camera;
    torch::Tensor bg_color;
};

// Test that manual forward/backward gives same gradients as autograd
TEST_F(GutManualVsAutogradTest, ManualMatchesAutograd) {
    const float tolerance = 1e-4f;

    // Declare gradient tensors outside scopes so they persist
    torch::Tensor grad_means_autograd, grad_opacity_autograd, grad_scaling_autograd, grad_rotation_autograd, grad_sh0_autograd;
    torch::Tensor grad_means_manual, grad_opacity_manual, grad_scaling_manual, grad_rotation_manual, grad_sh0_manual;
    float loss_autograd_val, loss_manual_val;

    // ========== PATH 1: AUTOGRAD (current implementation via rasterizer.cpp) ==========
    {
        // Create fresh tensors with requires_grad for autograd path
        auto means_auto = initial_means.clone();
        auto opacity_auto = initial_opacity.clone();
        auto scaling_auto = initial_scaling.clone();
        auto rotation_auto = initial_rotation.clone();
        auto sh0_auto = initial_sh0.clone();
        auto shN_auto = initial_shN.clone();

        means_auto.set_requires_grad(true);
        opacity_auto.set_requires_grad(true);
        scaling_auto.set_requires_grad(true);
        rotation_auto.set_requires_grad(true);
        sh0_auto.set_requires_grad(true);
        shN_auto.set_requires_grad(true);

        // Create temporary SplatData for autograd path
        auto splat_auto = gs::SplatData(0, means_auto, sh0_auto, shN_auto,
                                         scaling_auto, rotation_auto, opacity_auto, 1.0f);

        // Use the actual rasterizer function which internally uses FusedRasterizationWithSHFunction
        auto output_autograd = gs::training::rasterize(
            *camera, splat_auto, bg_color,
            1.0f,  // scaling_modifier
            false, // packed
            false, // antialiased
            gs::training::RenderMode::RGB,
            nullptr  // RenderSettings
        );

        // Debug: Check output
        std::cout << "Autograd output image shape: " << output_autograd.image.sizes() << std::endl;
        std::cout << "Autograd output image min/max: " << output_autograd.image.min().item().toFloat()
                  << " / " << output_autograd.image.max().item().toFloat() << std::endl;

        // Compute loss
        auto loss_autograd = output_autograd.image.sum();
        loss_autograd_val = loss_autograd.item().toFloat();
        std::cout << "Autograd loss: " << loss_autograd_val << std::endl;
        loss_autograd.backward();

        // Save autograd gradients on RAW parameters
        grad_means_autograd = splat_auto.means().grad().clone();
        grad_opacity_autograd = splat_auto.opacity_raw().grad().clone();
        grad_scaling_autograd = splat_auto.scaling_raw().grad().clone();
        grad_rotation_autograd = splat_auto.rotation_raw().grad().clone();
        grad_sh0_autograd = splat_auto.sh0().grad().clone();
    }

    // ========== PATH 2: MANUAL (what we want to implement) ==========
    {
        // Create fresh tensors with requires_grad for manual path
        auto means_manual = initial_means.clone();
        auto opacity_manual = initial_opacity.clone();
        auto scaling_manual = initial_scaling.clone();
        auto rotation_manual = initial_rotation.clone();
        auto sh0_manual = initial_sh0.clone();
        auto shN_manual = initial_shN.clone();

        means_manual.set_requires_grad(true);
        opacity_manual.set_requires_grad(true);
        scaling_manual.set_requires_grad(true);
        rotation_manual.set_requires_grad(true);
        sh0_manual.set_requires_grad(true);
        shN_manual.set_requires_grad(true);

        // Step 1: Apply activation functions (with autograd enabled)
        torch::Tensor means3D, opacities, scales, rotations, sh_coeffs;
        {
            torch::AutoGradMode enable_grad(true);
            means3D = means_manual;  // No activation
            opacities = torch::sigmoid(opacity_manual).squeeze(-1);  // sigmoid + squeeze
            scales = torch::exp(scaling_manual);  // exp
            rotations = torch::nn::functional::normalize(rotation_manual,
                torch::nn::functional::NormalizeFuncOptions().dim(-1));  // normalize
            sh_coeffs = torch::cat({sh0_manual, shN_manual}, 1);  // cat(sh0, shN)
        }

        // Step 2: Prepare camera matrices
        auto viewmat = camera->world_view_transform().to(torch::kCUDA);
        auto K = camera->K().to(torch::kCUDA);

    // Step 3: Call gsplat forward manually
    auto fwd_results = gsplat::rasterize_from_world_with_sh_fwd(
        means3D.contiguous(),
        rotations.contiguous(),
        scales.contiguous(),
        opacities.contiguous(),
        sh_coeffs.contiguous(),
        0,  // sh_degree
        bg_color.view({1, 3}),
        std::nullopt,  // masks
        camera->image_width(),
        camera->image_height(),
        16,  // tile_size
        viewmat,
        std::nullopt,  // viewmats1
        K,
        camera->camera_model_type(),
        0.3f,  // eps2d
        0.01f,  // near_plane
        10000.0f,  // far_plane
        0.0f,  // radius_clip
        1.0f,  // scaling_modifier
        false,  // calc_compensations
        0,  // render_mode (RGB)
        UnscentedTransformParameters{},
        ShutterType::GLOBAL,
        std::nullopt,  // radial
        std::nullopt,  // tangential
        std::nullopt   // thin_prism
    );

    auto rendered_image = std::get<0>(fwd_results);
    auto rendered_alpha = std::get<1>(fwd_results);
    auto radii = std::get<2>(fwd_results);
    auto means2d = std::get<3>(fwd_results);
    auto depths = std::get<4>(fwd_results);
    auto colors = std::get<5>(fwd_results);
    auto tile_offsets = std::get<6>(fwd_results);
    auto flatten_ids = std::get<7>(fwd_results);
    auto last_ids = std::get<8>(fwd_results);
    auto compensations = std::get<9>(fwd_results);

    // Step 4: Post-process to match rasterizer.cpp output format
    auto final_image = torch::clamp(rendered_image.squeeze(0).permute({2, 0, 1}), 0.0f, 1.0f);

        // Step 5: Compute loss (same as autograd path)
        auto loss_manual = final_image.sum();
        loss_manual_val = loss_manual.item().toFloat();

        // Step 6: Compute gradient w.r.t. rendered image (reverse the permute/squeeze/clamp)
    auto grad_final_image = torch::ones_like(final_image);
    auto grad_rendered_image = grad_final_image.permute({1, 2, 0}).unsqueeze(0);

    // Apply clamp gradient (gradient is 0 outside [0, 1])
    auto mask = (rendered_image >= 0.0f) & (rendered_image <= 1.0f);
    grad_rendered_image = grad_rendered_image * mask.to(torch::kFloat32);

    auto grad_rendered_alpha = torch::zeros_like(rendered_alpha);

    // Step 7: Call gsplat backward manually
    auto bwd_grads = gsplat::rasterize_from_world_with_sh_bwd(
        means3D.contiguous(),
        rotations.contiguous(),
        scales.contiguous(),
        opacities.contiguous(),
        sh_coeffs.contiguous(),
        0,  // sh_degree
        bg_color.view({1, 3}),
        std::nullopt,  // masks
        camera->image_width(),
        camera->image_height(),
        16,  // tile_size
        viewmat,
        std::nullopt,  // viewmats1
        K,
        camera->camera_model_type(),
        0.3f,  // eps2d
        0.01f,  // near_plane
        10000.0f,  // far_plane
        0.0f,  // radius_clip
        1.0f,  // scaling_modifier
        false,  // calc_compensations
        0,  // render_mode (RGB)
        UnscentedTransformParameters{},
        ShutterType::GLOBAL,
        std::nullopt,  // radial
        std::nullopt,  // tangential
        std::nullopt,  // thin_prism
        // Saved forward outputs
        rendered_alpha,
        last_ids,
        tile_offsets,
        flatten_ids,
        colors,
        radii,
        means2d,
        depths,
        compensations,
        // Gradients
        grad_rendered_image,
        grad_rendered_alpha
    );

    auto v_means = std::get<0>(bwd_grads);
    auto v_quats = std::get<1>(bwd_grads);
    auto v_scales = std::get<2>(bwd_grads);
    auto v_opacities = std::get<3>(bwd_grads);
    auto v_sh_coeffs = std::get<4>(bwd_grads);

    // Step 8: Use PyTorch autograd to propagate through activation functions
    std::vector<torch::Tensor> tensors{means3D, rotations, scales, opacities, sh_coeffs};
    std::vector<torch::Tensor> grad_tensors{v_means, v_quats, v_scales,
                                             v_opacities.reshape(opacities.sizes()),
                                             v_sh_coeffs};

        torch::autograd::backward(tensors, grad_tensors);

        // Step 9: Get manual gradients on raw parameters
        grad_means_manual = means_manual.grad();
        grad_opacity_manual = opacity_manual.grad();
        grad_scaling_manual = scaling_manual.grad();
        grad_rotation_manual = rotation_manual.grad();
        grad_sh0_manual = sh0_manual.grad();
    }

    // ========== COMPARE GRADIENTS ==========
    ASSERT_TRUE(grad_means_autograd.defined() && grad_means_manual.defined());
    ASSERT_TRUE(grad_opacity_autograd.defined() && grad_opacity_manual.defined());
    ASSERT_TRUE(grad_scaling_autograd.defined() && grad_scaling_manual.defined());
    ASSERT_TRUE(grad_rotation_autograd.defined() && grad_rotation_manual.defined());
    ASSERT_TRUE(grad_sh0_autograd.defined() && grad_sh0_manual.defined());

    // Compare means gradients
    auto means_diff = (grad_means_autograd - grad_means_manual).abs().max().item().toFloat();
    auto means_scale = grad_means_autograd.abs().max().item().toFloat();
    EXPECT_LT(means_diff / (means_scale + 1e-8f), tolerance)
        << "Means gradient mismatch: max_diff=" << means_diff << ", scale=" << means_scale;

    // Compare opacity gradients
    auto opacity_diff = (grad_opacity_autograd - grad_opacity_manual).abs().max().item().toFloat();
    auto opacity_scale = grad_opacity_autograd.abs().max().item().toFloat();
    EXPECT_LT(opacity_diff / (opacity_scale + 1e-8f), tolerance)
        << "Opacity gradient mismatch: max_diff=" << opacity_diff << ", scale=" << opacity_scale;

    // Compare scaling gradients
    auto scaling_diff = (grad_scaling_autograd - grad_scaling_manual).abs().max().item().toFloat();
    auto scaling_scale = grad_scaling_autograd.abs().max().item().toFloat();
    EXPECT_LT(scaling_diff / (scaling_scale + 1e-8f), tolerance)
        << "Scaling gradient mismatch: max_diff=" << scaling_diff << ", scale=" << scaling_scale;

    // Compare rotation gradients
    auto rotation_diff = (grad_rotation_autograd - grad_rotation_manual).abs().max().item().toFloat();
    auto rotation_scale = grad_rotation_autograd.abs().max().item().toFloat();
    EXPECT_LT(rotation_diff / (rotation_scale + 1e-8f), tolerance)
        << "Rotation gradient mismatch: max_diff=" << rotation_diff << ", scale=" << rotation_scale;

    // Compare SH gradients
    auto sh0_diff = (grad_sh0_autograd - grad_sh0_manual).abs().max().item().toFloat();
    auto sh0_scale = grad_sh0_autograd.abs().max().item().toFloat();
    EXPECT_LT(sh0_diff / (sh0_scale + 1e-8f), tolerance)
        << "SH0 gradient mismatch: max_diff=" << sh0_diff << ", scale=" << sh0_scale;

    // Print summary
    std::cout << "\nGradient comparison summary:\n";
    std::cout << "  Means:    max_diff=" << means_diff << ", rel_err=" << (means_diff / (means_scale + 1e-8f)) << "\n";
    std::cout << "  Opacity:  max_diff=" << opacity_diff << ", rel_err=" << (opacity_diff / (opacity_scale + 1e-8f)) << "\n";
    std::cout << "  Scaling:  max_diff=" << scaling_diff << ", rel_err=" << (scaling_diff / (scaling_scale + 1e-8f)) << "\n";
    std::cout << "  Rotation: max_diff=" << rotation_diff << ", rel_err=" << (rotation_diff / (rotation_scale + 1e-8f)) << "\n";
    std::cout << "  SH0:      max_diff=" << sh0_diff << ", rel_err=" << (sh0_diff / (sh0_scale + 1e-8f)) << "\n";

    // Also verify loss values are close
    float loss_diff = std::abs(loss_autograd_val - loss_manual_val);
    EXPECT_LT(loss_diff, 1e-3f)
        << "Loss values differ: autograd=" << loss_autograd_val
        << ", manual=" << loss_manual_val;
}
