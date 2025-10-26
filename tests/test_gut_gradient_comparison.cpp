/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "rasterization/rasterizer.hpp"
#include "rasterization/validation.hpp"
#include <spdlog/spdlog.h>

using namespace gs::training;

class GradientComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);

        // Create a simple test scene
        N = 100;  // Small number for easier debugging
        width = 64;
        height = 64;

        // Create simple Gaussian data
        auto means = torch::randn({N, 3}, torch::kCUDA) * 0.5f;
        auto opacities_raw = torch::randn({N, 1}, torch::kCUDA) * 0.5f;
        auto scales_raw = torch::randn({N, 3}, torch::kCUDA) * 0.2f - 2.0f;  // Start small
        auto rotations_raw = torch::randn({N, 4}, torch::kCUDA);
        rotations_raw = torch::nn::functional::normalize(rotations_raw,
                                                         torch::nn::functional::NormalizeFuncOptions().dim(-1));

        // Create SH coefficients (degree 0 only for simplicity)
        auto sh0 = torch::randn({N, 1, 3}, torch::kCUDA) * 0.5f;
        auto shN = torch::zeros({N, 15, 3}, torch::kCUDA);

        // Enable gradients on raw parameters
        means.set_requires_grad(true);
        opacities_raw.set_requires_grad(true);
        scales_raw.set_requires_grad(true);
        rotations_raw.set_requires_grad(true);
        sh0.set_requires_grad(true);

        // Store parameters
        params_means = means;
        params_opacities_raw = opacities_raw;
        params_scales_raw = scales_raw;
        params_rotations_raw = rotations_raw;
        params_sh0 = sh0;
        params_shN = shN;

        // Create a simple camera
        auto R = torch::eye(3, torch::kFloat32).to(torch::kCUDA);
        auto T = torch::tensor({0.0f, 0.0f, 3.0f}, torch::kCUDA);
        float focal = 50.0f;
        float cx = width / 2.0f;
        float cy = height / 2.0f;

        camera = std::make_unique<gs::Camera>(
            R,  // R
            T,  // T
            focal, focal,  // focal_x, focal_y
            cx, cy,  // center_x, center_y
            torch::Tensor(),  // radial_distortion
            torch::Tensor(),  // tangential_distortion
            gsplat::CameraModelType::PINHOLE,  // camera_model_type
            "test",  // image_name
            "",  // image_path
            width, height,  // camera_width, camera_height
            0  // uid
        );

        bg_color = torch::ones({3}, torch::kCUDA);
    }

    void TearDown() override {
        camera.reset();
    }

    int N;
    int width;
    int height;
    torch::Tensor params_means;
    torch::Tensor params_opacities_raw;
    torch::Tensor params_scales_raw;
    torch::Tensor params_rotations_raw;
    torch::Tensor params_sh0;
    torch::Tensor params_shN;
    torch::Tensor bg_color;
    std::unique_ptr<gs::Camera> camera;
};

TEST_F(GradientComparisonTest, CompareGradientsOnSmallScene) {
    spdlog::info("=== Gradient Comparison Test ===");

    // Test 1: Autograd version (validation.cpp)
    spdlog::info("Running autograd version...");
    {
        // Clone parameters for autograd test
        auto means = params_means.clone().detach().requires_grad_(true);
        auto opacities_raw = params_opacities_raw.clone().detach().requires_grad_(true);
        auto scales_raw = params_scales_raw.clone().detach().requires_grad_(true);
        auto rotations_raw = params_rotations_raw.clone().detach().requires_grad_(true);
        auto sh0 = params_sh0.clone().detach().requires_grad_(true);
        auto shN = params_shN.clone();

        // Create SplatData
        gs::SplatData model(0, means, opacities_raw, scales_raw, rotations_raw, sh0, shN, 1.0f);

        // Forward pass
        auto bg = bg_color.clone();
        auto output = validation::rasterize_validation(*camera, model, bg, 1.0f, false, false,
                                                       validation::RenderMode::RGB, nullptr);

        // Compute simple loss (mean squared)
        auto loss = output.image.pow(2).mean();
        spdlog::info("Autograd loss: {}", loss.template item<float>());

        // Backward pass
        loss.backward();

        // Store gradients
        auto grad_means_autograd = means.grad().clone();
        auto grad_opacities_autograd = opacities_raw.grad().clone();
        auto grad_scales_autograd = scales_raw.grad().clone();
        auto grad_rotations_autograd = rotations_raw.grad().clone();
        auto grad_sh0_autograd = sh0.grad().clone();

        spdlog::info("Autograd gradients computed:");
        spdlog::info("  means grad norm: {}", grad_means_autograd.norm().template item<float>());
        spdlog::info("  opacities grad norm: {}", grad_opacities_autograd.norm().template item<float>());
        spdlog::info("  scales grad norm: {}", grad_scales_autograd.norm().template item<float>());
        spdlog::info("  rotations grad norm: {}", grad_rotations_autograd.norm().template item<float>());
        spdlog::info("  sh0 grad norm: {}", grad_sh0_autograd.norm().template item<float>());

        // Test 2: Manual gradient version (rasterizer.cpp)
        spdlog::info("\nRunning manual gradient version...");
        {
            // Clone parameters for manual test
            auto means2 = params_means.clone().detach().requires_grad_(true);
            auto opacities_raw2 = params_opacities_raw.clone().detach().requires_grad_(true);
            auto scales_raw2 = params_scales_raw.clone().detach().requires_grad_(true);
            auto rotations_raw2 = params_rotations_raw.clone().detach().requires_grad_(true);
            auto sh02 = params_sh0.clone().detach().requires_grad_(true);
            auto shN2 = params_shN.clone();

            // Create SplatData
            gs::SplatData model2(0, means2, opacities_raw2, scales_raw2, rotations_raw2, sh02, shN2, 1.0f);

            // Forward pass
            auto bg2 = bg_color.clone();
            auto [output2, ctx] = rasterize_forward(*camera, model2, bg2, 1.0f, false, false,
                                                    RenderMode::RGB, nullptr);

            // Compute same loss
            auto loss2 = output2.image.pow(2).mean();
            spdlog::info("Manual loss: {}", loss2.template item<float>());

            // Check forward pass matches
            float loss_diff = std::abs(loss.template item<float>() - loss2.template item<float>());
            spdlog::info("Loss difference: {}", loss_diff);
            EXPECT_LT(loss_diff, 1e-4) << "Forward passes don't match!";

            // Compute gradient of loss w.r.t. output image
            auto grad_output = 2.0f * output2.image / (output2.image.numel());

            // Manual backward pass
            rasterize_backward(ctx, grad_output, model2);

            // Get gradients
            auto grad_means_manual = means2.grad();
            auto grad_opacities_manual = opacities_raw2.grad();
            auto grad_scales_manual = scales_raw2.grad();
            auto grad_rotations_manual = rotations_raw2.grad();
            auto grad_sh0_manual = sh02.grad();

            spdlog::info("Manual gradients computed:");
            if (grad_means_manual.defined()) {
                spdlog::info("  means grad norm: {}", grad_means_manual.norm().template item<float>());
            } else {
                spdlog::error("  means grad is UNDEFINED!");
            }

            if (grad_opacities_manual.defined()) {
                spdlog::info("  opacities grad norm: {}", grad_opacities_manual.norm().template item<float>());
            } else {
                spdlog::error("  opacities grad is UNDEFINED!");
            }

            if (grad_scales_manual.defined()) {
                spdlog::info("  scales grad norm: {}", grad_scales_manual.norm().template item<float>());
            } else {
                spdlog::error("  scales grad is UNDEFINED!");
            }

            if (grad_rotations_manual.defined()) {
                spdlog::info("  rotations grad norm: {}", grad_rotations_manual.norm().template item<float>());
            } else {
                spdlog::error("  rotations grad is UNDEFINED!");
            }

            if (grad_sh0_manual.defined()) {
                spdlog::info("  sh0 grad norm: {}", grad_sh0_manual.norm().template item<float>());
            } else {
                spdlog::error("  sh0 grad is UNDEFINED!");
            }

            // Compare gradients
            if (grad_means_manual.defined()) {
                auto diff = (grad_means_autograd - grad_means_manual).abs();
                spdlog::info("\nMeans gradient comparison:");
                spdlog::info("  Max diff: {}", diff.max().template item<float>());
                spdlog::info("  Mean diff: {}", diff.mean().template item<float>());
                spdlog::info("  Relative error: {}", (diff / grad_means_autograd.abs().clamp_min(1e-8)).mean().template item<float>());
            }

            if (grad_opacities_manual.defined()) {
                auto diff = (grad_opacities_autograd - grad_opacities_manual).abs();
                spdlog::info("\nOpacities gradient comparison:");
                spdlog::info("  Max diff: {}", diff.max().template item<float>());
                spdlog::info("  Mean diff: {}", diff.mean().template item<float>());
                spdlog::info("  Relative error: {}", (diff / grad_opacities_autograd.abs().clamp_min(1e-8)).mean().template item<float>());
            }

            if (grad_scales_manual.defined()) {
                auto diff = (grad_scales_autograd - grad_scales_manual).abs();
                spdlog::info("\nScales gradient comparison:");
                spdlog::info("  Max diff: {}", diff.max().template item<float>());
                spdlog::info("  Mean diff: {}", diff.mean().template item<float>());
                spdlog::info("  Relative error: {}", (diff / grad_scales_autograd.abs().clamp_min(1e-8)).mean().template item<float>());
            }

            if (grad_rotations_manual.defined()) {
                auto diff = (grad_rotations_autograd - grad_rotations_manual).abs();
                spdlog::info("\nRotations gradient comparison:");
                spdlog::info("  Max diff: {}", diff.max().template item<float>());
                spdlog::info("  Mean diff: {}", diff.mean().template item<float>());
                spdlog::info("  Relative error: {}", (diff / grad_rotations_autograd.abs().clamp_min(1e-8)).mean().template item<float>());
            }

            if (grad_sh0_manual.defined()) {
                auto diff = (grad_sh0_autograd - grad_sh0_manual).abs();
                spdlog::info("\nSH0 gradient comparison:");
                spdlog::info("  Max diff: {}", diff.max().template item<float>());
                spdlog::info("  Mean diff: {}", diff.mean().template item<float>());
                spdlog::info("  Relative error: {}", (diff / grad_sh0_autograd.abs().clamp_min(1e-8)).mean().template item<float>());
            }

            // Assertions - allow some tolerance for numerical differences
            ASSERT_TRUE(grad_means_manual.defined()) << "Means gradient not defined!";
            ASSERT_TRUE(grad_opacities_manual.defined()) << "Opacities gradient not defined!";
            ASSERT_TRUE(grad_scales_manual.defined()) << "Scales gradient not defined!";
            ASSERT_TRUE(grad_rotations_manual.defined()) << "Rotations gradient not defined!";
            ASSERT_TRUE(grad_sh0_manual.defined()) << "SH0 gradient not defined!";

            // Check gradients are close (relative tolerance)
            auto means_close = torch::allclose(grad_means_autograd, grad_means_manual, 1e-3, 1e-4);
            EXPECT_TRUE(means_close) << "Means gradients differ significantly!";

            auto opacities_close = torch::allclose(grad_opacities_autograd, grad_opacities_manual, 1e-3, 1e-4);
            EXPECT_TRUE(opacities_close) << "Opacities gradients differ significantly!";

            auto scales_close = torch::allclose(grad_scales_autograd, grad_scales_manual, 1e-3, 1e-4);
            EXPECT_TRUE(scales_close) << "Scales gradients differ significantly!";

            auto rotations_close = torch::allclose(grad_rotations_autograd, grad_rotations_manual, 1e-3, 1e-4);
            EXPECT_TRUE(rotations_close) << "Rotations gradients differ significantly!";

            auto sh0_close = torch::allclose(grad_sh0_autograd, grad_sh0_manual, 1e-3, 1e-4);
            EXPECT_TRUE(sh0_close) << "SH0 gradients differ significantly!";
        }
    }

    spdlog::info("\n=== Test Complete ===");
}
