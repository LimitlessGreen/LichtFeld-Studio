/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "rasterization/rasterizer.hpp"

class GutRasterizerGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple scene with a few Gaussians
        n_points = 100;
        int sh_degree = 0;  // Use degree 0 (only DC component)

        // Initialize tensors
        auto means = torch::randn({n_points, 3}, torch::kFloat32).cuda() * 0.1f;
        auto sh0 = torch::rand({n_points, 1, 3}, torch::kFloat32).cuda();
        auto shN = torch::zeros({n_points, 0, 3}, torch::kFloat32).cuda();
        auto scaling = torch::ones({n_points, 3}, torch::kFloat32).cuda() * -3.0f;
        auto rotation = torch::zeros({n_points, 4}, torch::kFloat32).cuda();
        rotation.index_put_({torch::indexing::Slice(), 0}, 1.0f);
        auto opacity = torch::ones({n_points, 1}, torch::kFloat32).cuda() * 0.5f;

        // Set requires_grad after initialization
        means.set_requires_grad(true);
        sh0.set_requires_grad(true);
        shN.set_requires_grad(true);
        scaling.set_requires_grad(true);
        rotation.set_requires_grad(true);
        opacity.set_requires_grad(true);

        // Create SplatData
        splat_data = std::make_unique<gs::SplatData>(
            sh_degree, means, sh0, shN, scaling, rotation, opacity, 1.0f);

        // Create a simple camera
        image_width = 256;
        image_height = 256;
        float fx = 300.0f;
        float fy = 300.0f;
        float cx = 128.0f;
        float cy = 128.0f;

        // Camera rotation (identity) and translation
        auto R = torch::eye(3, torch::kFloat32).cuda();
        auto T = torch::tensor({0.0f, 0.0f, 2.0f}, torch::kFloat32).cuda();  // Move camera back
        auto radial_dist = torch::zeros({4}, torch::kFloat32).cuda();
        auto tangential_dist = torch::zeros({2}, torch::kFloat32).cuda();

        camera = std::make_unique<gs::Camera>(
            R, T, fx, fy, cx, cy,
            radial_dist, tangential_dist,
            gsplat::CameraModelType::PINHOLE,
            "test_camera", "",
            image_width, image_height,
            0  // uid
        );

        bg_color = torch::zeros({3}, torch::kFloat32).cuda();
    }

    int n_points;
    int image_width;
    int image_height;
    std::unique_ptr<gs::SplatData> splat_data;
    std::unique_ptr<gs::Camera> camera;
    torch::Tensor bg_color;
};

// Test that autograd gradients flow correctly through the rasterizer
TEST_F(GutRasterizerGradientTest, AutogradGradientsFlow) {
    // Get raw parameters
    auto means = splat_data->means();
    auto opacity_raw = splat_data->opacity_raw();
    auto scaling_raw = splat_data->scaling_raw();
    auto rotation_raw = splat_data->rotation_raw();
    auto sh0 = splat_data->sh0();
    auto shN = splat_data->shN();

    // Ensure they require grad
    ASSERT_TRUE(means.requires_grad());
    ASSERT_TRUE(opacity_raw.requires_grad());
    ASSERT_TRUE(scaling_raw.requires_grad());
    ASSERT_TRUE(rotation_raw.requires_grad());
    ASSERT_TRUE(sh0.requires_grad());
    ASSERT_TRUE(shN.requires_grad());

    // Zero gradients
    if (means.grad().defined()) means.mutable_grad().zero_();
    if (opacity_raw.grad().defined()) opacity_raw.mutable_grad().zero_();
    if (scaling_raw.grad().defined()) scaling_raw.mutable_grad().zero_();
    if (rotation_raw.grad().defined()) rotation_raw.mutable_grad().zero_();
    if (sh0.grad().defined()) sh0.mutable_grad().zero_();
    if (shN.grad().defined()) shN.mutable_grad().zero_();

    // Render
    auto output = gs::training::rasterize(
        *camera,
        *splat_data,
        bg_color,
        1.0f,  // scaling_modifier
        false, // packed
        false, // antialiased
        gs::training::RenderMode::RGB,
        nullptr
    );

    ASSERT_TRUE(output.image.defined());
    ASSERT_EQ(output.image.dim(), 3);
    ASSERT_EQ(output.image.size(0), 3);  // RGB channels
    ASSERT_EQ(output.image.size(1), image_height);
    ASSERT_EQ(output.image.size(2), image_width);

    // Create a dummy loss (sum of all pixels)
    auto loss = output.image.sum();

    // Backward pass
    loss.backward();

    // Check that gradients were computed for ALL parameters
    EXPECT_TRUE(means.grad().defined()) << "means should have gradients";
    EXPECT_TRUE(opacity_raw.grad().defined()) << "opacity_raw should have gradients";
    EXPECT_TRUE(scaling_raw.grad().defined()) << "scaling_raw should have gradients";
    EXPECT_TRUE(rotation_raw.grad().defined()) << "rotation_raw should have gradients";
    EXPECT_TRUE(sh0.grad().defined()) << "sh0 should have gradients";
    EXPECT_TRUE(shN.grad().defined()) << "shN should have gradients";

    // Check that gradients are non-zero (at least some of them)
    if (means.grad().defined()) {
        float means_grad_norm = means.grad().norm().item().toFloat();
        EXPECT_GT(means_grad_norm, 0.0f) << "means gradients should be non-zero";
    }

    if (opacity_raw.grad().defined()) {
        float opacity_grad_norm = opacity_raw.grad().norm().item().toFloat();
        EXPECT_GT(opacity_grad_norm, 0.0f) << "opacity gradients should be non-zero";
    }

    if (sh0.grad().defined()) {
        float sh0_grad_norm = sh0.grad().norm().item().toFloat();
        EXPECT_GT(sh0_grad_norm, 0.0f) << "sh0 gradients should be non-zero";
    }
}

// Test that gradients have reasonable magnitudes
TEST_F(GutRasterizerGradientTest, GradientMagnitudes) {
    // Get raw parameters
    auto means = splat_data->means();
    auto opacity_raw = splat_data->opacity_raw();
    auto sh0 = splat_data->sh0();

    // Zero gradients
    if (means.grad().defined()) means.mutable_grad().zero_();
    if (opacity_raw.grad().defined()) opacity_raw.mutable_grad().zero_();
    if (sh0.grad().defined()) sh0.mutable_grad().zero_();

    // Render
    auto output = gs::training::rasterize(
        *camera,
        *splat_data,
        bg_color,
        1.0f, false, false,
        gs::training::RenderMode::RGB,
        nullptr
    );

    // Create loss and backward
    auto loss = output.image.sum();
    loss.backward();

    // Check gradient magnitudes are reasonable (not nan, not inf, not zero)
    if (means.grad().defined()) {
        auto grad = means.grad();
        EXPECT_FALSE(torch::isnan(grad).any().item().toBool()) << "means gradients contain NaN";
        EXPECT_FALSE(torch::isinf(grad).any().item().toBool()) << "means gradients contain Inf";
        float grad_norm = grad.norm().item().toFloat();
        EXPECT_GT(grad_norm, 0.0f) << "means gradients should be non-zero";
        EXPECT_LT(grad_norm, 1e6f) << "means gradients should not be excessively large";
    }

    if (opacity_raw.grad().defined()) {
        auto grad = opacity_raw.grad();
        EXPECT_FALSE(torch::isnan(grad).any().item().toBool()) << "opacity gradients contain NaN";
        EXPECT_FALSE(torch::isinf(grad).any().item().toBool()) << "opacity gradients contain Inf";
        float grad_norm = grad.norm().item().toFloat();
        EXPECT_GT(grad_norm, 0.0f) << "opacity gradients should be non-zero";
        EXPECT_LT(grad_norm, 1e6f) << "opacity gradients should not be excessively large";
    }
}
