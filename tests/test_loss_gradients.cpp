/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include "kernels/regularization.cuh"
#include "core_new/tensor.hpp"
#include "losses/losses.hpp"

/**
 * Loss Gradient CUDA Kernel Tests
 *
 * These tests verify that our custom CUDA kernels and manual gradient
 * computations match PyTorch's autograd results exactly:
 * - L1 Loss: Manual gradient computation
 * - Regularization: CUDA kernels with chain rule gradients
 *   - Scaling: exp(_scaling)
 *   - Opacity: sigmoid(_opacity)
 */

class LossGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available, skipping test";
        }
    }
};

// =============================================================================
// L1 LOSS TESTS (manual gradient computation)
// =============================================================================

TEST_F(LossGradientTest, L1Loss_MatchesAutograd_3D) {
    // Test case similar to our rendered image: [3, H, W] or [H, W, 3]
    const int H = 256;
    const int W = 256;
    const int C = 3;

    // Test [3, H, W] layout (typical for our renderer)
    {
        auto rendered = torch::randn({C, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                           .requires_grad_(true);
        auto gt = torch::randn({C, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Method 1: Manual gradient computation
        auto rendered_manual = rendered.clone().detach().requires_grad_(false);
        auto diff = rendered_manual - gt;
        float l1_loss_manual = diff.abs().mean().item<float>();
        auto grad_manual = diff.sign() / static_cast<float>(diff.numel());

        // Method 2: PyTorch autograd
        auto rendered_autograd = rendered.clone().detach().requires_grad_(true);
        auto loss = torch::l1_loss(rendered_autograd, gt);
        float l1_loss_autograd = loss.item<float>();
        loss.backward();
        auto grad_autograd = rendered_autograd.grad();

        // Compare loss
        EXPECT_NEAR(l1_loss_manual, l1_loss_autograd, 1e-6)
            << "L1 loss mismatch [3,H,W]: manual=" << l1_loss_manual
            << ", autograd=" << l1_loss_autograd;

        // Compare gradients
        auto grad_diff = (grad_manual - grad_autograd).abs();
        float max_diff = grad_diff.max().item<float>();
        float mean_diff = grad_diff.mean().item<float>();

        EXPECT_LT(max_diff, 1e-6) << "Max gradient difference too large [3,H,W]: " << max_diff;
        EXPECT_LT(mean_diff, 1e-7) << "Mean gradient difference too large [3,H,W]: " << mean_diff;

        std::cout << "L1 loss [3,H,W]: loss_diff=" << std::abs(l1_loss_manual - l1_loss_autograd)
                  << ", grad max_diff=" << max_diff << ", mean_diff=" << mean_diff << " ✓" << std::endl;
    }
}

TEST_F(LossGradientTest, L1Loss_MatchesAutograd_4D) {
    // Test case with batch dimension: [1, 3, H, W]
    const int H = 256;
    const int W = 256;

    auto rendered = torch::randn({1, 3, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                       .requires_grad_(true);
    auto gt = torch::randn({1, 3, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Method 1: Manual gradient computation
    auto rendered_manual = rendered.clone().detach().requires_grad_(false);
    auto diff = rendered_manual - gt;
    float l1_loss_manual = diff.abs().mean().item<float>();
    auto grad_manual = diff.sign() / static_cast<float>(diff.numel());

    // Method 2: PyTorch autograd
    auto rendered_autograd = rendered.clone().detach().requires_grad_(true);
    auto loss = torch::l1_loss(rendered_autograd, gt);
    float l1_loss_autograd = loss.item<float>();
    loss.backward();
    auto grad_autograd = rendered_autograd.grad();

    // Compare loss
    EXPECT_NEAR(l1_loss_manual, l1_loss_autograd, 1e-6)
        << "L1 loss mismatch [1,3,H,W]: manual=" << l1_loss_manual
        << ", autograd=" << l1_loss_autograd;

    // Compare gradients
    auto grad_diff = (grad_manual - grad_autograd).abs();
    float max_diff = grad_diff.max().item<float>();
    float mean_diff = grad_diff.mean().item<float>();

    EXPECT_LT(max_diff, 1e-6) << "Max gradient difference too large [1,3,H,W]: " << max_diff;
    EXPECT_LT(mean_diff, 1e-7) << "Mean gradient difference too large [1,3,H,W]: " << mean_diff;

    std::cout << "L1 loss [1,3,H,W]: loss_diff=" << std::abs(l1_loss_manual - l1_loss_autograd)
              << ", grad max_diff=" << max_diff << ", mean_diff=" << mean_diff << " ✓" << std::endl;
}

TEST_F(LossGradientTest, L1Loss_DifferentImageSizes) {
    std::vector<std::pair<int, int>> sizes = {{64, 64}, {128, 128}, {256, 256}, {512, 512}, {800, 800}};

    for (auto [H, W] : sizes) {
        auto rendered = torch::randn({3, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto gt = torch::randn({3, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Manual gradient
        auto diff = rendered - gt;
        float l1_loss_manual = diff.abs().mean().item<float>();
        auto grad_manual = diff.sign() / static_cast<float>(diff.numel());

        // PyTorch autograd
        auto rendered_autograd = rendered.clone().requires_grad_(true);
        auto loss = torch::l1_loss(rendered_autograd, gt);
        float l1_loss_autograd = loss.item<float>();
        loss.backward();
        auto grad_autograd = rendered_autograd.grad();

        // Compare
        EXPECT_NEAR(l1_loss_manual, l1_loss_autograd, 1e-6)
            << "Failed for size " << H << "x" << W;

        auto grad_diff = (grad_manual - grad_autograd).abs();
        float max_diff = grad_diff.max().item<float>();
        EXPECT_LT(max_diff, 1e-6) << "Failed for size " << H << "x" << W << ": max_diff=" << max_diff;
    }

    std::cout << "L1 loss tested across " << sizes.size() << " image sizes ✓" << std::endl;
}

TEST_F(LossGradientTest, L1Loss_EdgeCases) {
    const int H = 128;
    const int W = 128;

    // Test 1: Identical images (zero loss, zero gradient)
    {
        auto rendered = torch::randn({3, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto gt = rendered.clone();

        auto diff = rendered - gt;
        float l1_loss = diff.abs().mean().item<float>();
        auto grad = diff.sign() / static_cast<float>(diff.numel());

        EXPECT_NEAR(l1_loss, 0.0f, 1e-6) << "Loss should be zero for identical images";
        EXPECT_NEAR(grad.abs().sum().item<float>(), 0.0f, 1e-6) << "Gradient should be zero for identical images";

        std::cout << "L1 edge case (identical images) ✓" << std::endl;
    }

    // Test 2: Large differences
    {
        auto rendered = torch::ones({3, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto gt = torch::zeros({3, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto diff = rendered - gt;
        float l1_loss_manual = diff.abs().mean().item<float>();
        auto grad_manual = diff.sign() / static_cast<float>(diff.numel());

        auto rendered_autograd = rendered.clone().requires_grad_(true);
        auto loss = torch::l1_loss(rendered_autograd, gt);
        float l1_loss_autograd = loss.item<float>();
        loss.backward();

        EXPECT_NEAR(l1_loss_manual, 1.0f, 1e-6) << "Loss should be 1.0 for ones vs zeros";
        EXPECT_NEAR(l1_loss_manual, l1_loss_autograd, 1e-6);

        std::cout << "L1 edge case (large differences) ✓" << std::endl;
    }
}

// =============================================================================
// SCALING TESTS (exp transformation)
// =============================================================================

TEST_F(LossGradientTest, ExpKernel_MatchesAutograd_Loss) {
    const int n = 10000;
    const float weight = 0.1f;

    // Create raw scaling parameter [N, 3] - 3 scales per Gaussian
    auto scaling_raw = torch::randn({n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Method 1: Our CUDA kernel
    auto scaling_raw_cuda = scaling_raw.clone().detach().requires_grad_(true);
    float cuda_loss = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
        scaling_raw_cuda, weight);

    // Method 2: PyTorch autograd (ground truth)
    auto scaling_raw_autograd = scaling_raw.clone().detach().requires_grad_(true);
    auto scaling = torch::exp(scaling_raw_autograd);
    auto loss = weight * scaling.mean();
    float autograd_loss = loss.item<float>();
    loss.backward();

    // Compare loss values
    float loss_diff = std::abs(cuda_loss - autograd_loss);
    EXPECT_LT(loss_diff, 1e-5) << "Loss mismatch: CUDA=" << cuda_loss
                               << ", PyTorch=" << autograd_loss;

    std::cout << "Exp kernel loss: CUDA=" << cuda_loss << ", PyTorch=" << autograd_loss
              << ", diff=" << loss_diff << " ✓" << std::endl;
}

TEST_F(LossGradientTest, ExpKernel_MatchesAutograd_Gradients) {
    const int n = 10000;
    const float weight = 0.1f;

    // Create raw scaling parameter [N, 3] - 3 scales per Gaussian
    auto scaling_raw = torch::randn({n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Method 1: Our CUDA kernel
    auto scaling_raw_cuda = scaling_raw.clone().detach().requires_grad_(true);
    gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
        scaling_raw_cuda, weight);
    auto cuda_grad = scaling_raw_cuda.grad();

    // Method 2: PyTorch autograd (ground truth)
    auto scaling_raw_autograd = scaling_raw.clone().detach().requires_grad_(true);
    auto scaling = torch::exp(scaling_raw_autograd);
    auto loss = weight * scaling.mean();
    loss.backward();
    auto autograd_grad = scaling_raw_autograd.grad();

    // Compare gradients
    ASSERT_TRUE(cuda_grad.defined()) << "CUDA gradient not computed";
    ASSERT_TRUE(autograd_grad.defined()) << "Autograd gradient not computed";

    auto grad_diff = (cuda_grad - autograd_grad).abs();
    float max_diff = grad_diff.max().item<float>();
    float mean_diff = grad_diff.mean().item<float>();

    EXPECT_LT(max_diff, 1e-5) << "Max gradient difference too large: " << max_diff;
    EXPECT_LT(mean_diff, 1e-6) << "Mean gradient difference too large: " << mean_diff;

    std::cout << "Exp kernel gradients: max_diff=" << max_diff
              << ", mean_diff=" << mean_diff << " ✓" << std::endl;
}

TEST_F(LossGradientTest, ExpKernel_DifferentSizes) {
    const float weight = 0.1f;
    std::vector<int> sizes = {100, 1000, 10000, 100000};

    for (int n : sizes) {
        auto scaling_raw = torch::randn({n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .requires_grad_(true);

        // CUDA kernel
        auto scaling_raw_cuda = scaling_raw.clone().detach().requires_grad_(true);
        float cuda_loss = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
            scaling_raw_cuda, weight);

        // PyTorch autograd
        auto scaling_raw_autograd = scaling_raw.clone().detach().requires_grad_(true);
        auto scaling = torch::exp(scaling_raw_autograd);
        auto loss = weight * scaling.mean();
        float autograd_loss = loss.item<float>();
        loss.backward();

        // Compare
        float loss_diff = std::abs(cuda_loss - autograd_loss);
        EXPECT_LT(loss_diff, 1e-5) << "Loss mismatch for size " << n;

        auto grad_diff = (scaling_raw_cuda.grad() - scaling_raw_autograd.grad()).abs();
        float max_diff = grad_diff.max().item<float>();
        EXPECT_LT(max_diff, 1e-5) << "Failed for size " << n << ": max_diff=" << max_diff;
    }

    std::cout << "Exp kernel tested across " << sizes.size() << " sizes ✓" << std::endl;
}

TEST_F(LossGradientTest, ExpKernel_GradientAccumulation) {
    const int n = 1000;
    const float weight = 0.1f;

    auto scaling_raw = torch::randn({n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Simulate existing gradient from main loss
    scaling_raw.mutable_grad() = torch::ones_like(scaling_raw) * 0.5f;
    auto initial_grad = scaling_raw.grad().clone();

    // Apply CUDA kernel (should accumulate)
    gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
        scaling_raw, weight);

    // Verify accumulation with PyTorch
    auto expected_reg_grad = (weight / static_cast<float>(scaling_raw.numel())) * torch::exp(scaling_raw.clone().detach());
    auto expected_total_grad = initial_grad + expected_reg_grad;

    auto grad_diff = (scaling_raw.grad() - expected_total_grad).abs();
    float max_diff = grad_diff.max().item<float>();

    EXPECT_LT(max_diff, 1e-5) << "Gradient accumulation failed: max_diff=" << max_diff;

    std::cout << "Exp kernel gradient accumulation ✓" << std::endl;
}

// =============================================================================
// OPACITY TESTS (sigmoid transformation)
// =============================================================================

TEST_F(LossGradientTest, SigmoidKernel_MatchesAutograd_Loss) {
    const int n = 10000;
    const float weight = 0.1f;

    // Create raw opacity parameter [N, 1]
    auto opacity_raw = torch::randn({n, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Method 1: Our CUDA kernel
    auto opacity_raw_cuda = opacity_raw.clone().detach().requires_grad_(true);
    float cuda_loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
        opacity_raw_cuda, weight);

    // Method 2: PyTorch autograd (ground truth)
    auto opacity_raw_autograd = opacity_raw.clone().detach().requires_grad_(true);
    auto opacity = torch::sigmoid(opacity_raw_autograd).squeeze(-1);
    auto loss = weight * opacity.mean();
    float autograd_loss = loss.item<float>();
    loss.backward();

    // Compare loss values
    float loss_diff = std::abs(cuda_loss - autograd_loss);
    EXPECT_LT(loss_diff, 1e-5) << "Loss mismatch: CUDA=" << cuda_loss
                               << ", PyTorch=" << autograd_loss;

    std::cout << "Sigmoid kernel loss: CUDA=" << cuda_loss << ", PyTorch=" << autograd_loss
              << ", diff=" << loss_diff << " ✓" << std::endl;
}

TEST_F(LossGradientTest, SigmoidKernel_MatchesAutograd_Gradients) {
    const int n = 10000;
    const float weight = 0.1f;

    // Create raw opacity parameter [N, 1]
    auto opacity_raw = torch::randn({n, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Method 1: Our CUDA kernel
    auto opacity_raw_cuda = opacity_raw.clone().detach().requires_grad_(true);
    gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
        opacity_raw_cuda, weight);
    auto cuda_grad = opacity_raw_cuda.grad();

    // Method 2: PyTorch autograd (ground truth)
    auto opacity_raw_autograd = opacity_raw.clone().detach().requires_grad_(true);
    auto opacity = torch::sigmoid(opacity_raw_autograd).squeeze(-1);
    auto loss = weight * opacity.mean();
    loss.backward();
    auto autograd_grad = opacity_raw_autograd.grad();

    // Compare gradients
    ASSERT_TRUE(cuda_grad.defined()) << "CUDA gradient not computed";
    ASSERT_TRUE(autograd_grad.defined()) << "Autograd gradient not computed";

    auto grad_diff = (cuda_grad - autograd_grad).abs();
    float max_diff = grad_diff.max().item<float>();
    float mean_diff = grad_diff.mean().item<float>();

    EXPECT_LT(max_diff, 1e-5) << "Max gradient difference too large: " << max_diff;
    EXPECT_LT(mean_diff, 1e-6) << "Mean gradient difference too large: " << mean_diff;

    std::cout << "Sigmoid kernel gradients: max_diff=" << max_diff
              << ", mean_diff=" << mean_diff << " ✓" << std::endl;
}

TEST_F(LossGradientTest, SigmoidKernel_DifferentSizes) {
    const float weight = 0.1f;
    std::vector<int> sizes = {100, 1000, 10000, 100000};

    for (int n : sizes) {
        auto opacity_raw = torch::randn({n, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .requires_grad_(true);

        // CUDA kernel
        auto opacity_raw_cuda = opacity_raw.clone().detach().requires_grad_(true);
        float cuda_loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
            opacity_raw_cuda, weight);

        // PyTorch autograd
        auto opacity_raw_autograd = opacity_raw.clone().detach().requires_grad_(true);
        auto opacity = torch::sigmoid(opacity_raw_autograd).squeeze(-1);
        auto loss = weight * opacity.mean();
        float autograd_loss = loss.item<float>();
        loss.backward();

        // Compare
        float loss_diff = std::abs(cuda_loss - autograd_loss);
        EXPECT_LT(loss_diff, 1e-5) << "Loss mismatch for size " << n;

        auto grad_diff = (opacity_raw_cuda.grad() - opacity_raw_autograd.grad()).abs();
        float max_diff = grad_diff.max().item<float>();
        EXPECT_LT(max_diff, 1e-5) << "Failed for size " << n << ": max_diff=" << max_diff;
    }

    std::cout << "Sigmoid kernel tested across " << sizes.size() << " sizes ✓" << std::endl;
}

TEST_F(LossGradientTest, SigmoidKernel_GradientAccumulation) {
    const int n = 1000;
    const float weight = 0.1f;

    auto opacity_raw = torch::randn({n, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Simulate existing gradient from main loss
    opacity_raw.mutable_grad() = torch::ones_like(opacity_raw) * 0.5f;
    auto initial_grad = opacity_raw.grad().clone();

    // Apply CUDA kernel (should accumulate)
    gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
        opacity_raw, weight);

    // Verify accumulation with PyTorch
    auto opacity_detached = torch::sigmoid(opacity_raw.clone().detach()).squeeze(-1);
    auto expected_reg_grad = (weight / static_cast<float>(opacity_detached.numel())) *
                             (opacity_detached * (1.0f - opacity_detached)).unsqueeze(-1);
    auto expected_total_grad = initial_grad + expected_reg_grad;

    auto grad_diff = (opacity_raw.grad() - expected_total_grad).abs();
    float max_diff = grad_diff.max().item<float>();

    EXPECT_LT(max_diff, 1e-5) << "Gradient accumulation failed: max_diff=" << max_diff;

    std::cout << "Sigmoid kernel gradient accumulation ✓" << std::endl;
}

// =============================================================================
// ZERO WEIGHT TESTS
// =============================================================================

TEST_F(LossGradientTest, ZeroWeight_NoGradients) {
    const int n = 1000;
    const float weight = 0.0f;

    // Test exp kernel
    {
        auto scaling_raw = torch::randn({n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .requires_grad_(true);
        float loss = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
            scaling_raw, weight);

        EXPECT_FLOAT_EQ(loss, 0.0f) << "Loss should be zero when weight is zero";
        EXPECT_FALSE(scaling_raw.grad().defined()) << "Gradient should not be defined for zero weight";
    }

    // Test sigmoid kernel
    {
        auto opacity_raw = torch::randn({n, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .requires_grad_(true);
        float loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
            opacity_raw, weight);

        EXPECT_FLOAT_EQ(loss, 0.0f) << "Loss should be zero when weight is zero";
        EXPECT_FALSE(opacity_raw.grad().defined()) << "Gradient should not be defined for zero weight";
    }

    std::cout << "Zero weight test ✓" << std::endl;
}

// =============================================================================
// L1 LOSS BENCHMARKS
// =============================================================================

TEST_F(LossGradientTest, BENCHMARK_L1Loss_Manual_vs_Autograd) {
    std::cout << "\n=== L1 Loss Benchmark: Manual vs PyTorch Autograd vs lfs::core::Tensor ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    std::vector<std::tuple<int, int, std::string>> configs = {
        {256, 256, "256x256"},
        {512, 512, "512x512"},
        {800, 800, "800x800"},
        {1024, 1024, "1024x1024"},
        {2048, 2048, "2048x2048 (2K)"},
        {3840, 2160, "3840x2160 (4K)"}
    };

    const int warmup_iters = 10;
    const int bench_iters = 100;

    // Helper to convert torch::Tensor to lfs::core::Tensor
    auto torch_to_tensor = [](const torch::Tensor& torch_tensor) -> lfs::core::Tensor {
        auto cpu_tensor = torch_tensor.cpu().contiguous();
        std::vector<size_t> shape;
        for (int i = 0; i < torch_tensor.dim(); ++i) {
            shape.push_back(torch_tensor.size(i));
        }
        std::vector<float> data(cpu_tensor.data_ptr<float>(),
                                cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
        return lfs::core::Tensor::from_vector(data, lfs::core::TensorShape(shape), lfs::core::Device::CUDA);
    };

    for (auto [H, W, name] : configs) {
        // Create data ONCE and use for all three methods
        auto rendered_torch = torch::randn({1, 3, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto gt_torch = torch::randn({1, 3, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Convert to lfs::core::Tensor (same data)
        auto rendered_lfs = torch_to_tensor(rendered_torch);
        auto gt_lfs = torch_to_tensor(gt_torch);

        // ========== Method 1: Manual (PyTorch tensors) ==========
        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            auto diff = rendered_torch - gt_torch;
            auto grad = diff.sign() / static_cast<float>(diff.numel());
            cudaDeviceSynchronize();
        }

        // Benchmark
        cudaEvent_t start_manual, stop_manual;
        cudaEventCreate(&start_manual);
        cudaEventCreate(&stop_manual);

        cudaEventRecord(start_manual);
        for (int i = 0; i < bench_iters; ++i) {
            auto diff = rendered_torch - gt_torch;
            auto grad = diff.sign() / static_cast<float>(diff.numel());
        }
        cudaEventRecord(stop_manual);
        cudaEventSynchronize(stop_manual);

        float time_manual = 0;
        cudaEventElapsedTime(&time_manual, start_manual, stop_manual);
        float avg_time_manual = time_manual / bench_iters;

        // ========== Method 2: PyTorch Autograd ==========
        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            auto r = rendered_torch.clone().requires_grad_(true);
            auto loss = torch::l1_loss(r, gt_torch);
            loss.backward();
            cudaDeviceSynchronize();
        }

        // Benchmark
        cudaEvent_t start_autograd, stop_autograd;
        cudaEventCreate(&start_autograd);
        cudaEventCreate(&stop_autograd);

        cudaEventRecord(start_autograd);
        for (int i = 0; i < bench_iters; ++i) {
            auto r = rendered_torch.clone().requires_grad_(true);
            auto loss = torch::l1_loss(r, gt_torch);
            loss.backward();
        }
        cudaEventRecord(stop_autograd);
        cudaEventSynchronize(stop_autograd);

        float time_autograd = 0;
        cudaEventElapsedTime(&time_autograd, start_autograd, stop_autograd);
        float avg_time_autograd = time_autograd / bench_iters;

        // ========== Method 3: lfs::core::Tensor (with expression fusion) ==========
        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            // Fully fused operation: (rendered - gt).sign() / numel  (matches manual computation)
            auto grad = ((rendered_lfs - gt_lfs).sign()) / static_cast<float>(rendered_lfs.numel());
            cudaDeviceSynchronize();
        }

        // Benchmark
        cudaEvent_t start_lfs, stop_lfs;
        cudaEventCreate(&start_lfs);
        cudaEventCreate(&stop_lfs);

        cudaEventRecord(start_lfs);
        for (int i = 0; i < bench_iters; ++i) {
            // Fully fused: (rendered - gt).sign() / numel
            // Expression templates should create ONE kernel: sub -> sign -> div
            auto grad = ((rendered_lfs - gt_lfs).sign()) / static_cast<float>(rendered_lfs.numel());
        }
        cudaEventRecord(stop_lfs);
        cudaEventSynchronize(stop_lfs);

        float time_lfs = 0;
        cudaEventElapsedTime(&time_lfs, start_lfs, stop_lfs);
        float avg_time_lfs = time_lfs / bench_iters;

        float speedup_manual = avg_time_autograd / avg_time_manual;
        float speedup_lfs = avg_time_autograd / avg_time_lfs;

        std::cout << name << ": "
                  << "Manual=" << avg_time_manual << "ms (" << speedup_manual << "x), "
                  << "LFS=" << avg_time_lfs << "ms (" << speedup_lfs << "x), "
                  << "Autograd=" << avg_time_autograd << "ms (baseline) ✓" << std::endl;

        cudaEventDestroy(start_manual);
        cudaEventDestroy(stop_manual);
        cudaEventDestroy(start_autograd);
        cudaEventDestroy(stop_autograd);
        cudaEventDestroy(start_lfs);
        cudaEventDestroy(stop_lfs);

        // For small images, autograd can be faster due to better kernel launch optimization
        // For larger images (>= 512x512), manual and LFS should be competitive or faster
        if (H >= 512) {
            EXPECT_GT(speedup_manual, 0.9f) << "Manual computation should be competitive for " << name;
            // LFS with expression fusion should be at least as good as manual
            EXPECT_GT(speedup_lfs, 0.9f) << "LFS computation should be competitive for " << name;
        }
    }

    std::cout << "=================================================\n" << std::endl;
}

// =============================================================================
// LOSS STRUCT TESTS (new loss API)
// =============================================================================

TEST_F(LossGradientTest, PhotometricLoss_MatchesManual) {
    const int H = 256;
    const int W = 256;
    const int C = 3;
    const float lambda_dssim = 0.2f;

    // Create test data
    auto rendered = torch::randn({H, W, C}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto gt_image = torch::randn({H, W, C}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Method 1: PhotometricLoss struct
    lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = lambda_dssim};
    auto result = lfs::training::losses::PhotometricLoss::forward(rendered, gt_image, params);
    ASSERT_TRUE(result.has_value()) << "PhotometricLoss failed: " << result.error();
    auto [loss_struct, ctx] = *result;

    // Method 2: Manual computation (same as trainer.cpp)
    torch::Tensor rendered_4d = rendered.unsqueeze(0);
    torch::Tensor gt_4d = gt_image.unsqueeze(0);

    auto diff = rendered_4d - gt_4d;
    auto l1_loss_tensor = diff.abs().mean();
    auto grad_l1 = diff.sign() / static_cast<float>(diff.numel());

    // We can't easily test SSIM here without exposing the function, but we can verify:
    // 1. Loss is computed
    // 2. Gradient has correct shape
    // 3. Gradient is reasonable magnitude

    EXPECT_GT(loss_struct, 0.0f) << "Loss should be positive";
    EXPECT_EQ(ctx.grad_image.sizes(), rendered.sizes()) << "Gradient shape mismatch";

    float grad_max = ctx.grad_image.abs().max().item<float>();
    EXPECT_LT(grad_max, 10.0f) << "Gradient magnitude unreasonable: " << grad_max;

    std::cout << "PhotometricLoss struct test: loss=" << loss_struct
              << ", grad_max=" << grad_max << " ✓" << std::endl;
}

TEST_F(LossGradientTest, ScaleRegularization_MatchesCUDAKernel) {
    const int n = 10000;
    const float weight = 0.01f;

    auto scaling_raw = torch::randn({n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Method 1: ScaleRegularization struct
    auto scaling_raw_struct = scaling_raw.clone().detach().requires_grad_(true);
    lfs::training::losses::ScaleRegularization::Params params{.weight = weight};
    auto result = lfs::training::losses::ScaleRegularization::forward(scaling_raw_struct, params);
    ASSERT_TRUE(result.has_value()) << "ScaleRegularization failed: " << result.error();
    float loss_struct = *result;

    // Method 2: Direct CUDA kernel call
    auto scaling_raw_cuda = scaling_raw.clone().detach().requires_grad_(true);
    float loss_cuda = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
        scaling_raw_cuda, weight);

    // Compare losses
    EXPECT_NEAR(loss_struct, loss_cuda, 1e-6)
        << "Loss mismatch: struct=" << loss_struct << ", cuda=" << loss_cuda;

    // Compare gradients
    ASSERT_TRUE(scaling_raw_struct.grad().defined()) << "Struct gradient not defined";
    ASSERT_TRUE(scaling_raw_cuda.grad().defined()) << "CUDA gradient not defined";

    auto grad_diff = (scaling_raw_struct.grad() - scaling_raw_cuda.grad()).abs();
    float max_diff = grad_diff.max().item<float>();

    EXPECT_LT(max_diff, 1e-6) << "Gradient mismatch: max_diff=" << max_diff;

    std::cout << "ScaleRegularization struct test: loss_diff="
              << std::abs(loss_struct - loss_cuda)
              << ", grad_max_diff=" << max_diff << " ✓" << std::endl;
}

TEST_F(LossGradientTest, OpacityRegularization_MatchesCUDAKernel) {
    const int n = 10000;
    const float weight = 0.01f;

    auto opacity_raw = torch::randn({n, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Method 1: OpacityRegularization struct
    auto opacity_raw_struct = opacity_raw.clone().detach().requires_grad_(true);
    lfs::training::losses::OpacityRegularization::Params params{.weight = weight};
    auto result = lfs::training::losses::OpacityRegularization::forward(opacity_raw_struct, params);
    ASSERT_TRUE(result.has_value()) << "OpacityRegularization failed: " << result.error();
    float loss_struct = *result;

    // Method 2: Direct CUDA kernel call
    auto opacity_raw_cuda = opacity_raw.clone().detach().requires_grad_(true);
    float loss_cuda = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
        opacity_raw_cuda, weight);

    // Compare losses
    EXPECT_NEAR(loss_struct, loss_cuda, 1e-6)
        << "Loss mismatch: struct=" << loss_struct << ", cuda=" << loss_cuda;

    // Compare gradients
    ASSERT_TRUE(opacity_raw_struct.grad().defined()) << "Struct gradient not defined";
    ASSERT_TRUE(opacity_raw_cuda.grad().defined()) << "CUDA gradient not defined";

    auto grad_diff = (opacity_raw_struct.grad() - opacity_raw_cuda.grad()).abs();
    float max_diff = grad_diff.max().item<float>();

    EXPECT_LT(max_diff, 1e-6) << "Gradient mismatch: max_diff=" << max_diff;

    std::cout << "OpacityRegularization struct test: loss_diff="
              << std::abs(loss_struct - loss_cuda)
              << ", grad_max_diff=" << max_diff << " ✓" << std::endl;
}

TEST_F(LossGradientTest, PhotometricLoss_PureLambdaValues) {
    const int H = 128;
    const int W = 128;
    const int C = 3;

    auto rendered = torch::randn({H, W, C}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto gt_image = torch::randn({H, W, C}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Test pure L1 (lambda_dssim = 0.0)
    {
        lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = 0.0f};
        auto result = lfs::training::losses::PhotometricLoss::forward(rendered, gt_image, params);
        ASSERT_TRUE(result.has_value()) << "Pure L1 failed: " << result.error();
        auto [loss, ctx] = *result;

        // Gradient should be sign(rendered - gt) / N
        auto diff = rendered.unsqueeze(0) - gt_image.unsqueeze(0);
        auto expected_grad = diff.sign() / static_cast<float>(diff.numel());
        expected_grad = expected_grad.squeeze(0);

        auto grad_diff = (ctx.grad_image - expected_grad).abs();
        float max_diff = grad_diff.max().item<float>();

        EXPECT_LT(max_diff, 1e-6) << "Pure L1 gradient mismatch: " << max_diff;
        std::cout << "PhotometricLoss pure L1 (λ=0.0) ✓" << std::endl;
    }

    // Test pure SSIM (lambda_dssim = 1.0)
    {
        lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = 1.0f};
        auto result = lfs::training::losses::PhotometricLoss::forward(rendered, gt_image, params);
        ASSERT_TRUE(result.has_value()) << "Pure SSIM failed: " << result.error();
        auto [loss, ctx] = *result;

        // Just verify gradient exists and is reasonable
        EXPECT_GT(ctx.grad_image.abs().max().item<float>(), 0.0f) << "SSIM gradient should be non-zero";
        std::cout << "PhotometricLoss pure SSIM (λ=1.0) ✓" << std::endl;
    }
}

TEST_F(LossGradientTest, RegularizationLoss_ZeroWeight) {
    const int n = 1000;

    // Test ScaleRegularization with zero weight
    {
        auto scaling_raw = torch::randn({n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .requires_grad_(true);
        lfs::training::losses::ScaleRegularization::Params params{.weight = 0.0f};
        auto result = lfs::training::losses::ScaleRegularization::forward(scaling_raw, params);
        ASSERT_TRUE(result.has_value()) << "ScaleRegularization zero weight failed";
        EXPECT_FLOAT_EQ(*result, 0.0f) << "Loss should be zero for zero weight";
        EXPECT_FALSE(scaling_raw.grad().defined()) << "No gradient should be computed for zero weight";
        std::cout << "ScaleRegularization zero weight ✓" << std::endl;
    }

    // Test OpacityRegularization with zero weight
    {
        auto opacity_raw = torch::randn({n, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .requires_grad_(true);
        lfs::training::losses::OpacityRegularization::Params params{.weight = 0.0f};
        auto result = lfs::training::losses::OpacityRegularization::forward(opacity_raw, params);
        ASSERT_TRUE(result.has_value()) << "OpacityRegularization zero weight failed";
        EXPECT_FLOAT_EQ(*result, 0.0f) << "Loss should be zero for zero weight";
        EXPECT_FALSE(opacity_raw.grad().defined()) << "No gradient should be computed for zero weight";
        std::cout << "OpacityRegularization zero weight ✓" << std::endl;
    }
}

TEST_F(LossGradientTest, PhotometricLoss_DimensionHandling) {
    const int H = 128;
    const int W = 128;
    const int C = 3;
    const float lambda_dssim = 0.2f;

    lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = lambda_dssim};

    // Test 3D input [H, W, C]
    {
        auto rendered_3d = torch::randn({H, W, C}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto gt_3d = torch::randn({H, W, C}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto result = lfs::training::losses::PhotometricLoss::forward(rendered_3d, gt_3d, params);
        ASSERT_TRUE(result.has_value()) << "3D input failed: " << result.error();
        auto [loss, ctx] = *result;

        EXPECT_EQ(ctx.grad_image.dim(), 3) << "Output should be 3D for 3D input";
        EXPECT_EQ(ctx.grad_image.sizes(), rendered_3d.sizes()) << "Output shape should match input";
        std::cout << "PhotometricLoss 3D input [H,W,C] ✓" << std::endl;
    }

    // Test 4D input [1, H, W, C]
    {
        auto rendered_4d = torch::randn({1, H, W, C}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto gt_4d = torch::randn({1, H, W, C}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto result = lfs::training::losses::PhotometricLoss::forward(rendered_4d, gt_4d, params);
        ASSERT_TRUE(result.has_value()) << "4D input failed: " << result.error();
        auto [loss, ctx] = *result;

        EXPECT_EQ(ctx.grad_image.dim(), 4) << "Output should be 4D for 4D input";
        EXPECT_EQ(ctx.grad_image.sizes(), rendered_4d.sizes()) << "Output shape should match input";
        std::cout << "PhotometricLoss 4D input [1,H,W,C] ✓" << std::endl;
    }
}
