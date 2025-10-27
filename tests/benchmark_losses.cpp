/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "optimizer/adam_optimizer.hpp"  // For src/training_new include path
#include "losses/regularization.hpp"      // New libtorch-free losses
#include "losses/photometric_loss.hpp"    // New libtorch-free losses
#include "kernels/regularization.cuh"     // Old CUDA kernels for comparison
#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>

using namespace lfs::core;
using namespace lfs::training::losses;

namespace {

// Timing utilities
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }

    double stop_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    double stop_us() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

void print_benchmark_header(const std::string& title) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ " << std::left << std::setw(61) << title << "║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
}

void print_result(const std::string& test_name, double old_time, double new_time, const std::string& unit = "μs") {
    double speedup = old_time / new_time;
    std::cout << std::left << std::setw(45) << test_name << " | ";
    std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << old_time << " " << unit << " | ";
    std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2) << new_time << " " << unit << " | ";
    std::cout << std::right << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x";

    if (speedup > 1.05) {
        std::cout << " ✓";
    } else if (speedup < 0.95) {
        std::cout << " ⚠";
    }
    std::cout << "\n";
}

// Helper to convert lfs::core::Tensor to torch::Tensor
torch::Tensor to_torch(const lfs::core::Tensor& lfs_tensor) {
    auto vec = lfs_tensor.cpu().to_vector();
    std::vector<int64_t> torch_shape;
    for (size_t i = 0; i < lfs_tensor.ndim(); i++) {
        torch_shape.push_back(lfs_tensor.shape()[i]);
    }
    auto torch_t = torch::from_blob(vec.data(), torch_shape, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return lfs_tensor.device() == lfs::core::Device::CUDA ? torch_t.to(torch::kCUDA) : torch_t;
}

// ===================================================================================
// ScaleRegularization Benchmarks
// ===================================================================================

TEST(LossesBenchmark, ScaleRegularization_SingleCall) {
    print_benchmark_header("ScaleRegularization - Single Call Latency");
    std::cout << std::left << std::setw(45) << "Test" << " | ";
    std::cout << std::right << std::setw(10) << "Old (Torch)" << "    | ";
    std::cout << std::right << std::setw(10) << "New (Free)" << "    | ";
    std::cout << std::right << std::setw(6) << "Speedup\n";
    std::cout << std::string(85, '-') << "\n";

    const int n_trials = 1000;
    const size_t n = 10000;
    const float weight = 0.01f;

    // Warmup
    {
        auto scaling_raw = Tensor::randn({100, 3}, Device::CUDA);
        auto scaling_raw_grad = Tensor::zeros({100, 3}, Device::CUDA);
        ScaleRegularization::Params params{.weight = weight};
        ScaleRegularization::forward(scaling_raw, scaling_raw_grad, params);
    }

    // Benchmark old implementation
    auto scaling_raw_old = Tensor::randn({n, 3}, Device::CUDA);
    auto torch_scaling_raw = to_torch(scaling_raw_old);
    torch_scaling_raw.set_requires_grad(true);

    Timer timer;
    timer.start();
    for (int i = 0; i < n_trials; i++) {
        torch_scaling_raw.mutable_grad() = torch::zeros_like(torch_scaling_raw);
        gs::regularization::compute_exp_l1_regularization_with_grad_cuda(torch_scaling_raw, weight);
    }
    cudaDeviceSynchronize();
    double old_time = timer.stop_us() / n_trials;

    // Benchmark new implementation
    auto scaling_raw_new = Tensor::randn({n, 3}, Device::CUDA);
    auto scaling_raw_grad = Tensor::zeros({n, 3}, Device::CUDA);
    ScaleRegularization::Params params{.weight = weight};

    timer.start();
    for (int i = 0; i < n_trials; i++) {
        scaling_raw_grad = Tensor::zeros({n, 3}, Device::CUDA);
        ScaleRegularization::forward(scaling_raw_new, scaling_raw_grad, params);
    }
    cudaDeviceSynchronize();
    double new_time = timer.stop_us() / n_trials;

    print_result("10k Gaussians (1000 trials avg)", old_time, new_time);
}

TEST(LossesBenchmark, ScaleRegularization_Throughput) {
    print_benchmark_header("ScaleRegularization - Throughput");
    std::cout << std::left << std::setw(45) << "Scale" << " | ";
    std::cout << std::right << std::setw(15) << "Old (calls/s)" << " | ";
    std::cout << std::right << std::setw(15) << "New (calls/s)" << " | ";
    std::cout << std::right << std::setw(6) << "Speedup\n";
    std::cout << std::string(90, '-') << "\n";

    const double test_duration_ms = 100.0;
    const float weight = 0.01f;

    std::vector<size_t> sizes = {1000, 10000, 100000};

    for (size_t n : sizes) {
        // Old implementation
        auto scaling_raw_old = Tensor::randn({n, 3}, Device::CUDA);
        auto torch_scaling_raw = to_torch(scaling_raw_old);
        torch_scaling_raw.set_requires_grad(true);

        Timer timer;
        timer.start();
        int old_count = 0;
        while (timer.stop_ms() < test_duration_ms) {
            torch_scaling_raw.mutable_grad() = torch::zeros_like(torch_scaling_raw);
            gs::regularization::compute_exp_l1_regularization_with_grad_cuda(torch_scaling_raw, weight);
            old_count++;
        }
        cudaDeviceSynchronize();
        double old_elapsed = timer.stop_ms();
        double old_throughput = (old_count / old_elapsed) * 1000.0;

        // New implementation
        auto scaling_raw_new = Tensor::randn({n, 3}, Device::CUDA);
        auto scaling_raw_grad = Tensor::zeros({n, 3}, Device::CUDA);
        ScaleRegularization::Params params{.weight = weight};

        timer.start();
        int new_count = 0;
        while (timer.stop_ms() < test_duration_ms) {
            scaling_raw_grad = Tensor::zeros({n, 3}, Device::CUDA);
            ScaleRegularization::forward(scaling_raw_new, scaling_raw_grad, params);
            new_count++;
        }
        cudaDeviceSynchronize();
        double new_elapsed = timer.stop_ms();
        double new_throughput = (new_count / new_elapsed) * 1000.0;

        std::cout << std::left << std::setw(45) << (std::to_string(n) + " Gaussians") << " | ";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(0) << old_throughput << " | ";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(0) << new_throughput << " | ";
        std::cout << std::right << std::setw(6) << std::fixed << std::setprecision(2) << (new_throughput / old_throughput) << "x\n";
    }
}

// ===================================================================================
// OpacityRegularization Benchmarks
// ===================================================================================

TEST(LossesBenchmark, OpacityRegularization_SingleCall) {
    print_benchmark_header("OpacityRegularization - Single Call Latency");
    std::cout << std::left << std::setw(45) << "Test" << " | ";
    std::cout << std::right << std::setw(10) << "Old (Torch)" << "    | ";
    std::cout << std::right << std::setw(10) << "New (Free)" << "    | ";
    std::cout << std::right << std::setw(6) << "Speedup\n";
    std::cout << std::string(85, '-') << "\n";

    const int n_trials = 1000;
    const size_t n = 10000;
    const float weight = 0.01f;

    // Old implementation
    auto opacity_raw_old = Tensor::randn({n, 1}, Device::CUDA);
    auto torch_opacity_raw = to_torch(opacity_raw_old);
    torch_opacity_raw.set_requires_grad(true);

    Timer timer;
    timer.start();
    for (int i = 0; i < n_trials; i++) {
        torch_opacity_raw.mutable_grad() = torch::zeros_like(torch_opacity_raw);
        gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(torch_opacity_raw, weight);
    }
    cudaDeviceSynchronize();
    double old_time = timer.stop_us() / n_trials;

    // New implementation
    auto opacity_raw_new = Tensor::randn({n, 1}, Device::CUDA);
    auto opacity_raw_grad = Tensor::zeros({n, 1}, Device::CUDA);
    OpacityRegularization::Params params{.weight = weight};

    timer.start();
    for (int i = 0; i < n_trials; i++) {
        opacity_raw_grad = Tensor::zeros({n, 1}, Device::CUDA);
        OpacityRegularization::forward(opacity_raw_new, opacity_raw_grad, params);
    }
    cudaDeviceSynchronize();
    double new_time = timer.stop_us() / n_trials;

    print_result("10k Gaussians (1000 trials avg)", old_time, new_time);
}

// ===================================================================================
// PhotometricLoss Benchmarks
// ===================================================================================

TEST(LossesBenchmark, PhotometricLoss_L1Only) {
    print_benchmark_header("PhotometricLoss - L1 Only");
    std::cout << std::left << std::setw(45) << "Image Size" << " | ";
    std::cout << std::right << std::setw(10) << "Time (ms)" << "    | ";
    std::cout << std::right << std::setw(10) << "FPS" << "\n";
    std::cout << std::string(75, '-') << "\n";

    const int n_trials = 100;
    const float lambda_dssim = 0.0f;  // Pure L1

    std::vector<std::pair<int, int>> sizes = {{256, 256}, {512, 512}, {800, 800}, {1024, 1024}};

    for (auto [H, W] : sizes) {
        auto rendered = Tensor::rand({H, W, 3}, Device::CUDA);
        auto gt_image = Tensor::rand({H, W, 3}, Device::CUDA);

        PhotometricLoss::Params params{.lambda_dssim = lambda_dssim};

        Timer timer;
        timer.start();
        for (int i = 0; i < n_trials; i++) {
            auto result = PhotometricLoss::forward(rendered, gt_image, params);
        }
        cudaDeviceSynchronize();
        double elapsed_ms = timer.stop_ms();
        double time_per_call = elapsed_ms / n_trials;
        double fps = 1000.0 / time_per_call;

        std::cout << std::left << std::setw(45) << (std::to_string(H) + "x" + std::to_string(W)) << " | ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_per_call << " ms | ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << fps << "\n";
    }
}

TEST(LossesBenchmark, PhotometricLoss_SSIMOnly) {
    print_benchmark_header("PhotometricLoss - SSIM Only");
    std::cout << std::left << std::setw(45) << "Image Size" << " | ";
    std::cout << std::right << std::setw(10) << "Time (ms)" << "    | ";
    std::cout << std::right << std::setw(10) << "FPS" << "\n";
    std::cout << std::string(75, '-') << "\n";

    const int n_trials = 100;
    const float lambda_dssim = 1.0f;  // Pure SSIM

    std::vector<std::pair<int, int>> sizes = {{256, 256}, {512, 512}, {800, 800}};

    for (auto [H, W] : sizes) {
        auto rendered = Tensor::rand({H, W, 3}, Device::CUDA);
        auto gt_image = Tensor::rand({H, W, 3}, Device::CUDA);

        PhotometricLoss::Params params{.lambda_dssim = lambda_dssim};

        Timer timer;
        timer.start();
        for (int i = 0; i < n_trials; i++) {
            auto result = PhotometricLoss::forward(rendered, gt_image, params);
        }
        cudaDeviceSynchronize();
        double elapsed_ms = timer.stop_ms();
        double time_per_call = elapsed_ms / n_trials;
        double fps = 1000.0 / time_per_call;

        std::cout << std::left << std::setw(45) << (std::to_string(H) + "x" + std::to_string(W)) << " | ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_per_call << " ms | ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << fps << "\n";
    }
}

TEST(LossesBenchmark, PhotometricLoss_Combined) {
    print_benchmark_header("PhotometricLoss - Combined (lambda=0.2)");
    std::cout << std::left << std::setw(45) << "Image Size" << " | ";
    std::cout << std::right << std::setw(10) << "Time (ms)" << "    | ";
    std::cout << std::right << std::setw(10) << "FPS" << "\n";
    std::cout << std::string(75, '-') << "\n";

    const int n_trials = 100;
    const float lambda_dssim = 0.2f;  // Typical training value

    std::vector<std::pair<int, int>> sizes = {{256, 256}, {512, 512}, {800, 800}};

    for (auto [H, W] : sizes) {
        auto rendered = Tensor::rand({H, W, 3}, Device::CUDA);
        auto gt_image = Tensor::rand({H, W, 3}, Device::CUDA);

        PhotometricLoss::Params params{.lambda_dssim = lambda_dssim};

        Timer timer;
        timer.start();
        for (int i = 0; i < n_trials; i++) {
            auto result = PhotometricLoss::forward(rendered, gt_image, params);
        }
        cudaDeviceSynchronize();
        double elapsed_ms = timer.stop_ms();
        double time_per_call = elapsed_ms / n_trials;
        double fps = 1000.0 / time_per_call;

        std::cout << std::left << std::setw(45) << (std::to_string(H) + "x" + std::to_string(W)) << " | ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_per_call << " ms | ";
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << fps << "\n";
    }
}

// ===================================================================================
// Realistic Training Scenario
// ===================================================================================

TEST(LossesBenchmark, RealisticTraining_FullLossPipeline) {
    print_benchmark_header("Realistic Training - Full Loss Computation");
    std::cout << "Simulating typical Gaussian Splatting training iteration\n";
    std::cout << "Configuration: 100k Gaussians, 800x800 image, all losses enabled\n\n";

    const int n_iterations = 100;
    const size_t n_gaussians = 100000;
    const int H = 800, W = 800, C = 3;

    // Setup
    auto scaling_raw = Tensor::randn({n_gaussians, 3}, Device::CUDA);
    auto opacity_raw = Tensor::randn({n_gaussians, 1}, Device::CUDA);
    auto rendered = Tensor::rand({H, W, C}, Device::CUDA);
    auto gt_image = Tensor::rand({H, W, C}, Device::CUDA);

    auto scaling_grad = Tensor::zeros({n_gaussians, 3}, Device::CUDA);
    auto opacity_grad = Tensor::zeros({n_gaussians, 1}, Device::CUDA);

    ScaleRegularization::Params scale_params{.weight = 0.01f};
    OpacityRegularization::Params opacity_params{.weight = 0.01f};
    PhotometricLoss::Params photo_params{.lambda_dssim = 0.2f};

    Timer timer;
    timer.start();

    for (int i = 0; i < n_iterations; i++) {
        // Photometric loss (main loss)
        auto photo_result = PhotometricLoss::forward(rendered, gt_image, photo_params);

        // Regularization losses
        auto scale_result = ScaleRegularization::forward(scaling_raw, scaling_grad, scale_params);
        auto opacity_result = OpacityRegularization::forward(opacity_raw, opacity_grad, opacity_params);
    }

    cudaDeviceSynchronize();
    double total_ms = timer.stop_ms();
    double time_per_iter = total_ms / n_iterations;

    std::cout << std::left << std::setw(40) << "Total time for 100 iterations:" << std::right << std::setw(10)
              << std::fixed << std::setprecision(2) << total_ms << " ms\n";
    std::cout << std::left << std::setw(40) << "Time per iteration:" << std::right << std::setw(10)
              << std::fixed << std::setprecision(3) << time_per_iter << " ms\n";
    std::cout << std::left << std::setw(40) << "Throughput:" << std::right << std::setw(10)
              << std::fixed << std::setprecision(1) << (1000.0 / time_per_iter) << " iter/s\n";
}

// ===================================================================================
// Memory Overhead
// ===================================================================================

TEST(LossesBenchmark, MemoryOverhead) {
    print_benchmark_header("Memory Overhead Analysis");
    std::cout << "Measuring additional memory allocations during loss computation\n\n";

    const size_t n_gaussians = 100000;
    const int H = 512, W = 512, C = 3;

    // Setup tensors
    auto scaling_raw = Tensor::randn({n_gaussians, 3}, Device::CUDA);
    auto opacity_raw = Tensor::randn({n_gaussians, 1}, Device::CUDA);
    auto rendered = Tensor::rand({H, W, C}, Device::CUDA);
    auto gt_image = Tensor::rand({H, W, C}, Device::CUDA);

    auto scaling_grad = Tensor::zeros({n_gaussians, 3}, Device::CUDA);
    auto opacity_grad = Tensor::zeros({n_gaussians, 1}, Device::CUDA);

    // Get initial GPU memory
    cudaDeviceSynchronize();
    size_t free_before, total;
    cudaMemGetInfo(&free_before, &total);

    // Run losses multiple times
    for (int i = 0; i < 10; i++) {
        ScaleRegularization::Params scale_params{.weight = 0.01f};
        ScaleRegularization::forward(scaling_raw, scaling_grad, scale_params);

        OpacityRegularization::Params opacity_params{.weight = 0.01f};
        OpacityRegularization::forward(opacity_raw, opacity_grad, opacity_params);

        PhotometricLoss::Params photo_params{.lambda_dssim = 0.2f};
        PhotometricLoss::forward(rendered, gt_image, photo_params);
    }

    cudaDeviceSynchronize();
    size_t free_after;
    cudaMemGetInfo(&free_after, &total);

    long long memory_delta = static_cast<long long>(free_before) - static_cast<long long>(free_after);

    std::cout << "Memory change after 10 iterations: " << (memory_delta / 1024.0 / 1024.0) << " MB\n";
    std::cout << "(Should be minimal - zero-copy wrappers don't allocate)\n";
}

} // anonymous namespace
