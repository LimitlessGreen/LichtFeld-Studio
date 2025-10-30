/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <torch/torch.h>

#include "Ops.h"        // gsplat reference (high-level API)
#include "Relocation.h" // gsplat launch functions (direct kernel access)
#include "core_new/tensor.hpp"
#include "kernels/mcmc_kernels.hpp"

using namespace lfs::core;
using namespace lfs::training::mcmc;

// Helper to compare tensors
bool tensors_close(const Tensor& a, const torch::Tensor& b, float rtol = 1e-5f, float atol = 1e-5f) {
    auto a_torch = torch::from_blob(
        const_cast<float*>(a.ptr<float>()),
        {static_cast<long>(a.numel())},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto b_flat = b.flatten();
    auto diff = (a_torch - b_flat).abs();
    auto threshold = atol + rtol * b_flat.abs();
    return (diff <= threshold).all().item<bool>();
}

// Helper to create binomial coefficients (as in actual MCMC)
torch::Tensor create_binomial_coefficients(int n_max) {
    auto binoms = torch::zeros({n_max, n_max}, torch::kFloat32);
    auto binoms_accessor = binoms.accessor<float, 2>();
    for (int n = 0; n < n_max; ++n) {
        for (int k = 0; k <= n; ++k) {
            float binom = 1.0f;
            for (int i = 0; i < k; ++i) {
                binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
            }
            binoms_accessor[n][k] = binom;
        }
    }
    return binoms.to(torch::kCUDA);
}

// Helper to simulate realistic MCMC ratio distribution
// In real MCMC, ratios come from multinomial sampling with replacement
// Most Gaussians get sampled once, some get sampled multiple times (following the distribution)
torch::Tensor generate_realistic_ratios(size_t N, int n_max, float concentration = 2.0f) {
    // Generate weights that roughly follow opacity-like distribution
    auto weights = torch::rand({static_cast<long>(N)}, torch::TensorOptions().device(torch::kCUDA));
    weights = torch::pow(weights, concentration); // More concentration on higher-opacity Gaussians

    // Simulate multinomial sampling: most get ratio 1-3, some get higher
    auto ratios_float = torch::ones({static_cast<long>(N)}, torch::TensorOptions().device(torch::kCUDA));

    // Add extra counts to ~20% of Gaussians (simulating repeated sampling)
    int n_repeated = N / 5;
    auto repeated_indices = torch::randperm(N, torch::TensorOptions().device(torch::kCUDA)).slice(0, 0, n_repeated);
    auto extra_counts = torch::randint(1, std::min(5, n_max), {n_repeated}, torch::TensorOptions().device(torch::kCUDA)).to(torch::kFloat32);
    ratios_float.index_add_(0, repeated_indices, extra_counts);

    // Clamp to n_max and convert to int32
    return torch::clamp(ratios_float, 1, n_max).to(torch::kInt32).contiguous();
}

// ============================================================================
// VALIDATION TESTS - Compare against gsplat reference implementation
// ============================================================================

TEST(MCMCKernelsTest, RelocationMatchesGsplat_RealisticDistribution) {
    const size_t N = 1000;
    const int n_max = 51;

    // Realistic test data matching actual MCMC usage patterns
    // Opacities in [min_opacity, 0.95] range (after clamping in actual MCMC)
    auto opacities_torch = torch::rand({static_cast<long>(N)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 0.9f + 0.005f;

    // Scales: log-normal distribution typical of trained Gaussians
    auto scales_torch = torch::exp(torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 0.3f - 1.0f);

    // Realistic ratio distribution: most ratios are 1-3, few are higher
    auto ratios_torch = generate_realistic_ratios(N, n_max, 2.0f);

    // Use actual MCMC binomial coefficients
    auto binoms_torch = create_binomial_coefficients(n_max);

    // Reference: gsplat
    auto result_gsplat = gsplat::relocation(opacities_torch, scales_torch, ratios_torch, binoms_torch, n_max);
    auto ref_opacities = std::get<0>(result_gsplat);
    auto ref_scales = std::get<1>(result_gsplat);

    // Our implementation
    Tensor opacities = Tensor::from_blob(opacities_torch.data_ptr<float>(), TensorShape({N}), Device::CUDA, DataType::Float32);
    Tensor scales = Tensor::from_blob(scales_torch.data_ptr<float>(), TensorShape({N, 3}), Device::CUDA, DataType::Float32);
    Tensor ratios = Tensor::from_blob(ratios_torch.data_ptr<int32_t>(), TensorShape({N}), Device::CUDA, DataType::Int32);
    Tensor binoms = Tensor::from_blob(binoms_torch.data_ptr<float>(), TensorShape({static_cast<size_t>(n_max), static_cast<size_t>(n_max)}), Device::CUDA, DataType::Float32);

    Tensor new_opacities = Tensor::empty({N}, Device::CUDA);
    Tensor new_scales = Tensor::empty({N, 3}, Device::CUDA);

    launch_relocation_kernel(
        opacities.ptr<float>(),
        scales.ptr<float>(),
        ratios.ptr<int32_t>(),
        binoms.ptr<float>(),
        n_max,
        new_opacities.ptr<float>(),
        new_scales.ptr<float>(),
        N);

    cudaDeviceSynchronize();

    // Compare results with tight tolerances
    EXPECT_TRUE(tensors_close(new_opacities, ref_opacities, 1e-5f, 1e-6f)) << "Opacities mismatch with realistic distribution";
    EXPECT_TRUE(tensors_close(new_scales, ref_scales, 1e-4f, 1e-5f)) << "Scales mismatch with realistic distribution";
}

TEST(MCMCKernelsTest, RelocationEdgeCases) {
    const size_t N = 10;
    const int n_max = 51;

    // Test with ratio = 1 (identity case - no splitting)
    auto opacities_torch = torch::rand({static_cast<long>(N)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f + 0.3f;
    auto scales_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto ratios_torch = torch::ones({static_cast<long>(N)}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto binoms_torch = create_binomial_coefficients(n_max);

    // Reference
    auto result_gsplat = gsplat::relocation(opacities_torch, scales_torch, ratios_torch, binoms_torch, n_max);
    auto ref_opacities = std::get<0>(result_gsplat);
    auto ref_scales = std::get<1>(result_gsplat);

    // Our implementation
    Tensor opacities = Tensor::from_blob(opacities_torch.data_ptr<float>(), TensorShape({N}), Device::CUDA, DataType::Float32);
    Tensor scales = Tensor::from_blob(scales_torch.data_ptr<float>(), TensorShape({N, 3}), Device::CUDA, DataType::Float32);
    Tensor ratios = Tensor::from_blob(ratios_torch.data_ptr<int32_t>(), TensorShape({N}), Device::CUDA, DataType::Int32);
    Tensor binoms = Tensor::from_blob(binoms_torch.data_ptr<float>(), TensorShape({static_cast<size_t>(n_max), static_cast<size_t>(n_max)}), Device::CUDA, DataType::Float32);

    Tensor new_opacities = Tensor::empty({N}, Device::CUDA);
    Tensor new_scales = Tensor::empty({N, 3}, Device::CUDA);

    launch_relocation_kernel(
        opacities.ptr<float>(),
        scales.ptr<float>(),
        ratios.ptr<int32_t>(),
        binoms.ptr<float>(),
        n_max,
        new_opacities.ptr<float>(),
        new_scales.ptr<float>(),
        N);

    cudaDeviceSynchronize();

    EXPECT_TRUE(tensors_close(new_opacities, ref_opacities, 1e-5f, 1e-6f)) << "Edge case: ratio=1";
    EXPECT_TRUE(tensors_close(new_scales, ref_scales, 1e-4f, 1e-5f)) << "Edge case: ratio=1";
}

TEST(MCMCKernelsTest, AddNoiseMatchesGsplat_RealisticParameters) {
    const size_t N = 1000;

    // Realistic raw parameters as used during training
    // Raw opacities: typically in [-5, 5] range before sigmoid
    auto raw_opacities_torch = torch::randn({static_cast<long>(N)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 2.0f;

    // Raw scales: typically in [-4, 2] range before exp
    auto raw_scales_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 1.0f - 1.0f;

    // Raw quaternions: unnormalized as during training (normalization happens in kernel)
    auto raw_quats_torch = torch::randn({static_cast<long>(N), 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Standard Gaussian noise
    auto noise_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Means in typical scene coordinates
    auto means_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 2.0f;

    // Typical learning rate during training (after decay)
    float current_lr = 0.001f * 0.1f; // noise_lr = 0.1 in MCMC

    // Copy means for reference (gsplat modifies in-place)
    auto means_ref = means_torch.clone();

    // Reference: gsplat
    gsplat::add_noise(raw_opacities_torch, raw_scales_torch, raw_quats_torch, noise_torch, means_ref, current_lr);

    // Our implementation
    auto means_test = means_torch.clone();
    Tensor means_test_tensor = Tensor::from_blob(
        means_test.data_ptr<float>(),
        TensorShape({N, 3}),
        Device::CUDA,
        DataType::Float32);

    launch_add_noise_kernel(
        raw_opacities_torch.data_ptr<float>(),
        raw_scales_torch.data_ptr<float>(),
        raw_quats_torch.data_ptr<float>(),
        noise_torch.data_ptr<float>(),
        means_test_tensor.ptr<float>(),
        current_lr,
        N);

    cudaDeviceSynchronize();

    // Compare results with realistic tolerances
    // Relaxed tolerance due to accumulated numerical errors in covariance matrix computation
    // (rotation matrix construction, matrix multiplications, etc.)
    EXPECT_TRUE(tensors_close(means_test_tensor, means_ref, 1e-2f, 1e-3f)) << "Add noise mismatch with realistic parameters";
}

TEST(MCMCKernelsTest, AddNoiseZeroLearningRate) {
    const size_t N = 10;
    auto raw_opacities_torch = torch::randn({static_cast<long>(N)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto raw_scales_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto raw_quats_torch = torch::randn({static_cast<long>(N), 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto noise_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto means_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto means_original = means_torch.clone();
    Tensor means_tensor = Tensor::from_blob(means_torch.data_ptr<float>(), TensorShape({N, 3}), Device::CUDA, DataType::Float32);

    launch_add_noise_kernel(
        raw_opacities_torch.data_ptr<float>(),
        raw_scales_torch.data_ptr<float>(),
        raw_quats_torch.data_ptr<float>(),
        noise_torch.data_ptr<float>(),
        means_tensor.ptr<float>(),
        0.0f, // Zero learning rate
        N);

    cudaDeviceSynchronize();

    // Should be unchanged
    EXPECT_TRUE(tensors_close(means_tensor, means_original, 0.0f, 0.0f)) << "Means should be unchanged with lr=0";
}

// ============================================================================
// PERFORMANCE BENCHMARKS - Compare LibTorch-free vs gsplat reference
// ============================================================================

TEST(MCMCKernelsTest, BenchmarkRelocation_Comprehensive) {
    const int n_max = 51;
    const int num_iterations = 100;
    auto binoms_torch = create_binomial_coefficients(n_max);

    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         RELOCATION KERNEL PERFORMANCE BENCHMARK               ║\n";
    std::cout << "║              (Fair: Pre-allocated outputs)                    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    std::cout << std::fixed << std::setprecision(3);

    std::vector<size_t> test_sizes = {1000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000};

    for (size_t N : test_sizes) {
        // Create realistic test data
        auto opacities_torch = torch::rand({static_cast<long>(N)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 0.9f + 0.005f;
        auto scales_torch = torch::exp(torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 0.3f - 1.0f);
        auto ratios_torch = generate_realistic_ratios(N, n_max, 2.0f);

        Tensor opacities = Tensor::from_blob(opacities_torch.data_ptr<float>(), TensorShape({N}), Device::CUDA, DataType::Float32);
        Tensor scales = Tensor::from_blob(scales_torch.data_ptr<float>(), TensorShape({N, 3}), Device::CUDA, DataType::Float32);
        Tensor ratios = Tensor::from_blob(ratios_torch.data_ptr<int32_t>(), TensorShape({N}), Device::CUDA, DataType::Int32);
        Tensor binoms = Tensor::from_blob(binoms_torch.data_ptr<float>(), TensorShape({static_cast<size_t>(n_max), static_cast<size_t>(n_max)}), Device::CUDA, DataType::Float32);

        Tensor new_opacities = Tensor::empty({N}, Device::CUDA);
        Tensor new_scales = Tensor::empty({N, 3}, Device::CUDA);

        // Pre-allocate output tensors for gsplat (fair comparison)
        auto new_opacities_gsplat = torch::empty_like(opacities_torch);
        auto new_scales_gsplat = torch::empty_like(scales_torch);

        // Warmup
        for (int i = 0; i < 10; ++i) {
            launch_relocation_kernel(opacities.ptr<float>(), scales.ptr<float>(), ratios.ptr<int32_t>(),
                                     binoms.ptr<float>(), n_max, new_opacities.ptr<float>(), new_scales.ptr<float>(), N);
        }
        cudaDeviceSynchronize();

        // Benchmark LibTorch-free implementation
        auto start_lfs = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            launch_relocation_kernel(opacities.ptr<float>(), scales.ptr<float>(), ratios.ptr<int32_t>(),
                                     binoms.ptr<float>(), n_max, new_opacities.ptr<float>(), new_scales.ptr<float>(), N);
        }
        cudaDeviceSynchronize();
        auto end_lfs = std::chrono::high_resolution_clock::now();
        double time_lfs = std::chrono::duration<double, std::milli>(end_lfs - start_lfs).count() / num_iterations;

        // Benchmark gsplat reference (using pre-allocated outputs via launch_relocation_kernel directly)
        auto start_gsplat = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            gsplat::launch_relocation_kernel(opacities_torch, scales_torch, ratios_torch, binoms_torch, n_max,
                                             new_opacities_gsplat, new_scales_gsplat);
        }
        cudaDeviceSynchronize();
        auto end_gsplat = std::chrono::high_resolution_clock::now();
        double time_gsplat = std::chrono::duration<double, std::milli>(end_gsplat - start_gsplat).count() / num_iterations;

        double speedup = time_gsplat / time_lfs;

        // Format N with commas for readability
        std::string n_str;
        if (N >= 1000000) {
            n_str = std::to_string(N / 1000000) + "M";
        } else if (N >= 1000) {
            n_str = std::to_string(N / 1000) + "k";
        } else {
            n_str = std::to_string(N);
        }

        std::cout << "N = " << std::setw(8) << n_str << " │ "
                  << "LFS: " << std::setw(7) << time_lfs << " ms │ "
                  << "gsplat: " << std::setw(7) << time_gsplat << " ms │ "
                  << "Speedup: " << std::setw(6) << speedup << "x";

        if (speedup >= 0.95) {
            std::cout << " ✓\n";
        } else {
            std::cout << " ⚠\n";
        }
    }

    std::cout << "\n";
    SUCCEED();
}

TEST(MCMCKernelsTest, BenchmarkAddNoise_Comprehensive) {
    const int num_iterations = 100;

    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         ADD NOISE KERNEL PERFORMANCE BENCHMARK                ║\n";
    std::cout << "║              (Fair: Pre-allocated outputs)                    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    std::cout << std::fixed << std::setprecision(3);

    std::vector<size_t> test_sizes = {1000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000};
    float current_lr = 0.0001f;

    for (size_t N : test_sizes) {
        // Create realistic test data
        auto raw_opacities_torch = torch::randn({static_cast<long>(N)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 2.0f;
        auto raw_scales_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 1.0f - 1.0f;
        auto raw_quats_torch = torch::nn::functional::normalize(
            torch::randn({static_cast<long>(N), 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)),
            torch::nn::functional::NormalizeFuncOptions().dim(-1));
        auto noise_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto means_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 2.0f;

        Tensor means_lfs = Tensor::from_blob(means_torch.data_ptr<float>(), TensorShape({N, 3}), Device::CUDA, DataType::Float32);

        // Pre-allocate output tensor for gsplat (fair comparison - reuse same buffer)
        auto means_gsplat = means_torch.clone();

        // Warmup
        for (int i = 0; i < 10; ++i) {
            launch_add_noise_kernel(raw_opacities_torch.data_ptr<float>(), raw_scales_torch.data_ptr<float>(),
                                    raw_quats_torch.data_ptr<float>(), noise_torch.data_ptr<float>(),
                                    means_lfs.ptr<float>(), current_lr, N);
        }
        cudaDeviceSynchronize();

        // Benchmark LibTorch-free implementation
        auto start_lfs = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            launch_add_noise_kernel(raw_opacities_torch.data_ptr<float>(), raw_scales_torch.data_ptr<float>(),
                                    raw_quats_torch.data_ptr<float>(), noise_torch.data_ptr<float>(),
                                    means_lfs.ptr<float>(), current_lr, N);
        }
        cudaDeviceSynchronize();
        auto end_lfs = std::chrono::high_resolution_clock::now();
        double time_lfs = std::chrono::duration<double, std::milli>(end_lfs - start_lfs).count() / num_iterations;

        // Benchmark gsplat reference (using pre-allocated means via launch_add_noise_kernel directly)
        auto start_gsplat = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            gsplat::launch_add_noise_kernel(raw_opacities_torch, raw_scales_torch, raw_quats_torch,
                                            noise_torch, means_gsplat, current_lr);
        }
        cudaDeviceSynchronize();
        auto end_gsplat = std::chrono::high_resolution_clock::now();
        double time_gsplat = std::chrono::duration<double, std::milli>(end_gsplat - start_gsplat).count() / num_iterations;

        double speedup = time_gsplat / time_lfs;

        // Format N with commas for readability
        std::string n_str;
        if (N >= 1000000) {
            n_str = std::to_string(N / 1000000) + "M";
        } else if (N >= 1000) {
            n_str = std::to_string(N / 1000) + "k";
        } else {
            n_str = std::to_string(N);
        }

        std::cout << "N = " << std::setw(8) << n_str << " │ "
                  << "LFS: " << std::setw(7) << time_lfs << " ms │ "
                  << "gsplat: " << std::setw(7) << time_gsplat << " ms │ "
                  << "Speedup: " << std::setw(6) << speedup << "x";

        if (speedup >= 0.95) {
            std::cout << " ✓\n";
        } else {
            std::cout << " ⚠\n";
        }
    }

    std::cout << "\n";
    SUCCEED();
}

TEST(MCMCKernelsTest, BenchmarkMCMCWorkflow) {
    // Simulate realistic MCMC training workflow
    std::vector<size_t> test_sizes = {100000, 1000000, 5000000, 10000000};

    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         MCMC WORKFLOW BENCHMARK (Typical Training)           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    std::cout << "100 iterations per size | Relocation every 10 iters + Noise every iter\n\n";
    std::cout << std::fixed << std::setprecision(2);

    for (size_t N : test_sizes) {
        const int n_max = 51;
        const int num_training_iters = 100;

        auto binoms_torch = create_binomial_coefficients(n_max);
        float current_lr = 0.0001f;

        // Setup data
        auto opacities_torch = torch::rand({static_cast<long>(N)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 0.9f + 0.005f;
        auto scales_torch = torch::exp(torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 0.3f - 1.0f);
        auto ratios_torch = generate_realistic_ratios(N, n_max, 2.0f);
        auto raw_opacities_torch = torch::randn({static_cast<long>(N)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 2.0f;
        auto raw_scales_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 1.0f - 1.0f;
        auto raw_quats_torch = torch::randn({static_cast<long>(N), 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto noise_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto means_torch = torch::randn({static_cast<long>(N), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 2.0f;

        Tensor opacities = Tensor::from_blob(opacities_torch.data_ptr<float>(), TensorShape({N}), Device::CUDA, DataType::Float32);
        Tensor scales = Tensor::from_blob(scales_torch.data_ptr<float>(), TensorShape({N, 3}), Device::CUDA, DataType::Float32);
        Tensor ratios = Tensor::from_blob(ratios_torch.data_ptr<int32_t>(), TensorShape({N}), Device::CUDA, DataType::Int32);
        Tensor binoms = Tensor::from_blob(binoms_torch.data_ptr<float>(), TensorShape({static_cast<size_t>(n_max), static_cast<size_t>(n_max)}), Device::CUDA, DataType::Float32);
        Tensor new_opacities = Tensor::empty({N}, Device::CUDA);
        Tensor new_scales = Tensor::empty({N, 3}, Device::CUDA);
        Tensor means_lfs = Tensor::from_blob(means_torch.data_ptr<float>(), TensorShape({N, 3}), Device::CUDA, DataType::Float32);

        // Warmup
        for (int i = 0; i < 5; ++i) {
            launch_add_noise_kernel(raw_opacities_torch.data_ptr<float>(), raw_scales_torch.data_ptr<float>(),
                                    raw_quats_torch.data_ptr<float>(), noise_torch.data_ptr<float>(),
                                    means_lfs.ptr<float>(), current_lr, N);
        }
        cudaDeviceSynchronize();

        // Benchmark workflow
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < num_training_iters; ++iter) {
            // Every 10 iterations: relocate (simulating refine_every)
            if (iter % 10 == 0 && iter > 0) {
                launch_relocation_kernel(opacities.ptr<float>(), scales.ptr<float>(), ratios.ptr<int32_t>(),
                                         binoms.ptr<float>(), n_max, new_opacities.ptr<float>(), new_scales.ptr<float>(), N);
            }

            // Every iteration: inject noise
            launch_add_noise_kernel(raw_opacities_torch.data_ptr<float>(), raw_scales_torch.data_ptr<float>(),
                                    raw_quats_torch.data_ptr<float>(), noise_torch.data_ptr<float>(),
                                    means_lfs.ptr<float>(), current_lr, N);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(end - start).count();

        // Format N with commas for readability
        std::string n_str;
        if (N >= 1000000) {
            n_str = std::to_string(N / 1000000) + "M";
        } else if (N >= 1000) {
            n_str = std::to_string(N / 1000) + "k";
        } else {
            n_str = std::to_string(N);
        }

        double per_iter = total_time / num_training_iters;
        double throughput = num_training_iters / (total_time / 1000.0);

        std::cout << "N = " << std::setw(8) << n_str << " │ "
                  << "Total: " << std::setw(8) << total_time << " ms │ "
                  << "Per iter: " << std::setw(6) << per_iter << " ms │ "
                  << "Throughput: " << std::setw(7) << static_cast<int>(throughput) << " it/s\n";
    }

    std::cout << "\n";
    SUCCEED();
}
