/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>
#include <vector>

using namespace gs;

class TensorTorchCompatTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        gen.seed(42);
    }

    // Helper to create matching tensors
    std::pair<Tensor, torch::Tensor> create_matching_tensors(const std::vector<size_t>& shape, bool random = false) {
        size_t total = 1;
        for (auto dim : shape) {
            total *= dim;
        }

        std::vector<float> data(total);
        if (random) {
            for (auto& val : data) {
                val = dist(gen);
            }
        } else {
            std::iota(data.begin(), data.end(), 0.0f);
        }

        // Create our tensor
        auto our_tensor = Tensor::empty(TensorShape(shape), Device::CUDA);
        cudaMemcpy(our_tensor.ptr<float>(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Create torch tensor
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        auto cpu_tensor = torch::from_blob(data.data(), torch_shape, torch::TensorOptions().dtype(torch::kFloat32));
        auto torch_tensor = cpu_tensor.to(torch::kCUDA).clone();

        return {our_tensor, torch_tensor};
    }

    bool verify_same_results(const Tensor& our_tensor, const torch::Tensor& torch_tensor, float tolerance = 1e-5f) {
        auto our_values = our_tensor.to_vector();
        auto torch_cpu = torch_tensor.to(torch::kCPU).contiguous();
        auto torch_data = torch_cpu.data_ptr<float>();

        if (our_values.size() != static_cast<size_t>(torch_cpu.numel())) {
            std::cerr << "Size mismatch: " << our_values.size() << " vs " << torch_cpu.numel() << std::endl;
            return false;
        }

        for (size_t i = 0; i < our_values.size(); ++i) {
            if (std::abs(our_values[i] - torch_data[i]) > tolerance) {
                std::cerr << "Value mismatch at index " << i << ": "
                          << our_values[i] << " vs " << torch_data[i] << std::endl;
                return false;
            }
        }
        return true;
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};
};

TEST_F(TensorTorchCompatTest, ComplexExpression1) {
    // Test: (a + b) * c - d / e
    auto [a, torch_a] = create_matching_tensors({3, 4}, true);
    auto [b, torch_b] = create_matching_tensors({3, 4}, true);
    auto [c, torch_c] = create_matching_tensors({3, 4}, true);
    auto [d, torch_d] = create_matching_tensors({3, 4}, true);
    auto [e, torch_e] = create_matching_tensors({3, 4}, true);

    // Add small constant to e to avoid division by zero
    e = e.abs().add(1.0f);
    torch_e = torch::abs(torch_e) + 1.0f;

    auto our_result = (a + b) * c - d / e;
    auto torch_result = (torch_a + torch_b) * torch_c - torch_d / torch_e;

    EXPECT_TRUE(verify_same_results(our_result, torch_result, 1e-4f));
}

TEST_F(TensorTorchCompatTest, ComplexExpression2) {
    // Test: sigmoid(a * 2 + b) * relu(c - 1)
    auto [a, torch_a] = create_matching_tensors({5, 5}, true);
    auto [b, torch_b] = create_matching_tensors({5, 5}, true);
    auto [c, torch_c] = create_matching_tensors({5, 5}, true);

    auto our_result = (a * 2 + b).sigmoid() * (c - 1).relu();
    auto torch_result = torch::sigmoid(torch_a * 2 + torch_b) * torch::relu(torch_c - 1);

    EXPECT_TRUE(verify_same_results(our_result, torch_result, 1e-4f));
}

TEST_F(TensorTorchCompatTest, ComplexExpression3) {
    // Test: exp(log(abs(a) + 1)) + sqrt(b^2 + c^2)
    auto [a, torch_a] = create_matching_tensors({4, 6}, true);
    auto [b, torch_b] = create_matching_tensors({4, 6}, true);
    auto [c, torch_c] = create_matching_tensors({4, 6}, true);

    auto our_result = (a.abs() + 1).log().exp() + (b * b + c * c).sqrt();
    auto torch_result = torch::exp(torch::log(torch::abs(torch_a) + 1)) +
                        torch::sqrt(torch_b * torch_b + torch_c * torch_c);

    EXPECT_TRUE(verify_same_results(our_result, torch_result, 1e-4f));
}

TEST_F(TensorTorchCompatTest, ViewAndCompute) {
    // Test computations after view operations
    auto [tensor, torch_tensor] = create_matching_tensors({2, 3, 4}, true);

    // Reshape and compute
    auto our_view = tensor.view({6, 4});
    auto torch_view = torch_tensor.view({6, 4});

    auto our_result = (our_view + 1) * 2;
    auto torch_result = (torch_view + 1) * 2;

    EXPECT_TRUE(verify_same_results(our_result, torch_result));
}

TEST_F(TensorTorchCompatTest, SliceAndCompute) {
    // Test computations on sliced tensors
    auto [tensor, torch_tensor] = create_matching_tensors({10, 8}, true);

    auto our_slice = tensor.slice(0, 2, 7); // Rows 2-6
    auto torch_slice = torch_tensor.slice(0, 2, 7);

    auto our_result = our_slice.sigmoid() + 0.5f;
    auto torch_result = torch::sigmoid(torch_slice) + 0.5f;

    EXPECT_TRUE(verify_same_results(our_result, torch_result));
}

TEST_F(TensorTorchCompatTest, ReductionConsistency) {
    // Test that reductions match torch exactly
    auto [tensor, torch_tensor] = create_matching_tensors({7, 9}, true);

    // Sum
    EXPECT_NEAR(tensor.sum(), torch_tensor.sum().item<float>(), 1e-3f);

    // Mean
    EXPECT_NEAR(tensor.mean(), torch_tensor.mean().item<float>(), 1e-4f);

    // Min/Max
    EXPECT_FLOAT_EQ(tensor.min(), torch_tensor.min().item<float>());
    EXPECT_FLOAT_EQ(tensor.max(), torch_tensor.max().item<float>());

    // Norms
    EXPECT_NEAR(tensor.norm(2.0f), torch_tensor.norm().item<float>(), 1e-3f);
    EXPECT_NEAR(tensor.norm(1.0f), torch_tensor.norm(1).item<float>(), 1e-3f);
}

TEST_F(TensorTorchCompatTest, InPlaceOperations) {
    // Test that in-place operations match torch
    auto [our_tensor, torch_tensor] = create_matching_tensors({4, 4}, true);

    // Scalar in-place
    our_tensor.add_(2.0f);
    torch_tensor.add_(2.0f);
    EXPECT_TRUE(verify_same_results(our_tensor, torch_tensor));

    our_tensor.mul_(3.0f);
    torch_tensor.mul_(3.0f);
    EXPECT_TRUE(verify_same_results(our_tensor, torch_tensor));

    // Tensor in-place
    auto [other, torch_other] = create_matching_tensors({4, 4}, true);

    our_tensor.sub_(other);
    torch_tensor.sub_(torch_other);
    EXPECT_TRUE(verify_same_results(our_tensor, torch_tensor));
}

TEST_F(TensorTorchCompatTest, BatchProcessing) {
    // Simulate batch processing
    auto [batch, torch_batch] = create_matching_tensors({32, 64}, true);

    // Process: normalize -> relu -> scale
    auto our_normalized = batch.normalize();
    auto torch_mean = torch_batch.mean();
    auto torch_std = torch_batch.std(false); // unbiased=false
    auto torch_normalized = (torch_batch - torch_mean) / (torch_std + 1e-12f);

    auto our_relu = our_normalized.relu();
    auto torch_relu = torch::relu(torch_normalized);

    auto our_scaled = our_relu * 0.1f;
    auto torch_scaled = torch_relu * 0.1f;

    EXPECT_TRUE(verify_same_results(our_scaled, torch_scaled, 1e-4f));
}

TEST_F(TensorTorchCompatTest, GradientSimulation) {
    // Simulate gradient-like operations (without autograd)
    auto [params, torch_params] = create_matching_tensors({100}, true);
    auto [grads, torch_grads] = create_matching_tensors({100}, true);

    float learning_rate = 0.01f;

    // SGD step: params = params - lr * grads
    auto our_updated = params - grads * learning_rate;
    auto torch_updated = torch_params - torch_grads * learning_rate;

    EXPECT_TRUE(verify_same_results(our_updated, torch_updated));

    // Momentum simulation
    auto [momentum, torch_momentum] = create_matching_tensors({100}, false);
    float beta = 0.9f;

    // momentum = beta * momentum + (1 - beta) * grads
    auto our_new_momentum = momentum * beta + grads * (1 - beta);
    auto torch_new_momentum = torch_momentum * beta + torch_grads * (1 - beta);

    EXPECT_TRUE(verify_same_results(our_new_momentum, torch_new_momentum));
}

TEST_F(TensorTorchCompatTest, StressTest) {
    // Stress test with many operations
    const int num_iterations = 50;

    for (int i = 0; i < num_iterations; ++i) {
        // Random shape
        size_t dim1 = 10 + (gen() % 20);
        size_t dim2 = 10 + (gen() % 20);

        auto [tensor, torch_tensor] = create_matching_tensors({dim1, dim2}, true);

        // Apply random sequence of operations
        for (int op = 0; op < 5; ++op) {
            int op_type = gen() % 6;
            float scalar = dist(gen);

            switch (op_type) {
            case 0:
                tensor = tensor + scalar;
                torch_tensor = torch_tensor + scalar;
                break;
            case 1:
                tensor = tensor * std::abs(scalar);
                torch_tensor = torch_tensor * std::abs(scalar);
                break;
            case 2:
                tensor = tensor.abs();
                torch_tensor = torch::abs(torch_tensor);
                break;
            case 3:
                tensor = tensor.sigmoid();
                torch_tensor = torch::sigmoid(torch_tensor);
                break;
            case 4:
                tensor = tensor.relu();
                torch_tensor = torch::relu(torch_tensor);
                break;
            case 5:
                tensor = tensor.clamp(-5, 5);
                torch_tensor = torch::clamp(torch_tensor, -5, 5);
                break;
            }
        }

        EXPECT_TRUE(verify_same_results(tensor, torch_tensor, 1e-3f))
            << "Failed at iteration " << i;
    }
}

TEST_F(TensorTorchCompatTest, LargeScaleOperations) {
    // Test with larger tensors
    auto [large_tensor, torch_large] = create_matching_tensors({128, 256}, true);

    // Complex computation
    auto our_result = ((large_tensor.abs() + 1).log() * 2).sigmoid();
    auto torch_result = torch::sigmoid(torch::log(torch::abs(torch_large) + 1) * 2);

    EXPECT_TRUE(verify_same_results(our_result, torch_result, 1e-4f));

    // Verify memory efficiency
    EXPECT_EQ(our_result.numel(), 128 * 256);
}

TEST_F(TensorTorchCompatTest, EdgeCasesCompat) {
    // Test edge cases for compatibility

    // Very small values
    auto small_data = std::vector<float>{1e-7f, 1e-8f, 1e-9f};
    auto small_tensor = Tensor::empty({3}, Device::CUDA);
    cudaMemcpy(small_tensor.ptr<float>(), small_data.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    auto torch_small = torch::tensor(small_data, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto our_log = small_tensor.log();
    auto torch_log = torch::log(torch_small);
    EXPECT_TRUE(verify_same_results(our_log, torch_log, 1e-4f));

    // Very large values
    auto large_data = std::vector<float>{1e6f, 1e7f, 1e8f};
    auto large_tensor = Tensor::empty({3}, Device::CUDA);
    cudaMemcpy(large_tensor.ptr<float>(), large_data.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    auto torch_large = torch::tensor(large_data, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto our_sigmoid = large_tensor.sigmoid();
    auto torch_sigmoid = torch::sigmoid(torch_large);
    EXPECT_TRUE(verify_same_results(our_sigmoid, torch_sigmoid));
}

TEST_F(TensorTorchCompatTest, MultiDimensionalOps) {
    // Test with 3D and 4D tensors
    auto [tensor_3d, torch_3d] = create_matching_tensors({4, 5, 6}, true);
    auto [tensor_4d, torch_4d] = create_matching_tensors({2, 3, 4, 5}, true);

    // Operations on 3D
    auto our_3d_result = (tensor_3d + 1).exp().clamp(0, 100);
    auto torch_3d_result = torch::clamp(torch::exp(torch_3d + 1), 0, 100);
    EXPECT_TRUE(verify_same_results(our_3d_result, torch_3d_result, 1e-3f));

    // Operations on 4D
    auto our_4d_result = tensor_4d.abs().sqrt();
    auto torch_4d_result = torch::sqrt(torch::abs(torch_4d));
    EXPECT_TRUE(verify_same_results(our_4d_result, torch_4d_result));
}

TEST_F(TensorTorchCompatTest, ConsistencyAcrossDevices) {
    // Test CPU/CUDA consistency
    std::vector<float> data(100);
    for (auto& val : data) {
        val = dist(gen);
    }

    // Create CPU tensor
    auto cpu_tensor = Tensor::empty({10, 10}, Device::CPU);
    std::memcpy(cpu_tensor.ptr<float>(), data.data(), data.size() * sizeof(float));

    // Create CUDA tensor with same data
    auto cuda_tensor = cpu_tensor.to(Device::CUDA);

    // Apply operations on both
    auto cpu_result = (cpu_tensor + 1) * 2;
    auto cuda_result = (cuda_tensor + 1) * 2;

    // Results should be the same
    auto cpu_on_cuda = cpu_result.to(Device::CUDA);
    EXPECT_TRUE(cpu_on_cuda.all_close(cuda_result, 1e-6f));
}

TEST_F(TensorTorchCompatTest, ChainedComplexOperations) {
    // Test very long chain of operations
    auto [tensor, torch_tensor] = create_matching_tensors({8, 8}, true);

    // Apply long chain
    auto our_result = tensor
                          .abs()
                          .add(1.0f)
                          .log()
                          .mul(2.0f)
                          .sigmoid()
                          .sub(0.5f)
                          .relu()
                          .clamp(0, 1)
                          .sqrt()
                          .add(0.1f);

    auto torch_result = torch_tensor;
    torch_result = torch::abs(torch_result);
    torch_result = torch_result + 1.0f;
    torch_result = torch::log(torch_result);
    torch_result = torch_result * 2.0f;
    torch_result = torch::sigmoid(torch_result);
    torch_result = torch_result - 0.5f;
    torch_result = torch::relu(torch_result);
    torch_result = torch::clamp(torch_result, 0, 1);
    torch_result = torch::sqrt(torch_result);
    torch_result = torch_result + 0.1f;

    EXPECT_TRUE(verify_same_results(our_result, torch_result, 1e-4f));
}

TEST_F(TensorTorchCompatTest, NumericalStability) {
    // Test numerical stability in edge cases

    // Division by very small numbers
    auto [numerator, torch_numerator] = create_matching_tensors({5, 5}, true);
    auto small_divisor = Tensor::full({5, 5}, 1e-7f, Device::CUDA);
    auto torch_small_divisor = torch::full({5, 5}, 1e-7f,
                                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto our_div = numerator / small_divisor;
    auto torch_div = torch_numerator / torch_small_divisor;

    // Results should be large but not infinite
    auto our_values = our_div.to_vector();
    auto torch_values_cpu = torch_div.to(torch::kCPU);
    auto torch_values = torch_values_cpu.data_ptr<float>();

    for (size_t i = 0; i < our_values.size(); ++i) {
        EXPECT_FALSE(std::isinf(our_values[i]));
        EXPECT_FALSE(std::isnan(our_values[i]));
        // Allow larger tolerance for extreme values
        EXPECT_NEAR(our_values[i], torch_values[i], std::abs(torch_values[i]) * 0.01f);
    }
}

TEST_F(TensorTorchCompatTest, PerformanceComparison) {
    // Not a performance benchmark, but ensure we handle large operations
    const size_t large_size = 512;
    auto [large, torch_large] = create_matching_tensors({large_size, large_size}, true);

    // Time-intensive operation sequence
    auto start = std::chrono::high_resolution_clock::now();
    auto our_result = large.sigmoid().relu().add(1).mul(2).sqrt();
    auto our_time = std::chrono::high_resolution_clock::now() - start;

    start = std::chrono::high_resolution_clock::now();
    auto torch_result = torch::sqrt(torch::relu(torch::sigmoid(torch_large)).add(1).mul(2));
    auto torch_time = std::chrono::high_resolution_clock::now() - start;

    EXPECT_TRUE(verify_same_results(our_result, torch_result, 1e-4f));

    // Just log the times for information
    auto our_ms = std::chrono::duration_cast<std::chrono::microseconds>(our_time).count();
    auto torch_ms = std::chrono::duration_cast<std::chrono::microseconds>(torch_time).count();
    std::cout << "Our implementation: " << our_ms << " μs, "
              << "PyTorch: " << torch_ms << " μs" << std::endl;
}
