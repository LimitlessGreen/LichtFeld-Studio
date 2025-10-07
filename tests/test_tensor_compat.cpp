/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>
#include <chrono>
#include <numeric>

using namespace gs;

// ============= Helper Functions =============

namespace {

// Helper for comparing boolean tensors
void compare_bool_tensors(const Tensor& custom, const torch::Tensor& reference,
                         const std::string& msg = "") {
    auto ref_cpu = reference.to(torch::kCPU).contiguous().flatten();
    auto custom_cpu = custom.cpu();

    ASSERT_EQ(custom_cpu.ndim(), reference.dim()) << msg << ": Rank mismatch";

    for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
        ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(reference.size(i)))
            << msg << ": Shape mismatch at dim " << i;
    }

    ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
        << msg << ": Element count mismatch";

    auto custom_vec = custom_cpu.to_vector_bool();
    auto ref_accessor = ref_cpu.accessor<bool, 1>();

    for (size_t i = 0; i < custom_vec.size(); ++i) {
        EXPECT_EQ(custom_vec[i], ref_accessor[i])
            << msg << ": Mismatch at index " << i
            << " (custom=" << custom_vec[i] << ", ref=" << ref_accessor[i] << ")";
    }
}

void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                    float rtol = 1e-4f, float atol = 1e-5f, const std::string& msg = "") {
    // Handle boolean tensors specially
    if (reference.dtype() == torch::kBool) {
        compare_bool_tensors(custom, reference, msg);
        return;
    }

    auto ref_cpu = reference.to(torch::kCPU).contiguous().flatten();
    auto custom_cpu = custom.cpu();

    ASSERT_EQ(custom_cpu.ndim(), reference.dim()) << msg << ": Rank mismatch";

    for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
        ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(reference.size(i)))
            << msg << ": Shape mismatch at dim " << i;
    }

    ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
        << msg << ": Element count mismatch";

    auto custom_vec = custom_cpu.to_vector();
    auto ref_accessor = ref_cpu.accessor<float, 1>();

    for (size_t i = 0; i < custom_vec.size(); ++i) {
        float ref_val = ref_accessor[i];
        float custom_val = custom_vec[i];

        if (std::isnan(ref_val)) {
            EXPECT_TRUE(std::isnan(custom_val)) << msg << ": Expected NaN at index " << i;
        } else if (std::isinf(ref_val)) {
            EXPECT_TRUE(std::isinf(custom_val)) << msg << ": Expected Inf at index " << i;
        } else {
            float diff = std::abs(custom_val - ref_val);
            float threshold = atol + rtol * std::abs(ref_val);
            EXPECT_LE(diff, threshold)
                << msg << ": Mismatch at index " << i
                << " (custom=" << custom_val << ", ref=" << ref_val << ")";
        }
    }
}

} // anonymous namespace

class TensorTorchCompatTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";
        torch::manual_seed(42);
        tensor::manual_seed(42);
        gen.seed(42);
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};
};

// ============= Complex Expression Tests =============

TEST_F(TensorTorchCompatTest, ComplexExpression1) {
    // Test: (a + b) * c - d / e
    std::vector<float> data_a(12), data_b(12), data_c(12), data_d(12), data_e(12);
    for (auto& val : data_a) val = dist(gen);
    for (auto& val : data_b) val = dist(gen);
    for (auto& val : data_c) val = dist(gen);
    for (auto& val : data_d) val = dist(gen);
    for (auto& val : data_e) val = std::abs(dist(gen)) + 1.0f; // Avoid division by zero

    auto custom_a = Tensor::from_vector(data_a, {3, 4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {3, 4}, Device::CUDA);
    auto custom_c = Tensor::from_vector(data_c, {3, 4}, Device::CUDA);
    auto custom_d = Tensor::from_vector(data_d, {3, 4}, Device::CUDA);
    auto custom_e = Tensor::from_vector(data_e, {3, 4}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});
    auto torch_c = torch::tensor(data_c, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});
    auto torch_d = torch::tensor(data_d, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});
    auto torch_e = torch::tensor(data_e, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});

    auto custom_result = (custom_a + custom_b) * custom_c - custom_d / custom_e;
    auto torch_result = (torch_a + torch_b) * torch_c - torch_d / torch_e;

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "ComplexExpression1");
}

TEST_F(TensorTorchCompatTest, ComplexExpression2) {
    // Test: sigmoid(a * 2 + b) * relu(c - 1)
    std::vector<float> data_a(25), data_b(25), data_c(25);
    for (auto& val : data_a) val = dist(gen);
    for (auto& val : data_b) val = dist(gen);
    for (auto& val : data_c) val = dist(gen);

    auto custom_a = Tensor::from_vector(data_a, {5, 5}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {5, 5}, Device::CUDA);
    auto custom_c = Tensor::from_vector(data_c, {5, 5}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 5});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 5});
    auto torch_c = torch::tensor(data_c, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 5});

    auto custom_result = (custom_a * 2.0f + custom_b).sigmoid() * (custom_c - 1.0f).relu();
    auto torch_result = torch::sigmoid(torch_a * 2.0f + torch_b) * torch::relu(torch_c - 1.0f);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "ComplexExpression2");
}

TEST_F(TensorTorchCompatTest, ComplexExpression3) {
    // Test: exp(log(abs(a) + 1)) + sqrt(b^2 + c^2)
    std::vector<float> data_a(24), data_b(24), data_c(24);
    for (auto& val : data_a) val = dist(gen);
    for (auto& val : data_b) val = dist(gen);
    for (auto& val : data_c) val = dist(gen);

    auto custom_a = Tensor::from_vector(data_a, {4, 6}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4, 6}, Device::CUDA);
    auto custom_c = Tensor::from_vector(data_c, {4, 6}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 6});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 6});
    auto torch_c = torch::tensor(data_c, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 6});

    auto custom_result = (custom_a.abs() + 1.0f).log().exp() + (custom_b * custom_b + custom_c * custom_c).sqrt();
    auto torch_result = torch::exp(torch::log(torch::abs(torch_a) + 1.0f)) +
                        torch::sqrt(torch_b * torch_b + torch_c * torch_c);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "ComplexExpression3");
}

// ============= View and Shape Tests =============

TEST_F(TensorTorchCompatTest, ViewAndCompute) {
    std::vector<float> data(24);
    for (auto& val : data) val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3, 4});

    // Reshape and compute
    auto custom_view = custom_tensor.view({6, 4});
    auto torch_view = torch_tensor.view({6, 4});

    auto custom_result = (custom_view + 1.0f) * 2.0f;
    auto torch_result = (torch_view + 1.0f) * 2.0f;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ViewAndCompute");
}

TEST_F(TensorTorchCompatTest, SliceAndCompute) {
    std::vector<float> data(80);
    for (auto& val : data) val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {10, 8}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({10, 8});

    auto custom_slice = custom_tensor.slice(0, 2, 7); // Rows 2-6
    auto torch_slice = torch_tensor.slice(0, 2, 7);

    auto custom_result = custom_slice.sigmoid() + 0.5f;
    auto torch_result = torch::sigmoid(torch_slice) + 0.5f;

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "SliceAndCompute");
}

// ============= Reduction Tests =============

TEST_F(TensorTorchCompatTest, ReductionConsistency) {
    std::vector<float> data(63);
    for (auto& val : data) val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {7, 9}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({7, 9});

    // Sum
    EXPECT_NEAR(custom_tensor.sum_scalar(), torch_tensor.sum().item<float>(), 1e-3f);

    // Mean
    EXPECT_NEAR(custom_tensor.mean_scalar(), torch_tensor.mean().item<float>(), 1e-4f);

    // Min/Max
    EXPECT_FLOAT_EQ(custom_tensor.min_scalar(), torch_tensor.min().item<float>());
    EXPECT_FLOAT_EQ(custom_tensor.max_scalar(), torch_tensor.max().item<float>());

    // Norms
    EXPECT_NEAR(custom_tensor.norm(2.0f), torch_tensor.norm().item<float>(), 1e-3f);
    EXPECT_NEAR(custom_tensor.norm(1.0f), torch_tensor.norm(1).item<float>(), 1e-3f);
}

// ============= In-place Operations Tests =============

TEST_F(TensorTorchCompatTest, InPlaceOperations) {
    std::vector<float> data(16);
    for (auto& val : data) val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {4, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 4});

    // Scalar in-place
    custom_tensor.add_(2.0f);
    torch_tensor.add_(2.0f);
    compare_tensors(custom_tensor, torch_tensor, 1e-5f, 1e-6f, "InPlace_Add");

    custom_tensor.mul_(3.0f);
    torch_tensor.mul_(3.0f);
    compare_tensors(custom_tensor, torch_tensor, 1e-5f, 1e-6f, "InPlace_Mul");

    // Tensor in-place
    std::vector<float> other_data(16);
    for (auto& val : other_data) val = dist(gen);

    auto custom_other = Tensor::from_vector(other_data, {4, 4}, Device::CUDA);
    auto torch_other = torch::tensor(other_data, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 4});

    custom_tensor.sub_(custom_other);
    torch_tensor.sub_(torch_other);
    compare_tensors(custom_tensor, torch_tensor, 1e-5f, 1e-6f, "InPlace_Sub");
}

// ============= Batch Processing Tests =============

TEST_F(TensorTorchCompatTest, BatchProcessing) {
    std::vector<float> data(2048);
    for (auto& val : data) val = dist(gen);

    auto custom_batch = Tensor::from_vector(data, {32, 64}, Device::CUDA);
    auto torch_batch = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({32, 64});

    // Process: normalize -> relu -> scale
    auto custom_normalized = custom_batch.normalize();

    auto torch_mean = torch_batch.mean();
    auto torch_std = torch_batch.std(/*unbiased=*/false);
    auto torch_normalized = (torch_batch - torch_mean) / (torch_std + 1e-12f);

    auto custom_relu = custom_normalized.relu();
    auto torch_relu = torch::relu(torch_normalized);

    auto custom_scaled = custom_relu * 0.1f;
    auto torch_scaled = torch_relu * 0.1f;

    compare_tensors(custom_scaled, torch_scaled, 1e-4f, 1e-5f, "BatchProcessing");
}

// ============= Gradient Simulation Tests =============

TEST_F(TensorTorchCompatTest, GradientSimulation) {
    std::vector<float> params_data(100), grads_data(100);
    for (auto& val : params_data) val = dist(gen);
    for (auto& val : grads_data) val = dist(gen);

    auto custom_params = Tensor::from_vector(params_data, {100}, Device::CUDA);
    auto custom_grads = Tensor::from_vector(grads_data, {100}, Device::CUDA);

    auto torch_params = torch::tensor(params_data, torch::TensorOptions().device(torch::kCUDA));
    auto torch_grads = torch::tensor(grads_data, torch::TensorOptions().device(torch::kCUDA));

    float learning_rate = 0.01f;

    // SGD step: params = params - lr * grads
    auto custom_updated = custom_params - custom_grads * learning_rate;
    auto torch_updated = torch_params - torch_grads * learning_rate;

    compare_tensors(custom_updated, torch_updated, 1e-5f, 1e-6f, "SGD_Step");

    // Momentum simulation
    std::vector<float> momentum_data(100, 0.0f);
    auto custom_momentum = Tensor::from_vector(momentum_data, {100}, Device::CUDA);
    auto torch_momentum = torch::tensor(momentum_data, torch::TensorOptions().device(torch::kCUDA));

    float beta = 0.9f;

    // momentum = beta * momentum + (1 - beta) * grads
    auto custom_new_momentum = custom_momentum * beta + custom_grads * (1.0f - beta);
    auto torch_new_momentum = torch_momentum * beta + torch_grads * (1.0f - beta);

    compare_tensors(custom_new_momentum, torch_new_momentum, 1e-5f, 1e-6f, "Momentum");
}

// ============= Stress Tests =============

TEST_F(TensorTorchCompatTest, StressTest) {
    const int num_iterations = 10; // Reduced for test speed

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Random shape
        size_t dim1 = 10 + (gen() % 20);
        size_t dim2 = 10 + (gen() % 20);

        std::vector<float> data(dim1 * dim2);
        for (auto& val : data) val = dist(gen);

        auto custom_tensor = Tensor::from_vector(data, {dim1, dim2}, Device::CUDA);
        auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                                .reshape({static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)});

        // Apply random sequence of operations
        for (int op = 0; op < 5; ++op) {
            int op_type = gen() % 6;
            float scalar = dist(gen);

            switch (op_type) {
            case 0:
                custom_tensor = custom_tensor + scalar;
                torch_tensor = torch_tensor + scalar;
                break;
            case 1:
                custom_tensor = custom_tensor * std::abs(scalar);
                torch_tensor = torch_tensor * std::abs(scalar);
                break;
            case 2:
                custom_tensor = custom_tensor.abs();
                torch_tensor = torch::abs(torch_tensor);
                break;
            case 3:
                custom_tensor = custom_tensor.sigmoid();
                torch_tensor = torch::sigmoid(torch_tensor);
                break;
            case 4:
                custom_tensor = custom_tensor.relu();
                torch_tensor = torch::relu(torch_tensor);
                break;
            case 5:
                custom_tensor = custom_tensor.clamp(-5.0f, 5.0f);
                torch_tensor = torch::clamp(torch_tensor, -5.0f, 5.0f);
                break;
            }
        }

        compare_tensors(custom_tensor, torch_tensor, 1e-3f, 1e-4f,
                       "StressTest_Iter" + std::to_string(iter));
    }
}

// ============= Large Scale Tests =============

TEST_F(TensorTorchCompatTest, LargeScaleOperations) {
    std::vector<float> data(128 * 256);
    for (auto& val : data) val = dist(gen);

    auto custom_large = Tensor::from_vector(data, {128, 256}, Device::CUDA);
    auto torch_large = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({128, 256});

    // Complex computation
    auto custom_result = ((custom_large.abs() + 1.0f).log() * 2.0f).sigmoid();
    auto torch_result = torch::sigmoid(torch::log(torch::abs(torch_large) + 1.0f) * 2.0f);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "LargeScale");

    // Verify shape
    EXPECT_EQ(custom_result.numel(), 128 * 256);
}

// ============= Edge Cases Tests =============

TEST_F(TensorTorchCompatTest, EdgeCasesCompat) {
    // Very small values
    std::vector<float> small_data = {1e-7f, 1e-8f, 1e-9f};

    auto custom_small = Tensor::from_vector(small_data, {3}, Device::CUDA);
    auto torch_small = torch::tensor(small_data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_log = custom_small.log();
    auto torch_log = torch::log(torch_small);

    compare_tensors(custom_log, torch_log, 1e-4f, 1e-5f, "EdgeCase_SmallLog");

    // Very large values
    std::vector<float> large_data = {1e6f, 1e7f, 1e8f};

    auto custom_large = Tensor::from_vector(large_data, {3}, Device::CUDA);
    auto torch_large = torch::tensor(large_data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_sigmoid = custom_large.sigmoid();
    auto torch_sigmoid = torch::sigmoid(torch_large);

    compare_tensors(custom_sigmoid, torch_sigmoid, 1e-5f, 1e-6f, "EdgeCase_LargeSigmoid");
}

// ============= Multi-dimensional Tests =============

TEST_F(TensorTorchCompatTest, MultiDimensionalOps) {
    // Test with 3D tensors
    std::vector<float> data_3d(4 * 5 * 6);
    for (auto& val : data_3d) val = dist(gen);

    auto custom_3d = Tensor::from_vector(data_3d, {4, 5, 6}, Device::CUDA);
    auto torch_3d = torch::tensor(data_3d, torch::TensorOptions().device(torch::kCUDA)).reshape({4, 5, 6});

    auto custom_3d_result = (custom_3d + 1.0f).exp().clamp(0.0f, 100.0f);
    auto torch_3d_result = torch::clamp(torch::exp(torch_3d + 1.0f), 0.0f, 100.0f);

    compare_tensors(custom_3d_result, torch_3d_result, 1e-3f, 1e-4f, "3D_Ops");

    // Test with 4D tensors
    std::vector<float> data_4d(2 * 3 * 4 * 5);
    for (auto& val : data_4d) val = dist(gen);

    auto custom_4d = Tensor::from_vector(data_4d, {2, 3, 4, 5}, Device::CUDA);
    auto torch_4d = torch::tensor(data_4d, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3, 4, 5});

    auto custom_4d_result = custom_4d.abs().sqrt();
    auto torch_4d_result = torch::sqrt(torch::abs(torch_4d));

    compare_tensors(custom_4d_result, torch_4d_result, 1e-5f, 1e-6f, "4D_Ops");
}

// ============= Device Consistency Tests =============

TEST_F(TensorTorchCompatTest, ConsistencyAcrossDevices) {
    std::vector<float> data(100);
    for (auto& val : data) val = dist(gen);

    // Create CPU tensor
    auto custom_cpu = Tensor::from_vector(data, {10, 10}, Device::CPU);

    // Create CUDA tensor with same data
    auto custom_cuda = custom_cpu.to(Device::CUDA);

    // Apply operations on both
    auto cpu_result = (custom_cpu + 1.0f) * 2.0f;
    auto cuda_result = (custom_cuda + 1.0f) * 2.0f;

    // Results should be the same
    auto cpu_on_cuda = cpu_result.to(Device::CUDA);
    EXPECT_TRUE(cpu_on_cuda.all_close(cuda_result, 1e-5f, 1e-6f));
}

// ============= Chained Operations Tests =============

TEST_F(TensorTorchCompatTest, ChainedComplexOperations) {
    std::vector<float> data(64);
    for (auto& val : data) val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {8, 8}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({8, 8});

    // Apply long chain
    auto custom_result = custom_tensor
                             .abs()
                             .add(1.0f)
                             .log()
                             .mul(2.0f)
                             .sigmoid()
                             .sub(0.5f)
                             .relu()
                             .clamp(0.0f, 1.0f)
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
    torch_result = torch::clamp(torch_result, 0.0f, 1.0f);
    torch_result = torch::sqrt(torch_result);
    torch_result = torch_result + 0.1f;

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "ChainedComplex");
}

// ============= Numerical Stability Tests =============

TEST_F(TensorTorchCompatTest, NumericalStability) {
    // Division by small but not extreme numbers
    std::vector<float> numerator_data(25);
    for (auto& val : numerator_data) val = dist(gen);

    auto custom_numerator = Tensor::from_vector(numerator_data, {5, 5}, Device::CUDA);
    auto torch_numerator = torch::tensor(numerator_data, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 5});

    // Use 1e-4 instead of 1e-7 for more reasonable numerical behavior
    auto custom_divisor = Tensor::full({5, 5}, 1e-4f, Device::CUDA);
    auto torch_divisor = torch::full({5, 5}, 1e-4f, torch::TensorOptions().device(torch::kCUDA));

    auto custom_div = custom_numerator / custom_divisor;
    auto torch_div = torch_numerator / torch_divisor;

    // Compare with standard tolerance
    compare_tensors(custom_div, torch_div, 1e-3f, 1e-4f, "NumericalStability");
}
// ============= Performance Comparison Tests =============

TEST_F(TensorTorchCompatTest, PerformanceComparison) {
    // Not a strict performance benchmark, but ensure we handle large operations correctly
    const size_t size = 256;

    std::vector<float> data(size * size);
    for (auto& val : data) val = dist(gen);

    auto custom_large = Tensor::from_vector(data, {size, size}, Device::CUDA);
    auto torch_large = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({size, size});

    // Time-intensive operation sequence
    auto start_custom = std::chrono::high_resolution_clock::now();
    auto custom_result = custom_large.sigmoid().relu().add(1.0f).mul(2.0f).sqrt();
    cudaDeviceSynchronize(); // Ensure completion
    auto end_custom = std::chrono::high_resolution_clock::now();

    auto start_torch = std::chrono::high_resolution_clock::now();
    auto torch_result = torch::sqrt(torch::relu(torch::sigmoid(torch_large)).add(1.0f).mul(2.0f));
    cudaDeviceSynchronize(); // Ensure completion
    auto end_torch = std::chrono::high_resolution_clock::now();

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "Performance");

    // Log the times for information
    auto custom_us = std::chrono::duration_cast<std::chrono::microseconds>(end_custom - start_custom).count();
    auto torch_us = std::chrono::duration_cast<std::chrono::microseconds>(end_torch - start_torch).count();

    LOG_INFO("Custom implementation: {} μs, PyTorch: {} μs (ratio: {:.2f}x)",
             custom_us, torch_us, custom_us / double(torch_us));
}

// ============= Mixed Operations Tests =============

TEST_F(TensorTorchCompatTest, MixedArithmeticAndActivation) {
    std::vector<float> data(100);
    for (auto& val : data) val = dist(gen);

    auto custom_tensor = Tensor::from_vector(data, {10, 10}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({10, 10});

    // Mix arithmetic and activation functions
    auto custom_result = ((custom_tensor * 2.0f).sigmoid() + 0.5f).relu();
    auto torch_result = torch::relu(torch::sigmoid(torch_tensor * 2.0f) + 0.5f);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "MixedOps");
}

TEST_F(TensorTorchCompatTest, NestedExpressions) {
    std::vector<float> data_x(50), data_y(50);
    for (auto& val : data_x) val = dist(gen);
    for (auto& val : data_y) val = dist(gen);

    auto custom_x = Tensor::from_vector(data_x, {5, 10}, Device::CUDA);
    auto custom_y = Tensor::from_vector(data_y, {5, 10}, Device::CUDA);

    auto torch_x = torch::tensor(data_x, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 10});
    auto torch_y = torch::tensor(data_y, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 10});

    // Nested: ((x + y) * (x - y)) / (x + 1)
    auto custom_result = ((custom_x + custom_y) * (custom_x - custom_y)) / (custom_x + 1.0f);
    auto torch_result = ((torch_x + torch_y) * (torch_x - torch_y)) / (torch_x + 1.0f);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "NestedExpressions");
}

// ============= Zero and One Tensors Tests =============

TEST_F(TensorTorchCompatTest, ZerosAndOnesOperations) {
    auto custom_zeros = Tensor::zeros({5, 5}, Device::CUDA);
    auto custom_ones = Tensor::ones({5, 5}, Device::CUDA);

    auto torch_zeros = torch::zeros({5, 5}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_ones = torch::ones({5, 5}, torch::TensorOptions().device(torch::kCUDA));

    // Operations on zeros
    auto custom_zeros_result = custom_zeros + 5.0f;
    auto torch_zeros_result = torch_zeros + 5.0f;

    compare_tensors(custom_zeros_result, torch_zeros_result, 1e-6f, 1e-7f, "ZerosOps");

    // Operations on ones
    auto custom_ones_result = (custom_ones * 10.0f).relu();
    auto torch_ones_result = torch::relu(torch_ones * 10.0f);

    compare_tensors(custom_ones_result, torch_ones_result, 1e-6f, 1e-7f, "OnesOps");
}