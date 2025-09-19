/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>

using namespace gs;

class TensorMathTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        gen.seed(42);
    }

    Tensor create_tensor_from_vector(const std::vector<float>& data, const std::vector<size_t>& shape) {
        auto tensor = Tensor::empty(TensorShape(shape), Device::CUDA);
        cudaMemcpy(tensor.ptr<float>(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
        return tensor;
    }

    torch::Tensor create_torch_from_vector(const std::vector<float>& data, const std::vector<int64_t>& shape) {
        // CRITICAL FIX: Always create on CPU first, then move to CUDA if needed
        auto cpu_tensor = torch::from_blob(const_cast<float*>(data.data()), shape,
                                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        // Clone to ensure we own the data, then move to CUDA
        return cpu_tensor.clone().to(torch::kCUDA);
    }

    bool compare_tensors(const Tensor& our_tensor, const torch::Tensor& torch_tensor, float tolerance = 1e-5f) {
        auto our_values = our_tensor.to_vector();
        auto torch_cpu = torch_tensor.to(torch::kCPU).contiguous();
        auto torch_data = torch_cpu.data_ptr<float>();

        if (our_values.size() != static_cast<size_t>(torch_cpu.numel())) {
            return false;
        }

        for (size_t i = 0; i < our_values.size(); ++i) {
            if (std::abs(our_values[i] - torch_data[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-5.0f, 5.0f};
    std::uniform_real_distribution<float> positive_dist{0.1f, 5.0f};
};

TEST_F(TensorMathTest, Abs) {
    std::vector<float> data = {-3.5f, 2.1f, -1.0f, 0.0f, 4.2f, -5.5f};
    auto our_tensor = create_tensor_from_vector(data, {2, 3});
    auto torch_tensor = create_torch_from_vector(data, {2, 3});

    auto our_result = our_tensor.abs();
    auto torch_result = torch::abs(torch_tensor);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));

    // Check specific values
    auto values = our_result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 3.5f);
    EXPECT_FLOAT_EQ(values[1], 2.1f);
    EXPECT_FLOAT_EQ(values[2], 1.0f);
    EXPECT_FLOAT_EQ(values[3], 0.0f);
}

TEST_F(TensorMathTest, Sqrt) {
    std::vector<float> data = {0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f};
    auto our_tensor = create_tensor_from_vector(data, {6});
    auto torch_tensor = create_torch_from_vector(data, {6});

    auto our_result = our_tensor.sqrt();
    auto torch_result = torch::sqrt(torch_tensor);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));

    // Check specific values
    auto values = our_result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 0.0f);
    EXPECT_FLOAT_EQ(values[1], 1.0f);
    EXPECT_FLOAT_EQ(values[2], 2.0f);
    EXPECT_FLOAT_EQ(values[3], 3.0f);
    EXPECT_FLOAT_EQ(values[4], 4.0f);
    EXPECT_FLOAT_EQ(values[5], 5.0f);
}

TEST_F(TensorMathTest, SqrtNegativeHandling) {
    // Test that sqrt handles negative values gracefully
    std::vector<float> data = {-1.0f, 4.0f, -9.0f};
    auto tensor = create_tensor_from_vector(data, {3});

    auto result = tensor.sqrt();
    auto values = result.to_vector();

    // Negative values should be clamped to 0 before sqrt
    EXPECT_FLOAT_EQ(values[0], 0.0f);
    EXPECT_FLOAT_EQ(values[1], 2.0f);
    EXPECT_FLOAT_EQ(values[2], 0.0f);
}

TEST_F(TensorMathTest, Exp) {
    std::vector<float> data = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
    auto our_tensor = create_tensor_from_vector(data, {5});
    auto torch_tensor = create_torch_from_vector(data, {5});

    auto our_result = our_tensor.exp();
    auto torch_result = torch::exp(torch_tensor);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));

    // Check specific values
    auto values = our_result.to_vector();
    EXPECT_NEAR(values[0], 1.0f, 1e-5f);
    EXPECT_NEAR(values[1], std::exp(1.0f), 1e-5f);
    EXPECT_NEAR(values[2], std::exp(-1.0f), 1e-5f);
}

TEST_F(TensorMathTest, Log) {
    std::vector<float> data = {1.0f, std::exp(1.0f), 10.0f, 100.0f};
    auto our_tensor = create_tensor_from_vector(data, {4});
    auto torch_tensor = create_torch_from_vector(data, {4});

    auto our_result = our_tensor.log();
    auto torch_result = torch::log(torch_tensor);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));

    // Check specific values
    auto values = our_result.to_vector();
    EXPECT_NEAR(values[0], 0.0f, 1e-5f);
    EXPECT_NEAR(values[1], 1.0f, 1e-5f);
    EXPECT_NEAR(values[2], std::log(10.0f), 1e-5f);
    EXPECT_NEAR(values[3], std::log(100.0f), 1e-5f);
}

TEST_F(TensorMathTest, LogNegativeHandling) {
    // Test that log handles negative/zero values gracefully
    std::vector<float> data = {0.0f, -1.0f, 1.0f};
    auto tensor = create_tensor_from_vector(data, {3});

    auto result = tensor.log();
    auto values = result.to_vector();

    // Values <= 0 should be clamped to eps before log
    EXPECT_FLOAT_EQ(values[2], 0.0f); // log(1) = 0
    // First two values should be large negative but finite
    EXPECT_TRUE(std::isfinite(values[0]));
    EXPECT_TRUE(std::isfinite(values[1]));
}

TEST_F(TensorMathTest, Sigmoid) {
    std::vector<float> data = {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f};
    auto our_tensor = create_tensor_from_vector(data, {5});
    auto torch_tensor = create_torch_from_vector(data, {5});

    auto our_result = our_tensor.sigmoid();
    auto torch_result = torch::sigmoid(torch_tensor);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));

    // Check specific values
    auto values = our_result.to_vector();
    EXPECT_NEAR(values[2], 0.5f, 1e-5f); // sigmoid(0) = 0.5
    EXPECT_NEAR(values[0], 1.0f / (1.0f + std::exp(5.0f)), 1e-5f);
    EXPECT_NEAR(values[4], 1.0f / (1.0f + std::exp(-5.0f)), 1e-5f);
}

TEST_F(TensorMathTest, ReLU) {
    std::vector<float> data = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    auto our_tensor = create_tensor_from_vector(data, {5});
    auto torch_tensor = create_torch_from_vector(data, {5});

    auto our_result = our_tensor.relu();
    auto torch_result = torch::relu(torch_tensor);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));

    // Check specific values
    auto values = our_result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 0.0f);
    EXPECT_FLOAT_EQ(values[1], 0.0f);
    EXPECT_FLOAT_EQ(values[2], 0.0f);
    EXPECT_FLOAT_EQ(values[3], 1.0f);
    EXPECT_FLOAT_EQ(values[4], 3.0f);
}

TEST_F(TensorMathTest, Clamp) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    auto our_tensor = create_tensor_from_vector(data, {5});
    auto torch_tensor = create_torch_from_vector(data, {5});

    float min_val = -1.0f;
    float max_val = 3.0f;

    auto our_result = our_tensor.clamp(min_val, max_val);
    auto torch_result = torch::clamp(torch_tensor, min_val, max_val);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));

    // Check specific values
    auto values = our_result.to_vector();
    EXPECT_FLOAT_EQ(values[0], -1.0f); // Clamped to min
    EXPECT_FLOAT_EQ(values[1], -1.0f); // Clamped to min
    EXPECT_FLOAT_EQ(values[2], 0.0f);  // Within range
    EXPECT_FLOAT_EQ(values[3], 2.0f);  // Within range
    EXPECT_FLOAT_EQ(values[4], 3.0f);  // Clamped to max
}

TEST_F(TensorMathTest, ClampMin) {
    std::vector<float> data = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    auto our_tensor = create_tensor_from_vector(data, {5});
    auto torch_tensor = create_torch_from_vector(data, {5});

    float min_val = 0.5f;

    auto our_result = our_tensor.clamp_min(min_val);
    auto torch_result = torch::clamp_min(torch_tensor, min_val);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));
}

TEST_F(TensorMathTest, ClampMax) {
    std::vector<float> data = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    auto our_tensor = create_tensor_from_vector(data, {5});
    auto torch_tensor = create_torch_from_vector(data, {5});

    float max_val = 0.5f;

    auto our_result = our_tensor.clamp_max(max_val);
    auto torch_result = torch::clamp_max(torch_tensor, max_val);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));
}

TEST_F(TensorMathTest, Normalize) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto our_tensor = create_tensor_from_vector(data, {5});

    auto normalized = our_tensor.normalize();

    // Check mean is approximately 0 - use scalar version
    EXPECT_NEAR(normalized.mean_scalar(), 0.0f, 1e-5f);

    // Check std is approximately 1 - use scalar version
    EXPECT_NEAR(normalized.std_scalar(), 1.0f, 1e-5f);
}


TEST_F(TensorMathTest, RandomDataMathFunctions) {
    // Test with random data for comprehensive coverage
    for (int test = 0; test < 10; ++test) {
        std::vector<float> data(50);
        std::vector<float> positive_data(50);

        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = dist(gen);
            positive_data[i] = positive_dist(gen);
        }

        auto our_tensor = create_tensor_from_vector(data, {50});
        auto torch_tensor = create_torch_from_vector(data, {50});

        auto our_positive = create_tensor_from_vector(positive_data, {50});
        auto torch_positive = create_torch_from_vector(positive_data, {50});

        // Test abs
        auto our_abs = our_tensor.abs();
        auto torch_abs = torch::abs(torch_tensor);
        EXPECT_TRUE(compare_tensors(our_abs, torch_abs, 1e-4f));

        // Test sigmoid
        auto our_sigmoid = our_tensor.sigmoid();
        auto torch_sigmoid = torch::sigmoid(torch_tensor);
        EXPECT_TRUE(compare_tensors(our_sigmoid, torch_sigmoid, 1e-4f));

        // Test relu
        auto our_relu = our_tensor.relu();
        auto torch_relu = torch::relu(torch_tensor);
        EXPECT_TRUE(compare_tensors(our_relu, torch_relu, 1e-4f));

        // Test sqrt (on positive values)
        auto our_sqrt = our_positive.sqrt();
        auto torch_sqrt = torch::sqrt(torch_positive);
        EXPECT_TRUE(compare_tensors(our_sqrt, torch_sqrt, 1e-4f));

        // Test log (on positive values)
        auto our_log = our_positive.log();
        auto torch_log = torch::log(torch_positive);
        EXPECT_TRUE(compare_tensors(our_log, torch_log, 1e-4f));

        // Test exp
        auto small_tensor = our_tensor.clamp(-10, 10); // Avoid overflow
        auto small_torch = torch::clamp(torch_tensor, -10, 10);
        auto our_exp = small_tensor.exp();
        auto torch_exp = torch::exp(small_torch);
        EXPECT_TRUE(compare_tensors(our_exp, torch_exp, 1e-3f));
    }
}

TEST_F(TensorMathTest, ChainedMathOperations) {
    std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto our_tensor = create_tensor_from_vector(data, {5});
    auto torch_tensor = create_torch_from_vector(data, {5});

    // Chain: abs -> add 1 -> log -> exp
    auto our_result = our_tensor.abs().add(1.0f).log().exp();
    auto torch_result = torch::exp(torch::log(torch::abs(torch_tensor) + 1.0f));

    EXPECT_TRUE(compare_tensors(our_result, torch_result, 1e-4f));
}

TEST_F(TensorMathTest, NaNAndInfHandling) {
    auto tensor = Tensor::full({5}, 2.0f, Device::CUDA);

    // Check no NaN initially
    EXPECT_FALSE(tensor.has_nan());
    EXPECT_FALSE(tensor.has_inf());

    // Create NaN by dividing zero by zero
    auto zeros = Tensor::zeros({5}, Device::CUDA);
    auto nan_result = zeros.div(zeros);

    // Note: Our implementation protects against div by zero
    // So this should not actually produce NaN
    EXPECT_FALSE(nan_result.has_nan());

    // Very large exponential might produce inf
    auto large = Tensor::full({5}, 1000.0f, Device::CUDA);
    auto exp_result = large.exp();

    // Check if implementation handles this
    auto values = exp_result.to_vector();
    for (float val : values) {
        // Should either be very large or inf, but not NaN
        EXPECT_FALSE(std::isnan(val));
    }
}

TEST_F(TensorMathTest, AssertFinite) {
    auto tensor = Tensor::full({3, 3}, 1.0f, Device::CUDA);

    // Should not throw for finite values
    EXPECT_NO_THROW(tensor.assert_finite());

    // Test with zero
    auto zeros = Tensor::zeros({2, 2}, Device::CUDA);
    EXPECT_NO_THROW(zeros.assert_finite());

    // Test with negative
    auto negative = Tensor::full({2, 2}, -5.0f, Device::CUDA);
    EXPECT_NO_THROW(negative.assert_finite());
}
