/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>

using namespace gs;

class TensorReductionTest : public ::testing::Test {
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
        auto cpu_tensor = torch::from_blob(const_cast<float*>(data.data()), shape,
                                           torch::TensorOptions().dtype(torch::kFloat32));
        return cpu_tensor.to(torch::kCUDA).clone();
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};
};

TEST_F(TensorReductionTest, Sum) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto our_tensor = create_tensor_from_vector(data, {10});
    auto torch_tensor = create_torch_from_vector(data, {10});

    float our_sum = our_tensor.sum();
    float torch_sum = torch_tensor.sum().item<float>();

    EXPECT_FLOAT_EQ(our_sum, torch_sum);
    EXPECT_FLOAT_EQ(our_sum, 55.0f);
}

TEST_F(TensorReductionTest, SumLargeTensor) {
    // Test with larger tensor
    std::vector<float> data(1000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }

    auto our_tensor = create_tensor_from_vector(data, {1000});
    auto torch_tensor = create_torch_from_vector(data, {1000});

    float our_sum = our_tensor.sum();
    float torch_sum = torch_tensor.sum().item<float>();

    EXPECT_NEAR(our_sum, torch_sum, 1e-3f); // Allow some floating point error
}

TEST_F(TensorReductionTest, Mean) {
    std::vector<float> data = {2, 4, 6, 8, 10};
    auto our_tensor = create_tensor_from_vector(data, {5});
    auto torch_tensor = create_torch_from_vector(data, {5});

    float our_mean = our_tensor.mean();
    float torch_mean = torch_tensor.mean().item<float>();

    EXPECT_FLOAT_EQ(our_mean, torch_mean);
    EXPECT_FLOAT_EQ(our_mean, 6.0f);
}

TEST_F(TensorReductionTest, MeanMultiDim) {
    std::vector<float> data(24);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i + 1);
    }

    auto our_tensor = create_tensor_from_vector(data, {2, 3, 4});
    auto torch_tensor = create_torch_from_vector(data, {2, 3, 4});

    float our_mean = our_tensor.mean();
    float torch_mean = torch_tensor.mean().item<float>();

    EXPECT_FLOAT_EQ(our_mean, torch_mean);
    EXPECT_FLOAT_EQ(our_mean, 12.5f); // Mean of 1..24
}

TEST_F(TensorReductionTest, MinMax) {
    std::vector<float> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    auto our_tensor = create_tensor_from_vector(data, {9});
    auto torch_tensor = create_torch_from_vector(data, {9});

    float our_min = our_tensor.min();
    float our_max = our_tensor.max();
    float torch_min = torch_tensor.min().item<float>();
    float torch_max = torch_tensor.max().item<float>();

    EXPECT_FLOAT_EQ(our_min, torch_min);
    EXPECT_FLOAT_EQ(our_max, torch_max);
    EXPECT_FLOAT_EQ(our_min, 1.0f);
    EXPECT_FLOAT_EQ(our_max, 9.0f);

    // Test minmax
    auto [min_val, max_val] = our_tensor.minmax();
    EXPECT_FLOAT_EQ(min_val, 1.0f);
    EXPECT_FLOAT_EQ(max_val, 9.0f);
}

TEST_F(TensorReductionTest, StandardDeviation) {
    std::vector<float> data = {2, 4, 4, 4, 5, 5, 7, 9};
    auto our_tensor = create_tensor_from_vector(data, {8});
    auto torch_tensor = create_torch_from_vector(data, {8});

    float our_std = our_tensor.std();
    float torch_std = torch_tensor.std(/*unbiased=*/false).item<float>();

    EXPECT_NEAR(our_std, torch_std, 1e-4f);
}

TEST_F(TensorReductionTest, Variance) {
    std::vector<float> data = {1, 2, 3, 4, 5};
    auto our_tensor = create_tensor_from_vector(data, {5});
    auto torch_tensor = create_torch_from_vector(data, {5});

    float our_var = our_tensor.var();
    float torch_var = torch_tensor.var(/*unbiased=*/false).item<float>();

    EXPECT_NEAR(our_var, torch_var, 1e-4f);
    EXPECT_NEAR(our_var, 2.0f, 1e-4f); // Variance of [1,2,3,4,5] = 2
}

TEST_F(TensorReductionTest, L2Norm) {
    std::vector<float> data = {3, 4}; // 3-4-5 triangle
    auto our_tensor = create_tensor_from_vector(data, {2});
    auto torch_tensor = create_torch_from_vector(data, {2});

    float our_norm = our_tensor.norm(2.0f);
    float torch_norm = torch_tensor.norm().item<float>();

    EXPECT_FLOAT_EQ(our_norm, torch_norm);
    EXPECT_FLOAT_EQ(our_norm, 5.0f);
}

TEST_F(TensorReductionTest, L1Norm) {
    std::vector<float> data = {-3, 4, -2, 1};
    auto our_tensor = create_tensor_from_vector(data, {4});
    auto torch_tensor = create_torch_from_vector(data, {4});

    float our_norm = our_tensor.norm(1.0f);
    float torch_norm = torch_tensor.norm(1).item<float>();

    EXPECT_FLOAT_EQ(our_norm, torch_norm);
    EXPECT_FLOAT_EQ(our_norm, 10.0f); // |−3| + |4| + |−2| + |1| = 10
}

TEST_F(TensorReductionTest, Item) {
    // Test item() for scalar tensor
    std::vector<float> data = {3.14f};
    auto our_tensor = create_tensor_from_vector(data, {1});

    float value = our_tensor.item();
    EXPECT_FLOAT_EQ(value, 3.14f);

    // Test that item() fails for non-scalar
    auto multi_elem = create_tensor_from_vector({1, 2, 3}, {3});
    float invalid_value = multi_elem.item();
    EXPECT_FLOAT_EQ(invalid_value, 0.0f); // Should return 0 for error
}

TEST_F(TensorReductionTest, EmptyTensorReductions) {
    auto empty_tensor = Tensor::empty({0}, Device::CUDA);

    EXPECT_FLOAT_EQ(empty_tensor.sum(), 0.0f);
    EXPECT_FLOAT_EQ(empty_tensor.mean(), 0.0f);
    EXPECT_FLOAT_EQ(empty_tensor.min(), 0.0f);
    EXPECT_FLOAT_EQ(empty_tensor.max(), 0.0f);

    auto [min_val, max_val] = empty_tensor.minmax();
    EXPECT_FLOAT_EQ(min_val, 0.0f);
    EXPECT_FLOAT_EQ(max_val, 0.0f);
}

TEST_F(TensorReductionTest, SingleElementReductions) {
    std::vector<float> data = {5.0f};
    auto tensor = create_tensor_from_vector(data, {1});

    EXPECT_FLOAT_EQ(tensor.sum(), 5.0f);
    EXPECT_FLOAT_EQ(tensor.mean(), 5.0f);
    EXPECT_FLOAT_EQ(tensor.min(), 5.0f);
    EXPECT_FLOAT_EQ(tensor.max(), 5.0f);
    EXPECT_FLOAT_EQ(tensor.std(), 0.0f);
    EXPECT_NEAR(tensor.var(), 0.0f, 1e-6f);
    EXPECT_FLOAT_EQ(tensor.norm(2.0f), 5.0f);
    EXPECT_FLOAT_EQ(tensor.item(), 5.0f);
}

TEST_F(TensorReductionTest, NegativeValuesReductions) {
    std::vector<float> data = {-5, -3, -1, 0, 1, 3, 5};
    auto our_tensor = create_tensor_from_vector(data, {7});
    auto torch_tensor = create_torch_from_vector(data, {7});

    EXPECT_FLOAT_EQ(our_tensor.sum(), torch_tensor.sum().item<float>());
    EXPECT_FLOAT_EQ(our_tensor.mean(), torch_tensor.mean().item<float>());
    EXPECT_FLOAT_EQ(our_tensor.min(), torch_tensor.min().item<float>());
    EXPECT_FLOAT_EQ(our_tensor.max(), torch_tensor.max().item<float>());
    EXPECT_NEAR(our_tensor.std(), torch_tensor.std(false).item<float>(), 1e-4f);
}

TEST_F(TensorReductionTest, LargeValueReductions) {
    // Test with large values to check numerical stability
    std::vector<float> data = {1e6f, 1e6f + 1, 1e6f + 2, 1e6f + 3};
    auto our_tensor = create_tensor_from_vector(data, {4});
    auto torch_tensor = create_torch_from_vector(data, {4});

    float our_sum = our_tensor.sum();
    float torch_sum = torch_tensor.sum().item<float>();

    EXPECT_NEAR(our_sum, torch_sum, 10.0f); // Allow some error for large numbers

    float our_mean = our_tensor.mean();
    float torch_mean = torch_tensor.mean().item<float>();

    EXPECT_NEAR(our_mean, torch_mean, 1.0f);
}

TEST_F(TensorReductionTest, RandomDataConsistency) {
    // Test with random data
    for (int test = 0; test < 10; ++test) {
        std::vector<float> data(100);
        for (auto& val : data) {
            val = dist(gen);
        }

        auto our_tensor = create_tensor_from_vector(data, {100});
        auto torch_tensor = create_torch_from_vector(data, {100});

        EXPECT_NEAR(our_tensor.sum(), torch_tensor.sum().item<float>(), 1e-3f);
        EXPECT_NEAR(our_tensor.mean(), torch_tensor.mean().item<float>(), 1e-4f);
        EXPECT_NEAR(our_tensor.min(), torch_tensor.min().item<float>(), 1e-5f);
        EXPECT_NEAR(our_tensor.max(), torch_tensor.max().item<float>(), 1e-5f);
        EXPECT_NEAR(our_tensor.norm(2.0f), torch_tensor.norm().item<float>(), 1e-3f);
        EXPECT_NEAR(our_tensor.norm(1.0f), torch_tensor.norm(1).item<float>(), 1e-3f);
    }
}
