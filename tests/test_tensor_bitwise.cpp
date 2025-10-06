/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core/tensor.hpp"

using namespace gs;

class TensorBitwiseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set seed for reproducibility
        tensor::manual_seed(42);
    }
};

// ============= Bitwise NOT (~) Tests =============

TEST_F(TensorBitwiseTest, BitwiseNotCPU) {
    std::vector<bool> data = {true, false, true, false};
    auto t = Tensor::from_vector(data, {4}, Device::CPU);
    auto result = ~t;

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.dtype(), DataType::Bool);
    EXPECT_EQ(result.device(), Device::CPU);

    auto values = result.to_vector_bool();
    EXPECT_FALSE(values[0]);  // ~true = false
    EXPECT_TRUE(values[1]);   // ~false = true
    EXPECT_FALSE(values[2]);
    EXPECT_TRUE(values[3]);
}

TEST_F(TensorBitwiseTest, BitwiseNotCUDA) {
    std::vector<bool> data = {true, false, true, false};
    auto t = Tensor::from_vector(data, {4}, Device::CUDA);
    auto result = ~t;

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.dtype(), DataType::Bool);
    EXPECT_EQ(result.device(), Device::CUDA);

    auto values = result.to_vector_bool();
    EXPECT_FALSE(values[0]);
    EXPECT_TRUE(values[1]);
    EXPECT_FALSE(values[2]);
    EXPECT_TRUE(values[3]);
}

TEST_F(TensorBitwiseTest, BitwiseNotMultiDimensional) {
    std::vector<bool> data = {true, false, false, true, true, false};
    auto t = Tensor::from_vector(data, {2, 3}, Device::CPU);
    auto result = ~t;

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.shape(), TensorShape({2, 3}));

    auto values = result.to_vector_bool();
    std::vector<bool> expected = {false, true, true, false, false, true};
    EXPECT_EQ(values, expected);
}

TEST_F(TensorBitwiseTest, BitwiseNotOnNonBoolFails) {
    auto t = Tensor::ones({4}, Device::CPU, DataType::Float32);
    auto result = ~t;

    EXPECT_FALSE(result.is_valid());
}

// ============= Bitwise OR (|) Tests =============

TEST_F(TensorBitwiseTest, BitwiseOrCPU) {
    std::vector<bool> data_a = {true, false, true, false};
    std::vector<bool> data_b = {true, true, false, false};
    auto a = Tensor::from_vector(data_a, {4}, Device::CPU);
    auto b = Tensor::from_vector(data_b, {4}, Device::CPU);
    auto result = a | b;

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.dtype(), DataType::Bool);

    auto values = result.to_vector_bool();
    EXPECT_TRUE(values[0]);   // true | true = true
    EXPECT_TRUE(values[1]);   // false | true = true
    EXPECT_TRUE(values[2]);   // true | false = true
    EXPECT_FALSE(values[3]);  // false | false = false
}

TEST_F(TensorBitwiseTest, BitwiseOrCUDA) {
    std::vector<bool> data_a = {true, false, true, false};
    std::vector<bool> data_b = {true, true, false, false};
    auto a = Tensor::from_vector(data_a, {4}, Device::CUDA);
    auto b = Tensor::from_vector(data_b, {4}, Device::CUDA);
    auto result = a | b;

    ASSERT_TRUE(result.is_valid());
    auto values = result.to_vector_bool();
    EXPECT_TRUE(values[0]);
    EXPECT_TRUE(values[1]);
    EXPECT_TRUE(values[2]);
    EXPECT_FALSE(values[3]);
}

TEST_F(TensorBitwiseTest, BitwiseOrBroadcast) {
    std::vector<bool> data_a = {true, false};
    std::vector<bool> data_b = {true, false, true};
    auto a = Tensor::from_vector(data_a, {2, 1}, Device::CPU);
    auto b = Tensor::from_vector(data_b, {3}, Device::CPU);
    auto result = a | b;

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.shape(), TensorShape({2, 3}));

    auto values = result.to_vector_bool();
    // Row 0: true | {true, false, true} = {true, true, true}
    EXPECT_TRUE(values[0]);
    EXPECT_TRUE(values[1]);
    EXPECT_TRUE(values[2]);
    // Row 1: false | {true, false, true} = {true, false, true}
    EXPECT_TRUE(values[3]);
    EXPECT_FALSE(values[4]);
    EXPECT_TRUE(values[5]);
}

TEST_F(TensorBitwiseTest, BitwiseOrOnNonBoolFails) {
    auto a = Tensor::ones({4}, Device::CPU);
    auto b = Tensor::zeros({4}, Device::CPU);
    auto result = a | b;

    EXPECT_FALSE(result.is_valid());
}

// ============= Logical Operations for Comparison =============

TEST_F(TensorBitwiseTest, LogicalAndVsBitwiseOr) {
    // Test that && and || work as expected (logical, not bitwise)
    std::vector<bool> data_a = {true, false, true, false};
    std::vector<bool> data_b = {true, true, false, false};
    auto a = Tensor::from_vector(data_a, {4}, Device::CPU);
    auto b = Tensor::from_vector(data_b, {4}, Device::CPU);

    auto logical_and = a && b;
    auto logical_or = a || b;
    auto bitwise_or = a | b;

    // Logical OR and bitwise OR should be the same for bool tensors
    auto logical_values = logical_or.to_vector_bool();
    auto bitwise_values = bitwise_or.to_vector_bool();
    EXPECT_EQ(logical_values, bitwise_values);

    // Verify logical AND
    auto and_values = logical_and.to_vector_bool();
    EXPECT_TRUE(and_values[0]);
    EXPECT_FALSE(and_values[1]);
    EXPECT_FALSE(and_values[2]);
    EXPECT_FALSE(and_values[3]);
}

// ============= Combined Operations =============

TEST_F(TensorBitwiseTest, CombinedBitwiseOps) {
    // Test: ~(a | b) == (~a) && (~b)  (De Morgan's law)
    std::vector<bool> data_a = {true, false, true, false};
    std::vector<bool> data_b = {true, true, false, false};
    auto a = Tensor::from_vector(data_a, {4}, Device::CPU);
    auto b = Tensor::from_vector(data_b, {4}, Device::CPU);

    auto left = ~(a | b);
    auto right = (~a) && (~b);

    auto left_values = left.to_vector_bool();
    auto right_values = right.to_vector_bool();

    EXPECT_EQ(left_values, right_values);
}

TEST_F(TensorBitwiseTest, BitwiseNotPreservesShape) {
    std::vector<bool> data = {true, false, true, false, false, true, true, false};
    auto t = Tensor::from_vector(data, {2, 2, 2}, Device::CPU);
    auto result = ~t;

    EXPECT_EQ(result.shape(), t.shape());
    EXPECT_EQ(result.numel(), t.numel());
}

TEST_F(TensorBitwiseTest, BitwiseOrPreservesDeviceMatch) {
    std::vector<bool> data_a = {true, false};
    std::vector<bool> data_b = {false, true};
    auto cpu_a = Tensor::from_vector(data_a, {2}, Device::CPU);
    auto cpu_b = Tensor::from_vector(data_b, {2}, Device::CPU);
    auto result_cpu = cpu_a | cpu_b;
    EXPECT_EQ(result_cpu.device(), Device::CPU);

    auto cuda_a = cpu_a.cuda();
    auto cuda_b = cpu_b.cuda();
    auto result_cuda = cuda_a | cuda_b;
    EXPECT_EQ(result_cuda.device(), Device::CUDA);
}

// ============= Edge Cases =============

TEST_F(TensorBitwiseTest, BitwiseNotEmptyTensor) {
    std::vector<bool> empty_data;
    auto t = Tensor::from_vector(empty_data, {0}, Device::CPU);
    auto result = ~t;

    EXPECT_TRUE(result.is_valid());
    EXPECT_EQ(result.numel(), 0);
}

TEST_F(TensorBitwiseTest, BitwiseOrEmptyTensor) {
    std::vector<bool> empty_data;
    auto a = Tensor::from_vector(empty_data, {0}, Device::CPU);
    auto b = Tensor::from_vector(empty_data, {0}, Device::CPU);
    auto result = a | b;

    EXPECT_TRUE(result.is_valid());
    EXPECT_EQ(result.numel(), 0);
}

TEST_F(TensorBitwiseTest, BitwiseNotLargeTensor) {
    std::vector<bool> data(10000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (i % 2 == 0);
    }

    auto t = Tensor::from_vector(data, {10000}, Device::CUDA);
    auto result = ~t;

    ASSERT_TRUE(result.is_valid());
    auto values = result.to_vector_bool();

    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(values[i], (i % 2 != 0));
    }
}
