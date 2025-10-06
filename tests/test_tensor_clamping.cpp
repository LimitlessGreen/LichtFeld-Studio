/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include <limits>

using namespace gs;

class TensorClampTest : public ::testing::Test {
protected:
    void SetUp() override {
        tensor::manual_seed(42);
    }
};

// ============= Clamp (non-inplace) Tests =============

TEST_F(TensorClampTest, ClampBasic) {
    auto t = Tensor::from_vector(std::vector<float>{-5.0f, -2.0f, 0.0f, 2.0f, 5.0f}, {5}, Device::CPU);
    auto result = t.clamp(-1.0f, 3.0f);

    ASSERT_TRUE(result.is_valid());
    auto values = result.to_vector();

    EXPECT_FLOAT_EQ(values[0], -1.0f);  // -5 clamped to -1
    EXPECT_FLOAT_EQ(values[1], -1.0f);  // -2 clamped to -1
    EXPECT_FLOAT_EQ(values[2], 0.0f);   // 0 unchanged
    EXPECT_FLOAT_EQ(values[3], 2.0f);   // 2 unchanged
    EXPECT_FLOAT_EQ(values[4], 3.0f);   // 5 clamped to 3

    // Original should be unchanged
    auto original = t.to_vector();
    EXPECT_FLOAT_EQ(original[0], -5.0f);
    EXPECT_FLOAT_EQ(original[4], 5.0f);
}

TEST_F(TensorClampTest, ClampMinOnly) {
    auto t = Tensor::from_vector(std::vector<float>{-5.0f, -2.0f, 0.0f, 2.0f, 5.0f}, {5}, Device::CPU);
    auto result = t.clamp_min(0.0f);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 0.0f);   // -5 clamped to 0
    EXPECT_FLOAT_EQ(values[1], 0.0f);   // -2 clamped to 0
    EXPECT_FLOAT_EQ(values[2], 0.0f);   // 0 unchanged
    EXPECT_FLOAT_EQ(values[3], 2.0f);   // 2 unchanged
    EXPECT_FLOAT_EQ(values[4], 5.0f);   // 5 unchanged
}

TEST_F(TensorClampTest, ClampMaxOnly) {
    auto t = Tensor::from_vector(std::vector<float>{-5.0f, -2.0f, 0.0f, 2.0f, 5.0f}, {5}, Device::CPU);
    auto result = t.clamp_max(3.0f);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], -5.0f);  // -5 unchanged
    EXPECT_FLOAT_EQ(values[1], -2.0f);  // -2 unchanged
    EXPECT_FLOAT_EQ(values[2], 0.0f);   // 0 unchanged
    EXPECT_FLOAT_EQ(values[3], 2.0f);   // 2 unchanged
    EXPECT_FLOAT_EQ(values[4], 3.0f);   // 5 clamped to 3
}

// ============= Clamp_ (in-place) Tests =============

TEST_F(TensorClampTest, ClampInPlaceBasic) {
    auto t = Tensor::from_vector(std::vector<float>{-5.0f, -2.0f, 0.0f, 2.0f, 5.0f}, {5}, Device::CPU);
    auto& result = t.clamp_(-1.0f, 3.0f);

    EXPECT_EQ(&result, &t);  // Should return reference to same object

    auto values = t.to_vector();
    EXPECT_FLOAT_EQ(values[0], -1.0f);
    EXPECT_FLOAT_EQ(values[1], -1.0f);
    EXPECT_FLOAT_EQ(values[2], 0.0f);
    EXPECT_FLOAT_EQ(values[3], 2.0f);
    EXPECT_FLOAT_EQ(values[4], 3.0f);
}

TEST_F(TensorClampTest, ClampMinInPlace) {
    auto t = Tensor::from_vector(std::vector<float>{-5.0f, -2.0f, 0.0f, 2.0f, 5.0f}, {5}, Device::CPU);
    t.clamp_min_(0.0f);

    auto values = t.to_vector();
    EXPECT_FLOAT_EQ(values[0], 0.0f);
    EXPECT_FLOAT_EQ(values[1], 0.0f);
    EXPECT_FLOAT_EQ(values[2], 0.0f);
    EXPECT_FLOAT_EQ(values[3], 2.0f);
    EXPECT_FLOAT_EQ(values[4], 5.0f);
}

TEST_F(TensorClampTest, ClampMaxInPlace) {
    auto t = Tensor::from_vector(std::vector<float>{-5.0f, -2.0f, 0.0f, 2.0f, 5.0f}, {5}, Device::CPU);
    t.clamp_max_(3.0f);

    auto values = t.to_vector();
    EXPECT_FLOAT_EQ(values[0], -5.0f);
    EXPECT_FLOAT_EQ(values[1], -2.0f);
    EXPECT_FLOAT_EQ(values[2], 0.0f);
    EXPECT_FLOAT_EQ(values[3], 2.0f);
    EXPECT_FLOAT_EQ(values[4], 3.0f);
}

// ============= CUDA Tests =============

TEST_F(TensorClampTest, ClampCUDA) {
    auto t = Tensor::from_vector(std::vector<float>{-5.0f, -2.0f, 0.0f, 2.0f, 5.0f}, {5}, Device::CUDA);
    auto result = t.clamp(-1.0f, 3.0f);

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.device(), Device::CUDA);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], -1.0f);
    EXPECT_FLOAT_EQ(values[4], 3.0f);
}

TEST_F(TensorClampTest, ClampInPlaceCUDA) {
    auto t = Tensor::from_vector(std::vector<float>{-5.0f, -2.0f, 0.0f, 2.0f, 5.0f}, {5}, Device::CUDA);
    t.clamp_(-1.0f, 3.0f);

    auto values = t.to_vector();
    EXPECT_FLOAT_EQ(values[0], -1.0f);
    EXPECT_FLOAT_EQ(values[4], 3.0f);
}

TEST_F(TensorClampTest, ClampMinCUDA) {
    auto t = Tensor::from_vector(std::vector<float>{-5.0f, 0.0f, 5.0f}, {3}, Device::CUDA);
    auto result = t.clamp_min(0.0f);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 0.0f);
    EXPECT_FLOAT_EQ(values[1], 0.0f);
    EXPECT_FLOAT_EQ(values[2], 5.0f);
}

TEST_F(TensorClampTest, ClampMaxCUDA) {
    auto t = Tensor::from_vector(std::vector<float>{-5.0f, 0.0f, 5.0f}, {3}, Device::CUDA);
    auto result = t.clamp_max(3.0f);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], -5.0f);
    EXPECT_FLOAT_EQ(values[1], 0.0f);
    EXPECT_FLOAT_EQ(values[2], 3.0f);
}

// ============= Multi-dimensional Tests =============

TEST_F(TensorClampTest, Clamp2D) {
    auto t = Tensor::from_vector(
        std::vector<float>{-5.0f, -2.0f, 0.0f, 2.0f, 5.0f, 10.0f},
        {2, 3},
        Device::CPU
    );
    auto result = t.clamp(-1.0f, 3.0f);

    EXPECT_EQ(result.shape(), TensorShape({2, 3}));

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], -1.0f);  // -5 -> -1
    EXPECT_FLOAT_EQ(values[1], -1.0f);  // -2 -> -1
    EXPECT_FLOAT_EQ(values[2], 0.0f);   // 0 -> 0
    EXPECT_FLOAT_EQ(values[3], 2.0f);   // 2 -> 2
    EXPECT_FLOAT_EQ(values[4], 3.0f);   // 5 -> 3
    EXPECT_FLOAT_EQ(values[5], 3.0f);   // 10 -> 3
}

TEST_F(TensorClampTest, Clamp3D) {
    auto t = Tensor::randn({2, 3, 4}, Device::CPU);
    auto result = t.clamp(-1.0f, 1.0f);

    EXPECT_EQ(result.shape(), t.shape());

    auto values = result.to_vector();
    for (float v : values) {
        EXPECT_GE(v, -1.0f);
        EXPECT_LE(v, 1.0f);
    }
}

// ============= Edge Cases =============

TEST_F(TensorClampTest, ClampWithEqualMinMax) {
    auto t = Tensor::from_vector(std::vector<float>{-5.0f, 0.0f, 5.0f}, {3}, Device::CPU);
    auto result = t.clamp(2.0f, 2.0f);

    // All values should be 2.0
    auto values = result.to_vector();
    for (float v : values) {
        EXPECT_FLOAT_EQ(v, 2.0f);
    }
}

TEST_F(TensorClampTest, ClampWithInfinity) {
    auto t = Tensor::from_vector(std::vector<float>{-1000.0f, 0.0f, 1000.0f}, {3}, Device::CPU);
    
    // Clamp with infinity as max
    auto result = t.clamp(0.0f, std::numeric_limits<float>::infinity());
    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 0.0f);
    EXPECT_FLOAT_EQ(values[1], 0.0f);
    EXPECT_FLOAT_EQ(values[2], 1000.0f);

    // Clamp with -infinity as min
    auto result2 = t.clamp(-std::numeric_limits<float>::infinity(), 0.0f);
    auto values2 = result2.to_vector();
    EXPECT_FLOAT_EQ(values2[0], -1000.0f);
    EXPECT_FLOAT_EQ(values2[1], 0.0f);
    EXPECT_FLOAT_EQ(values2[2], 0.0f);
}

TEST_F(TensorClampTest, ClampEmptyTensor) {
    std::vector<float> empty_data;
    auto t = Tensor::from_vector(empty_data, {0}, Device::CPU);
    auto result = t.clamp(-1.0f, 1.0f);

    EXPECT_TRUE(result.is_valid());
    EXPECT_EQ(result.numel(), 0);
}

TEST_F(TensorClampTest, ClampInPlaceEmptyTensor) {
    std::vector<float> empty_data;
    auto t = Tensor::from_vector(empty_data, {0}, Device::CPU);
    t.clamp_(-1.0f, 1.0f);

    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.numel(), 0);
}

TEST_F(TensorClampTest, ClampScalarTensor) {
    auto t = Tensor::from_vector(std::vector<float>{5.0f}, {1}, Device::CPU);
    auto result = t.clamp(0.0f, 10.0f);

    auto value = result.item();
    EXPECT_FLOAT_EQ(value, 5.0f);
}

// ============= Chain Operations =============

TEST_F(TensorClampTest, ClampChaining) {
    auto t = Tensor::randn({100}, Device::CPU);
    
    // Chain multiple clamps
    auto result = t.clamp(-10.0f, 10.0f)
                   .clamp(-5.0f, 5.0f)
                   .clamp(-1.0f, 1.0f);

    auto values = result.to_vector();
    for (float v : values) {
        EXPECT_GE(v, -1.0f);
        EXPECT_LE(v, 1.0f);
    }
}

TEST_F(TensorClampTest, ClampInPlaceChaining) {
    auto t = Tensor::randn({100}, Device::CPU);
    
    t.clamp_min_(-5.0f).clamp_max_(5.0f);

    auto values = t.to_vector();
    for (float v : values) {
        EXPECT_GE(v, -5.0f);
        EXPECT_LE(v, 5.0f);
    }
}

// ============= Integration with Other Operations =============

TEST_F(TensorClampTest, ClampAfterArithmetic) {
    auto a = Tensor::randn({100}, Device::CPU);
    auto b = Tensor::randn({100}, Device::CPU);
    
    auto result = (a + b).clamp(-1.0f, 1.0f);

    auto values = result.to_vector();
    for (float v : values) {
        EXPECT_GE(v, -1.0f);
        EXPECT_LE(v, 1.0f);
    }
}

TEST_F(TensorClampTest, ClampGradient) {
    // Test that clamp behaves correctly for gradient-like operations
    auto grad = Tensor::randn({100}, Device::CPU);
    
    // Clamp gradients (common in optimization)
    auto clamped = grad.clamp(-1.0f, 1.0f);

    // Verify no NaNs or Infs
    EXPECT_FALSE(clamped.has_nan());
    EXPECT_FALSE(clamped.has_inf());
}

// ============= Performance Test =============

TEST_F(TensorClampTest, ClampLargeScale) {
    auto t = Tensor::randn({1000, 1000}, Device::CUDA);
    auto result = t.clamp(-1.0f, 1.0f);

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.shape(), t.shape());

    // Sample check
    auto cpu_result = result.cpu();
    auto values = cpu_result.debug_values(100);
    for (float v : values) {
        EXPECT_GE(v, -1.0f);
        EXPECT_LE(v, 1.0f);
    }
}

// ============= Non-Float32 Behavior =============

TEST_F(TensorClampTest, ClampNonFloat32) {
    // Test that clamp_ on non-float32 is handled
    auto t = Tensor::ones({5}, Device::CPU, DataType::Int32);
    t.clamp_(-1.0f, 1.0f);

    // Should either work or gracefully handle
    EXPECT_TRUE(t.is_valid());
}
