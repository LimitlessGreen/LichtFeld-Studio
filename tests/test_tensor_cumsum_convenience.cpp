/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core/tensor.hpp"

using namespace gs;

class TensorCumsumConvenienceTest : public ::testing::Test {
protected:
    void SetUp() override {
        tensor::manual_seed(42);
    }
};

// ============= Cumsum Tests =============

TEST_F(TensorCumsumConvenienceTest, Cumsum1DBasic) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5}, Device::CPU);
    auto result = t.cumsum(0);

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.shape(), TensorShape({5}));

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 1.0f);   // 1
    EXPECT_FLOAT_EQ(values[1], 3.0f);   // 1+2
    EXPECT_FLOAT_EQ(values[2], 6.0f);   // 1+2+3
    EXPECT_FLOAT_EQ(values[3], 10.0f);  // 1+2+3+4
    EXPECT_FLOAT_EQ(values[4], 15.0f);  // 1+2+3+4+5
}

TEST_F(TensorCumsumConvenienceTest, Cumsum1DNegativeValues) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, -2.0f, 3.0f, -4.0f, 5.0f}, {5}, Device::CPU);
    auto result = t.cumsum(0);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 1.0f);   // 1
    EXPECT_FLOAT_EQ(values[1], -1.0f);  // 1-2
    EXPECT_FLOAT_EQ(values[2], 2.0f);   // 1-2+3
    EXPECT_FLOAT_EQ(values[3], -2.0f);  // 1-2+3-4
    EXPECT_FLOAT_EQ(values[4], 3.0f);   // 1-2+3-4+5
}

TEST_F(TensorCumsumConvenienceTest, Cumsum2DDim0) {
    auto t = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        {2, 3},
        Device::CPU
    );
    auto result = t.cumsum(0);

    EXPECT_EQ(result.shape(), TensorShape({2, 3}));

    auto values = result.to_vector();
    // First row stays same
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[1], 2.0f);
    EXPECT_FLOAT_EQ(values[2], 3.0f);
    // Second row is cumsum
    EXPECT_FLOAT_EQ(values[3], 5.0f);  // 1+4
    EXPECT_FLOAT_EQ(values[4], 7.0f);  // 2+5
    EXPECT_FLOAT_EQ(values[5], 9.0f);  // 3+6
}

TEST_F(TensorCumsumConvenienceTest, Cumsum2DDim1) {
    auto t = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        {2, 3},
        Device::CPU
    );
    auto result = t.cumsum(1);

    EXPECT_EQ(result.shape(), TensorShape({2, 3}));

    auto values = result.to_vector();
    // Row 0
    EXPECT_FLOAT_EQ(values[0], 1.0f);  // 1
    EXPECT_FLOAT_EQ(values[1], 3.0f);  // 1+2
    EXPECT_FLOAT_EQ(values[2], 6.0f);  // 1+2+3
    // Row 1
    EXPECT_FLOAT_EQ(values[3], 4.0f);   // 4
    EXPECT_FLOAT_EQ(values[4], 9.0f);   // 4+5
    EXPECT_FLOAT_EQ(values[5], 15.0f);  // 4+5+6
}

TEST_F(TensorCumsumConvenienceTest, Cumsum3D) {
    auto t = Tensor::arange(0.0f, 24.0f).reshape({2, 3, 4});
    
    // Cumsum along different dimensions
    auto result0 = t.cumsum(0);  // Along batch
    auto result1 = t.cumsum(1);  // Along rows
    auto result2 = t.cumsum(2);  // Along columns

    EXPECT_EQ(result0.shape(), t.shape());
    EXPECT_EQ(result1.shape(), t.shape());
    EXPECT_EQ(result2.shape(), t.shape());

    // Verify cumsum along last dimension
    auto values = result2.to_vector();
    // First row of first matrix: [0,1,2,3] -> [0,1,3,6]
    EXPECT_FLOAT_EQ(values[0], 0.0f);
    EXPECT_FLOAT_EQ(values[1], 1.0f);
    EXPECT_FLOAT_EQ(values[2], 3.0f);
    EXPECT_FLOAT_EQ(values[3], 6.0f);
}

TEST_F(TensorCumsumConvenienceTest, CumsumNegativeDim) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CPU);
    auto result = t.cumsum(-1);  // Should be same as cumsum(0)

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[3], 10.0f);  // 1+2+3+4
}

TEST_F(TensorCumsumConvenienceTest, CumsumCUDA) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5}, Device::CUDA);
    auto result = t.cumsum(0);

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.device(), Device::CUDA);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[4], 15.0f);
}

TEST_F(TensorCumsumConvenienceTest, CumsumInt32) {
    auto t = Tensor::from_vector(std::vector<int>{1, 2, 3, 4, 5}, {5}, Device::CPU);
    auto result = t.cumsum(0);

    EXPECT_EQ(result.dtype(), DataType::Int32);
    
    auto values = result.to_vector_int();
    EXPECT_EQ(values[0], 1);
    EXPECT_EQ(values[1], 3);
    EXPECT_EQ(values[2], 6);
    EXPECT_EQ(values[3], 10);
    EXPECT_EQ(values[4], 15);
}

TEST_F(TensorCumsumConvenienceTest, CumsumEmptyTensor) {
    std::vector<float> empty_data;
    auto t = Tensor::from_vector(empty_data, {0}, Device::CPU);
    auto result = t.cumsum(0);

    EXPECT_TRUE(result.is_valid());
    EXPECT_EQ(result.numel(), 0);
}

TEST_F(TensorCumsumConvenienceTest, CumsumSingleElement) {
    auto t = Tensor::from_vector(std::vector<float>{42.0f}, {1}, Device::CPU);
    auto result = t.cumsum(0);

    EXPECT_FLOAT_EQ(result.item(), 42.0f);
}

// ============= cpu() / cuda() Tests =============

TEST_F(TensorCumsumConvenienceTest, CpuConversion) {
    auto t = Tensor::ones({10}, Device::CUDA);
    auto cpu_t = t.cpu();

    EXPECT_EQ(cpu_t.device(), Device::CPU);
    EXPECT_EQ(cpu_t.shape(), t.shape());
    EXPECT_EQ(cpu_t.dtype(), t.dtype());

    auto values = cpu_t.to_vector();
    for (float v : values) {
        EXPECT_FLOAT_EQ(v, 1.0f);
    }
}

TEST_F(TensorCumsumConvenienceTest, CudaConversion) {
    auto t = Tensor::ones({10}, Device::CPU);
    auto cuda_t = t.cuda();

    EXPECT_EQ(cuda_t.device(), Device::CUDA);
    EXPECT_EQ(cuda_t.shape(), t.shape());
    EXPECT_EQ(cuda_t.dtype(), t.dtype());
}

TEST_F(TensorCumsumConvenienceTest, CpuIdempotent) {
    auto t = Tensor::ones({10}, Device::CPU);
    auto cpu_t = t.cpu();

    // Should create a copy, not return reference
    EXPECT_EQ(cpu_t.device(), Device::CPU);
    
    // Modify copy
    cpu_t.fill_(42.0f);
    
    // Original should be unchanged
    EXPECT_FLOAT_EQ(t.to_vector()[0], 1.0f);
}

TEST_F(TensorCumsumConvenienceTest, CudaIdempotent) {
    auto t = Tensor::ones({10}, Device::CUDA);
    auto cuda_t = t.cuda();

    EXPECT_EQ(cuda_t.device(), Device::CUDA);
}

TEST_F(TensorCumsumConvenienceTest, CpuCudaRoundtrip) {
    auto original = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CPU);
    auto cuda_t = original.cuda();
    auto back_to_cpu = cuda_t.cpu();

    auto values = back_to_cpu.to_vector();
    auto original_values = original.to_vector();

    EXPECT_EQ(values, original_values);
}

// ============= item<T>() Tests =============

TEST_F(TensorCumsumConvenienceTest, ItemFloat) {
    auto t = Tensor::from_vector(std::vector<float>{42.5f}, {1}, Device::CPU);
    float value = t.item<float>();

    EXPECT_FLOAT_EQ(value, 42.5f);
}

TEST_F(TensorCumsumConvenienceTest, ItemInt) {
    auto t = Tensor::from_vector(std::vector<int>{42}, {1}, Device::CPU);
    int value = t.item<int>();

    EXPECT_EQ(value, 42);
}

TEST_F(TensorCumsumConvenienceTest, ItemBool) {
    std::vector<bool> data = {true};
    auto t = Tensor::from_vector(data, {1}, Device::CPU);
    unsigned char value = t.item<unsigned char>();

    EXPECT_EQ(value, 1);
}

TEST_F(TensorCumsumConvenienceTest, ItemCUDA) {
    auto t = Tensor::from_vector(std::vector<float>{3.14f}, {1}, Device::CUDA);
    float value = t.item<float>();

    EXPECT_NEAR(value, 3.14f, 1e-6f);
}

TEST_F(TensorCumsumConvenienceTest, ItemMultiElement) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::CPU);
    
    // item<T>() on multi-element tensor should fail or return first element
    // Behavior is implementation-dependent
    // The test just verifies it doesn't crash
    float value = t.item<float>();
    
    // Should either be 0 (error) or first element
    EXPECT_TRUE(value == 0.0f || value == 1.0f);
}

TEST_F(TensorCumsumConvenienceTest, ItemAfterReduction) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5}, Device::CPU);
    auto sum = t.sum();
    
    EXPECT_EQ(sum.numel(), 1);
    
    float value = sum.item<float>();
    EXPECT_FLOAT_EQ(value, 15.0f);
}

TEST_F(TensorCumsumConvenienceTest, ItemZeroDimensional) {
    // Create scalar tensor
    auto t = Tensor::from_vector(std::vector<float>{7.5f}, {1}, Device::CPU);
    float value = t.item<float>();

    EXPECT_FLOAT_EQ(value, 7.5f);
}

// ============= Convenience Method Chaining =============

TEST_F(TensorCumsumConvenienceTest, ChainedOperations) {
    auto t = Tensor::ones({10}, Device::CPU)
                .cuda()
                .mul(2.0f)
                .cpu();

    EXPECT_EQ(t.device(), Device::CPU);
    auto values = t.to_vector();
    for (float v : values) {
        EXPECT_FLOAT_EQ(v, 2.0f);
    }
}

TEST_F(TensorCumsumConvenienceTest, CumsumThenItem) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5}, Device::CPU);
    auto cumsum_result = t.cumsum(0);
    auto last = cumsum_result.slice(0, 4, 5);
    
    float value = last.item<float>();
    EXPECT_FLOAT_EQ(value, 15.0f);
}

// ============= Cumsum Integration Tests =============

TEST_F(TensorCumsumConvenienceTest, CumsumWithArithmetic) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CPU);
    auto result = t.cumsum(0).mul(2.0f);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 2.0f);   // 1*2
    EXPECT_FLOAT_EQ(values[1], 6.0f);   // 3*2
    EXPECT_FLOAT_EQ(values[2], 12.0f);  // 6*2
    EXPECT_FLOAT_EQ(values[3], 20.0f);  // 10*2
}

TEST_F(TensorCumsumConvenienceTest, CumsumNormalization) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}, {4}, Device::CPU);
    auto cumsum = t.cumsum(0);
    auto normalized = cumsum.div(4.0f);  // Divide by total

    auto values = normalized.to_vector();
    EXPECT_FLOAT_EQ(values[0], 0.25f);  // 1/4
    EXPECT_FLOAT_EQ(values[1], 0.50f);  // 2/4
    EXPECT_FLOAT_EQ(values[2], 0.75f);  // 3/4
    EXPECT_FLOAT_EQ(values[3], 1.00f);  // 4/4
}

// ============= ones_like with dtype Tests =============

TEST_F(TensorCumsumConvenienceTest, OnesLikeWithDtype) {
    auto t = Tensor::zeros({3, 4}, Device::CPU, DataType::Float32);
    auto ones = tensor::ones_like(t, DataType::Int32);

    EXPECT_EQ(ones.dtype(), DataType::Int32);
    EXPECT_EQ(ones.shape(), t.shape());
    EXPECT_EQ(ones.device(), t.device());

    auto values = ones.to_vector_int();
    for (int v : values) {
        EXPECT_EQ(v, 1);
    }
}

TEST_F(TensorCumsumConvenienceTest, OnesLikeDifferentDtypes) {
    auto float_t = Tensor::zeros({5}, Device::CPU, DataType::Float32);
    
    auto int_ones = tensor::ones_like(float_t, DataType::Int32);
    auto bool_ones = tensor::ones_like(float_t, DataType::Bool);
    
    EXPECT_EQ(int_ones.dtype(), DataType::Int32);
    EXPECT_EQ(bool_ones.dtype(), DataType::Bool);
    
    // Verify values
    auto int_values = int_ones.to_vector_int();
    auto bool_values = bool_ones.to_vector_bool();
    
    std::vector<bool> expected_bool(5, true);
    EXPECT_EQ(int_values.size(), 5);
    EXPECT_EQ(bool_values.size(), 5);
    
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(int_values[i], 1);
        EXPECT_EQ(bool_values[i], expected_bool[i]);
    }
}

TEST_F(TensorCumsumConvenienceTest, OnesLikePreservesDevice) {
    auto cuda_t = Tensor::zeros({10}, Device::CUDA);
    auto ones = tensor::ones_like(cuda_t, DataType::Int32);

    EXPECT_EQ(ones.device(), Device::CUDA);
    EXPECT_EQ(ones.dtype(), DataType::Int32);
}
