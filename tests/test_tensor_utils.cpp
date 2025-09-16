/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include <cuda_runtime.h>

using namespace gs;
using namespace gs::tensor;

class TensorUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

// Test arange function
TEST_F(TensorUtilsTest, ArangeBasic) {
    // Test basic arange
    auto t1 = arange(5);
    EXPECT_EQ(t1.numel(), 5);
    auto vals1 = t1.to_vector();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(vals1[i], static_cast<float>(i));
    }

    // Test arange with start and end
    auto t2 = arange(2, 7);
    EXPECT_EQ(t2.numel(), 5);
    auto vals2 = t2.to_vector();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(vals2[i], static_cast<float>(i + 2));
    }

    // Test arange with step
    auto t3 = arange(0, 10, 2);
    EXPECT_EQ(t3.numel(), 5);
    auto vals3 = t3.to_vector();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(vals3[i], static_cast<float>(i * 2));
    }
}

TEST_F(TensorUtilsTest, ArangeNegativeStep) {
    // Test negative step
    auto t = arange(10, 0, -2);
    EXPECT_EQ(t.numel(), 5);
    auto vals = t.to_vector();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(vals[i], static_cast<float>(10 - i * 2));
    }

    // Test invalid range with negative step
    auto invalid = arange(0, 10, -1);  // Start < end with negative step
    EXPECT_FALSE(invalid.is_valid());
}

TEST_F(TensorUtilsTest, ArangeEdgeCases) {
    // Test invalid range
    auto invalid1 = arange(5, 0, 1);  // Start > end with positive step
    EXPECT_FALSE(invalid1.is_valid());

    // Test zero step
    auto invalid2 = arange(0, 5, 0);  // Zero step
    EXPECT_FALSE(invalid2.is_valid());

    // Test single element
    auto single = arange(0, 1, 1);
    EXPECT_EQ(single.numel(), 1);
    EXPECT_FLOAT_EQ(single.item(), 0.0f);
}

TEST_F(TensorUtilsTest, Linspace) {
    // Test basic linspace
    auto t1 = linspace(0, 10, 11);
    EXPECT_EQ(t1.numel(), 11);
    auto vals1 = t1.to_vector();
    for (size_t i = 0; i < 11; ++i) {
        EXPECT_FLOAT_EQ(vals1[i], static_cast<float>(i));
    }

    // Test linspace with 2 points
    auto t2 = linspace(-5, 5, 2);
    EXPECT_EQ(t2.numel(), 2);
    auto vals2 = t2.to_vector();
    EXPECT_FLOAT_EQ(vals2[0], -5.0f);
    EXPECT_FLOAT_EQ(vals2[1], 5.0f);

    // Test single point
    auto t3 = linspace(3.14f, 3.14f, 1);
    EXPECT_EQ(t3.numel(), 1);
    EXPECT_FLOAT_EQ(t3.item(), 3.14f);

    // Test invalid (0 steps)
    auto invalid = linspace(0, 1, 0);
    EXPECT_FALSE(invalid.is_valid());
}

TEST_F(TensorUtilsTest, Stack) {
    // Create test tensors
    auto t1 = Tensor::full({3, 4}, 1.0f, Device::CUDA);
    auto t2 = Tensor::full({3, 4}, 2.0f, Device::CUDA);
    auto t3 = Tensor::full({3, 4}, 3.0f, Device::CUDA);

    // Stack along dimension 0
    std::vector<Tensor> tensors;
    tensors.push_back(t1.clone());
    tensors.push_back(t2.clone());
    tensors.push_back(t3.clone());
    
    auto stacked = stack(std::move(tensors), 0);
    EXPECT_EQ(stacked.shape().rank(), 3);
    EXPECT_EQ(stacked.shape()[0], 3);
    EXPECT_EQ(stacked.shape()[1], 3);
    EXPECT_EQ(stacked.shape()[2], 4);

    // Verify values
    auto values = stacked.to_vector();
    // First tensor should be all 1s
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(values[i], 1.0f);
    }
    // Second tensor should be all 2s
    for (size_t i = 12; i < 24; ++i) {
        EXPECT_FLOAT_EQ(values[i], 2.0f);
    }
    // Third tensor should be all 3s
    for (size_t i = 24; i < 36; ++i) {
        EXPECT_FLOAT_EQ(values[i], 3.0f);
    }
}

TEST_F(TensorUtilsTest, StackEmpty) {
    // Test stacking empty list
    std::vector<Tensor> empty_list;
    auto invalid = stack(std::move(empty_list));
    EXPECT_FALSE(invalid.is_valid());
}

TEST_F(TensorUtilsTest, StackMismatchedShapes) {
    // Test stacking tensors with different shapes
    std::vector<Tensor> mismatched;
    mismatched.push_back(Tensor::zeros({2, 3}, Device::CUDA));
    mismatched.push_back(Tensor::zeros({3, 3}, Device::CUDA));  // Different first dimension
    auto invalid = stack(std::move(mismatched));
    EXPECT_FALSE(invalid.is_valid());
}

TEST_F(TensorUtilsTest, Concatenate) {
    // Create test tensors with different sizes along dim 0
    auto t1 = Tensor::full({2, 3}, 1.0f, Device::CUDA);
    auto t2 = Tensor::full({3, 3}, 2.0f, Device::CUDA);
    auto t3 = Tensor::full({1, 3}, 3.0f, Device::CUDA);

    // Concatenate along dimension 0
    std::vector<Tensor> tensors;
    tensors.push_back(t1.clone());
    tensors.push_back(t2.clone());
    tensors.push_back(t3.clone());
    
    auto concatenated = cat(std::move(tensors), 0);
    EXPECT_EQ(concatenated.shape().rank(), 2);
    EXPECT_EQ(concatenated.shape()[0], 6);  // 2 + 3 + 1
    EXPECT_EQ(concatenated.shape()[1], 3);

    // Verify values
    auto values = concatenated.to_vector();
    // First 6 values (2x3) should be 1s
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(values[i], 1.0f);
    }
    // Next 9 values (3x3) should be 2s
    for (size_t i = 6; i < 15; ++i) {
        EXPECT_FLOAT_EQ(values[i], 2.0f);
    }
    // Last 3 values (1x3) should be 3s
    for (size_t i = 15; i < 18; ++i) {
        EXPECT_FLOAT_EQ(values[i], 3.0f);
    }
}

TEST_F(TensorUtilsTest, ConcatenateMismatchedDimensions) {
    // Test mismatched shapes (should fail)
    std::vector<Tensor> mismatched;
    mismatched.push_back(Tensor::zeros({2, 3}, Device::CUDA));
    mismatched.push_back(Tensor::zeros({2, 4}, Device::CUDA));  // Different dim 1
    auto invalid = cat(std::move(mismatched), 0);
    EXPECT_FALSE(invalid.is_valid());
}

TEST_F(TensorUtilsTest, ConcatenateNonZeroDim) {
    // Test concatenation along non-zero dimension (currently not implemented)
    std::vector<Tensor> tensors;
    tensors.push_back(Tensor::zeros({2, 3}, Device::CUDA));
    tensors.push_back(Tensor::zeros({2, 4}, Device::CUDA));
    auto result = cat(std::move(tensors), 1);
    EXPECT_FALSE(result.is_valid());  // Should fail as only dim=0 is implemented
}

TEST_F(TensorUtilsTest, TensorBuilder) {
    // Test basic builder
    auto t1 = TensorBuilder()
        .with_shape({3, 4, 5})
        .on_device(Device::CUDA)
        .with_dtype(DataType::Float32)
        .filled_with(2.5f)
        .build();

    EXPECT_TRUE(t1.is_valid());
    EXPECT_EQ(t1.shape().rank(), 3);
    EXPECT_EQ(t1.device(), Device::CUDA);
    EXPECT_EQ(t1.dtype(), DataType::Float32);
    
    auto values = t1.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 2.5f);
    }

    // Test builder without fill value
    auto t2 = TensorBuilder()
        .with_shape({10})
        .on_device(Device::CPU)
        .build();

    EXPECT_TRUE(t2.is_valid());
    EXPECT_EQ(t2.device(), Device::CPU);
    EXPECT_EQ(t2.numel(), 10);
}

TEST_F(TensorUtilsTest, SafeOperations) {
    // Test safe division
    auto a = Tensor::full({3, 3}, 10.0f, Device::CUDA);
    auto b = Tensor::full({3, 3}, 0.0f, Device::CUDA);  // Zero divisor
    
    auto result = SafeOps::divide(a, b, 1e-6f);
    EXPECT_TRUE(result.is_valid());
    EXPECT_FALSE(result.has_inf());
    
    // Values should be large but not infinite
    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FALSE(std::isinf(val));
        EXPECT_GT(std::abs(val), 1e5f);  // Should be large
    }

    // Test safe log
    auto negative = Tensor::full({2, 2}, -5.0f, Device::CUDA);
    auto log_result = SafeOps::log(negative, 1e-6f);
    EXPECT_TRUE(log_result.is_valid());
    EXPECT_FALSE(log_result.has_nan());

    // Test safe sqrt
    auto neg_sqrt = Tensor::full({2, 2}, -4.0f, Device::CUDA);
    auto sqrt_result = SafeOps::sqrt(neg_sqrt, 0.0f);
    EXPECT_TRUE(sqrt_result.is_valid());
    EXPECT_FALSE(sqrt_result.has_nan());
}

TEST_F(TensorUtilsTest, MemoryInfo) {
    auto initial_info = MemoryInfo::cuda();
    
    // Allocate a large tensor
    const size_t large_size = 1024 * 1024;  // 1M elements = 4MB
    auto large_tensor = Tensor::zeros({large_size}, Device::CUDA);
    
    auto after_alloc_info = MemoryInfo::cuda();
    
    // Should have more allocated memory
    EXPECT_GT(after_alloc_info.allocated_bytes, initial_info.allocated_bytes);
    
    // Log the info (for debugging)
    initial_info.log();
    after_alloc_info.log();
}

TEST_F(TensorUtilsTest, LikeOperations) {
    auto original = Tensor::full({3, 4, 5}, 2.5f, Device::CUDA);
    
    auto zeros = zeros_like(original);
    EXPECT_EQ(zeros.shape(), original.shape());
    EXPECT_EQ(zeros.device(), original.device());
    EXPECT_FLOAT_EQ(zeros.sum(), 0.0f);
    
    auto ones = ones_like(original);
    EXPECT_EQ(ones.shape(), original.shape());
    EXPECT_EQ(ones.device(), original.device());
    EXPECT_FLOAT_EQ(ones.sum(), 60.0f);  // 3*4*5 = 60
}
