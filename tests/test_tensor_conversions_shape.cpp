/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core/tensor.hpp"

using namespace gs;

class TensorConversionsShapesTest : public ::testing::Test {
protected:
    void SetUp() override {
        tensor::manual_seed(42);
    }
};

// ============= DataType Conversion Tests =============

TEST_F(TensorConversionsShapesTest, Float32ToInt32) {
    auto t = Tensor::from_vector(std::vector<float>{1.5f, 2.7f, 3.2f, 4.9f}, {4}, Device::CPU);
    auto result = t.to(DataType::Int32);

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.dtype(), DataType::Int32);
    
    auto values = result.to_vector_int();
    EXPECT_EQ(values[0], 1);  // Truncated
    EXPECT_EQ(values[1], 2);
    EXPECT_EQ(values[2], 3);
    EXPECT_EQ(values[3], 4);
}

TEST_F(TensorConversionsShapesTest, Int32ToFloat32) {
    auto t = Tensor::from_vector(std::vector<int>{1, 2, 3, 4, 5}, {5}, Device::CPU);
    auto result = t.to(DataType::Float32);

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.dtype(), DataType::Float32);
    
    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[1], 2.0f);
    EXPECT_FLOAT_EQ(values[2], 3.0f);
    EXPECT_FLOAT_EQ(values[3], 4.0f);
    EXPECT_FLOAT_EQ(values[4], 5.0f);
}

TEST_F(TensorConversionsShapesTest, Float32ToBool) {
    auto t = Tensor::from_vector(std::vector<float>{0.0f, 1.0f, 2.0f, 0.0f, -1.0f}, {5}, Device::CPU);
    auto result = t.to(DataType::Bool);

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.dtype(), DataType::Bool);
    
    auto values = result.to_vector_bool();
    std::vector<bool> expected = {false, true, true, false, true};
    EXPECT_EQ(values.size(), expected.size());
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(values[i], expected[i]);
    }
}

TEST_F(TensorConversionsShapesTest, BoolToFloat32) {
    std::vector<bool> data = {true, false, true, false};
    auto t = Tensor::from_vector(data, {4}, Device::CPU);
    auto result = t.to(DataType::Float32);

    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.dtype(), DataType::Float32);
    
    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[1], 0.0f);
    EXPECT_FLOAT_EQ(values[2], 1.0f);
    EXPECT_FLOAT_EQ(values[3], 0.0f);
}

TEST_F(TensorConversionsShapesTest, ConversionCUDA) {
    auto t = Tensor::from_vector(std::vector<float>{1.5f, 2.7f, 3.2f}, {3}, Device::CUDA);
    auto result = t.to(DataType::Int32);

    EXPECT_EQ(result.device(), Device::CUDA);
    EXPECT_EQ(result.dtype(), DataType::Int32);
    
    auto values = result.to_vector_int();
    EXPECT_EQ(values[0], 1);
    EXPECT_EQ(values[1], 2);
    EXPECT_EQ(values[2], 3);
}

TEST_F(TensorConversionsShapesTest, ConversionPreservesShape) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, Device::CPU);
    auto result = t.to(DataType::Int32);

    EXPECT_EQ(result.shape(), t.shape());
}

TEST_F(TensorConversionsShapesTest, ConversionIdempotent) {
    auto t = Tensor::ones({5}, Device::CPU, DataType::Float32);
    auto result = t.to(DataType::Float32);

    // Should create a copy
    EXPECT_EQ(result.dtype(), DataType::Float32);
    
    // Modifying result shouldn't affect original
    result.fill_(42.0f);
    EXPECT_FLOAT_EQ(t.item(), 1.0f);
}

TEST_F(TensorConversionsShapesTest, Int32ToBoolConversion) {
    auto t = Tensor::from_vector(std::vector<int>{0, 1, 2, 0, -1}, {5}, Device::CPU);
    
    // First convert to float, then to bool
    auto float_t = t.to(DataType::Float32);
    auto result = float_t.to(DataType::Bool);

    auto values = result.to_vector_bool();
    std::vector<bool> expected = {false, true, true, false, true};
    EXPECT_EQ(values.size(), expected.size());
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(values[i], expected[i]);
    }
}

// ============= Shape Edge Cases =============

TEST_F(TensorConversionsShapesTest, SqueezeNegativeDim) {
    auto t = Tensor::zeros({2, 1, 3, 1, 4}, Device::CPU);
    
    // Squeeze last dimension (index -1)
    auto result = t.squeeze(-1);
    EXPECT_EQ(result.shape(), TensorShape({2, 1, 3, 4}));
    
    // Squeeze second-to-last dimension (index -2)
    auto result2 = t.squeeze(-2);
    EXPECT_EQ(result2.shape(), TensorShape({2, 1, 3, 4}));
}

TEST_F(TensorConversionsShapesTest, SqueezeAllOnes) {
    auto t = Tensor::zeros({1, 1, 1}, Device::CPU);
    auto result = t.squeeze();

    EXPECT_EQ(result.shape(), TensorShape({1}));  // Scalar-like
}

TEST_F(TensorConversionsShapesTest, SqueezeNoDims) {
    auto t = Tensor::zeros({2, 3, 4}, Device::CPU);
    auto result = t.squeeze();

    EXPECT_EQ(result.shape(), t.shape());  // No change
}

TEST_F(TensorConversionsShapesTest, SqueezeSpecificDimNotOne) {
    auto t = Tensor::zeros({2, 3, 4}, Device::CPU);
    auto result = t.squeeze(1);  // Dim 1 has size 3, not 1

    // Should not squeeze
    EXPECT_EQ(result.shape(), t.shape());
}

TEST_F(TensorConversionsShapesTest, FlattenDefault) {
    auto t = Tensor::arange(0.0f, 24.0f).reshape({2, 3, 4});
    auto result = t.flatten();

    EXPECT_EQ(result.shape(), TensorShape({24}));
    
    // Verify data is contiguous
    auto values = result.to_vector();
    for (size_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(values[i], static_cast<float>(i));
    }
}

TEST_F(TensorConversionsShapesTest, FlattenPartial) {
    auto t = Tensor::zeros({2, 3, 4, 5}, Device::CPU);
    auto result = t.flatten(1, 2);  // Flatten dims 1 and 2

    // Shape should be [2, 12, 5]
    EXPECT_EQ(result.shape(), TensorShape({2, 12, 5}));
}

TEST_F(TensorConversionsShapesTest, FlattenNegativeDims) {
    auto t = Tensor::zeros({2, 3, 4}, Device::CPU);
    auto result = t.flatten(-2, -1);  // Flatten last two dims

    EXPECT_EQ(result.shape(), TensorShape({2, 12}));
}

TEST_F(TensorConversionsShapesTest, FlattenSingleDim) {
    auto t = Tensor::zeros({10}, Device::CPU);
    auto result = t.flatten();

    EXPECT_EQ(result.shape(), TensorShape({10}));  // Already flat
}

// ============= BoundaryMode Edge Cases =============

TEST_F(TensorConversionsShapesTest, IndexSelectAssertModeInBounds) {
    auto t = Tensor::arange(0.0f, 5.0f);
    auto indices = Tensor::from_vector(std::vector<int>{0, 2, 4}, {3}, Device::CPU);
    
    auto result = t.index_select(0, indices, BoundaryMode::Assert);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 0.0f);
    EXPECT_FLOAT_EQ(values[1], 2.0f);
    EXPECT_FLOAT_EQ(values[2], 4.0f);
}

TEST_F(TensorConversionsShapesTest, IndexSelectClampAllOutOfBounds) {
    auto t = Tensor::arange(0.0f, 5.0f);
    auto indices = Tensor::from_vector(std::vector<int>{-10, 100, 200}, {3}, Device::CPU);
    
    auto result = t.index_select(0, indices, BoundaryMode::Clamp);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 0.0f);  // Clamped to min
    EXPECT_FLOAT_EQ(values[1], 4.0f);  // Clamped to max
    EXPECT_FLOAT_EQ(values[2], 4.0f);  // Clamped to max
}

TEST_F(TensorConversionsShapesTest, IndexSelectWrapNegative) {
    auto t = Tensor::arange(0.0f, 5.0f);
    auto indices = Tensor::from_vector(std::vector<int>{-1, -2, -3}, {3}, Device::CPU);
    
    auto result = t.index_select(0, indices, BoundaryMode::Wrap);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 4.0f);  // -1 wraps to 4
    EXPECT_FLOAT_EQ(values[1], 3.0f);  // -2 wraps to 3
    EXPECT_FLOAT_EQ(values[2], 2.0f);  // -3 wraps to 2
}

TEST_F(TensorConversionsShapesTest, GatherDefaultBoundary) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CPU);
    auto indices = Tensor::from_vector(std::vector<int>{0, 1, 2, 3}, {4}, Device::CPU);
    
    // Default mode (Assert)
    auto result = t.gather(0, indices);

    auto values = result.to_vector();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(values[i], static_cast<float>(i + 1));
    }
}

// ============= from_blob Tests =============

TEST_F(TensorConversionsShapesTest, FromBlobOwnership) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    
    auto t = Tensor::from_blob(data.data(), {4}, Device::CPU, DataType::Float32);

    EXPECT_FALSE(t.owns_memory());  // Should be a view
    EXPECT_EQ(t.numel(), 4);
    
    // Verify data is accessible
    EXPECT_FLOAT_EQ(t.at({0}), 1.0f);
}

TEST_F(TensorConversionsShapesTest, FromBlobModification) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    auto t = Tensor::from_blob(data.data(), {3}, Device::CPU, DataType::Float32);

    // Modify through tensor
    t.at({0}) = 10.0f;

    // Should reflect in original data
    EXPECT_FLOAT_EQ(data[0], 10.0f);
}

TEST_F(TensorConversionsShapesTest, FromBlobMultiDimensional) {
    std::vector<float> data(12);
    for (size_t i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto t = Tensor::from_blob(data.data(), {3, 4}, Device::CPU, DataType::Float32);

    EXPECT_EQ(t.shape(), TensorShape({3, 4}));
    EXPECT_FLOAT_EQ(t.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(t.at({2, 3}), 11.0f);
}

// ============= clone() Tests =============

TEST_F(TensorConversionsShapesTest, CloneAllocatesNew) {
    auto t = Tensor::ones({5}, Device::CPU);
    auto cloned = t.clone();

    // Cloned should have separate memory
    cloned.fill_(42.0f);
    
    EXPECT_FLOAT_EQ(t.item(), 1.0f);
    EXPECT_FLOAT_EQ(cloned.item(), 42.0f);
}

TEST_F(TensorConversionsShapesTest, ClonePreservesProperties) {
    auto t = Tensor::ones({3, 4}, Device::CPU, DataType::Int32);
    auto cloned = t.clone();

    EXPECT_EQ(cloned.shape(), t.shape());
    EXPECT_EQ(cloned.device(), t.device());
    EXPECT_EQ(cloned.dtype(), t.dtype());
    EXPECT_TRUE(cloned.owns_memory());
}

TEST_F(TensorConversionsShapesTest, CloneCUDA) {
    auto t = Tensor::randn({10}, Device::CUDA);
    auto cloned = t.clone();

    EXPECT_EQ(cloned.device(), Device::CUDA);
    EXPECT_TRUE(cloned.owns_memory());
    
    // Data should match
    EXPECT_TRUE(t.all_close(cloned));
}

// ============= cat() Static Tests =============

TEST_F(TensorConversionsShapesTest, CatBasic) {
    auto a = Tensor::ones({3, 4}, Device::CPU);
    auto b = Tensor::ones({2, 4}, Device::CPU).mul(2.0f);
    
    std::vector<Tensor> tensors;
    tensors.push_back(a.clone());
    tensors.push_back(b.clone());
    
    auto result = Tensor::cat(tensors, 0);

    EXPECT_EQ(result.shape(), TensorShape({5, 4}));
    
    // First 3 rows should be 1, next 2 rows should be 2
    auto acc = result.accessor<float, 2>();
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(acc(i, 0), 1.0f);
    }
    for (size_t i = 3; i < 5; ++i) {
        EXPECT_FLOAT_EQ(acc(i, 0), 2.0f);
    }
}

TEST_F(TensorConversionsShapesTest, CatMultiple) {
    std::vector<Tensor> tensors;
    for (int i = 0; i < 5; ++i) {
        tensors.push_back(Tensor::full({2, 3}, static_cast<float>(i), Device::CPU));
    }
    
    auto result = Tensor::cat(tensors, 0);

    EXPECT_EQ(result.shape(), TensorShape({10, 3}));
}

TEST_F(TensorConversionsShapesTest, CatSingleTensor) {
    auto t = Tensor::ones({3, 4}, Device::CPU);
    std::vector<Tensor> tensors;
    tensors.push_back(t.clone());
    
    auto result = Tensor::cat(tensors, 0);

    EXPECT_EQ(result.shape(), t.shape());
}

TEST_F(TensorConversionsShapesTest, CatEmpty) {
    std::vector<Tensor> tensors;
    auto result = Tensor::cat(tensors, 0);

    EXPECT_FALSE(result.is_valid());
}

// ============= Reshape Edge Cases =============

TEST_F(TensorConversionsShapesTest, ReshapeWithInference) {
    auto t = Tensor::arange(0.0f, 24.0f);
    auto result = t.reshape({-1, 6});  // Infer first dimension

    EXPECT_EQ(result.shape(), TensorShape({4, 6}));
}

TEST_F(TensorConversionsShapesTest, ReshapeMultipleInferences) {
    auto t = Tensor::arange(0.0f, 24.0f);
    auto result = t.reshape({-1, -1});  // Multiple -1s should fail

    EXPECT_FALSE(result.is_valid());
}

TEST_F(TensorConversionsShapesTest, ReshapeInvalidSize) {
    auto t = Tensor::arange(0.0f, 24.0f);
    auto result = t.reshape({5, 5});  // 25 != 24

    EXPECT_FALSE(result.is_valid());
}

TEST_F(TensorConversionsShapesTest, ReshapeToScalar) {
    auto t = Tensor::from_vector(std::vector<float>{42.0f}, {1}, Device::CPU);
    auto result = t.reshape({1});

    EXPECT_EQ(result.numel(), 1);
    EXPECT_FLOAT_EQ(result.item(), 42.0f);
}

// ============= Integration Tests =============

TEST_F(TensorConversionsShapesTest, ConversionChain) {
    auto t = Tensor::from_vector(std::vector<float>{1.5f, 2.7f, 3.2f}, {3}, Device::CPU);
    
    auto result = t.to(DataType::Int32)
                   .to(DataType::Float32)
                   .cuda()
                   .cpu();

    EXPECT_EQ(result.dtype(), DataType::Float32);
    EXPECT_EQ(result.device(), Device::CPU);
    
    // Should have lost fractional parts from int conversion
    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[1], 2.0f);
    EXPECT_FLOAT_EQ(values[2], 3.0f);
}

TEST_F(TensorConversionsShapesTest, ShapeManipulationChain) {
    auto t = Tensor::arange(0.0f, 24.0f);
    
    auto result = t.reshape({2, 3, 4})
                   .squeeze()
                   .flatten(0, 1)
                   .unsqueeze(0);

    // Should end up with shape [1, 6, 4]
    EXPECT_EQ(result.shape(), TensorShape({1, 6, 4}));
}
