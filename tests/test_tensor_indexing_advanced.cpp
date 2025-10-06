/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core/tensor.hpp"

using namespace gs;

class TensorIndexingAdvancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        tensor::manual_seed(42);
    }
};

// ============= index_add_ Tests =============

TEST_F(TensorIndexingAdvancedTest, IndexAddBasic) {
    auto t = Tensor::zeros({5}, Device::CPU);
    auto indices = Tensor::from_vector(std::vector<int>{0, 2, 4}, {3}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::CPU);

    t.index_add_(0, indices, values);

    auto result = t.to_vector();
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 0.0f);
    EXPECT_FLOAT_EQ(result[2], 2.0f);
    EXPECT_FLOAT_EQ(result[3], 0.0f);
    EXPECT_FLOAT_EQ(result[4], 3.0f);
}

TEST_F(TensorIndexingAdvancedTest, IndexAddAccumulate) {
    auto t = Tensor::ones({5}, Device::CPU);
    auto indices = Tensor::from_vector(std::vector<int>{0, 0, 1}, {3}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{10.0f, 20.0f, 30.0f}, {3}, Device::CPU);

    t.index_add_(0, indices, values);

    auto result = t.to_vector();
    EXPECT_FLOAT_EQ(result[0], 31.0f);  // 1 + 10 + 20
    EXPECT_FLOAT_EQ(result[1], 31.0f);  // 1 + 30
    EXPECT_FLOAT_EQ(result[2], 1.0f);   // unchanged
}

TEST_F(TensorIndexingAdvancedTest, IndexAdd2D) {
    auto t = Tensor::zeros({3, 4}, Device::CPU);
    auto indices = Tensor::from_vector(std::vector<int>{0, 2}, {2}, Device::CPU);
    auto values = Tensor::ones({2, 4}, Device::CPU);

    t.index_add_(0, indices, values);

    // Row 0 and row 2 should be 1, row 1 should be 0
    auto acc = t.accessor<float, 2>();
    for (size_t j = 0; j < 4; ++j) {
        EXPECT_FLOAT_EQ(acc(0, j), 1.0f);
        EXPECT_FLOAT_EQ(acc(1, j), 0.0f);
        EXPECT_FLOAT_EQ(acc(2, j), 1.0f);
    }
}

TEST_F(TensorIndexingAdvancedTest, IndexAddCUDA) {
    auto t = Tensor::zeros({5}, Device::CUDA);
    auto indices = Tensor::from_vector(std::vector<int>{0, 2, 4}, {3}, Device::CUDA);
    auto values = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::CUDA);

    t.index_add_(0, indices, values);

    auto result = t.to_vector();
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[2], 2.0f);
    EXPECT_FLOAT_EQ(result[4], 3.0f);
}

// ============= index_put_ with Advanced Indexing =============

TEST_F(TensorIndexingAdvancedTest, IndexPutSingle) {
    auto t = Tensor::zeros({5}, Device::CPU);
    auto indices = Tensor::from_vector(std::vector<int>{1, 3}, {2}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{10.0f, 20.0f}, {2}, Device::CPU);

    t.index_put_(indices, values);

    auto result = t.to_vector();
    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 10.0f);
    EXPECT_FLOAT_EQ(result[2], 0.0f);
    EXPECT_FLOAT_EQ(result[3], 20.0f);
    EXPECT_FLOAT_EQ(result[4], 0.0f);
}

TEST_F(TensorIndexingAdvancedTest, IndexPutNegativeIndices) {
    auto t = Tensor::zeros({5}, Device::CPU);
    auto indices = Tensor::from_vector(std::vector<int>{-1, -2}, {2}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{100.0f, 200.0f}, {2}, Device::CPU);

    t.index_put_(indices, values);

    auto result = t.to_vector();
    EXPECT_FLOAT_EQ(result[3], 200.0f);  // -2 -> index 3
    EXPECT_FLOAT_EQ(result[4], 100.0f);  // -1 -> index 4
}

TEST_F(TensorIndexingAdvancedTest, IndexPutMultiDimensional) {
    auto t = Tensor::zeros({3, 4}, Device::CPU);
    auto indices = Tensor::from_vector(std::vector<int>{0, 5, 11}, {3}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::CPU);

    // Treat as flattened
    t.index_put_(indices, values);

    auto flat = t.flatten();
    auto result = flat.to_vector();
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[5], 2.0f);
    EXPECT_FLOAT_EQ(result[11], 3.0f);
}

TEST_F(TensorIndexingAdvancedTest, IndexPutVectorOfTensors) {
    auto t = Tensor::zeros({3, 3}, Device::CPU);
    
    // Index with [row_indices, col_indices]
    auto row_idx = Tensor::from_vector(std::vector<int>{0, 1, 2}, {3}, Device::CPU);
    auto col_idx = Tensor::from_vector(std::vector<int>{0, 1, 2}, {3}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::CPU);

    // Can't use initializer list with non-copyable Tensor, use push_back instead
    std::vector<Tensor> indices;
    indices.push_back(std::move(row_idx));
    indices.push_back(std::move(col_idx));
    t.index_put_(indices, values);

    // Should set diagonal
    auto acc = t.accessor<float, 2>();
    EXPECT_FLOAT_EQ(acc(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(acc(1, 1), 2.0f);
    EXPECT_FLOAT_EQ(acc(2, 2), 3.0f);
    EXPECT_FLOAT_EQ(acc(0, 1), 0.0f);  // Off-diagonal should be 0
}

TEST_F(TensorIndexingAdvancedTest, IndexPutOutOfBounds) {
    auto t = Tensor::zeros({5}, Device::CPU);
    auto indices = Tensor::from_vector(std::vector<int>{0, 10, 2}, {3}, Device::CPU);  // 10 is out of bounds
    auto values = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::CPU);

    // Should handle gracefully (skip out-of-bounds or clamp)
    t.index_put_(indices, values);

    auto result = t.to_vector();
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);
    // Index 10 should be ignored or handled safely
}

// ============= Boundary Mode Tests =============

TEST_F(TensorIndexingAdvancedTest, IndexSelectWithAssertMode) {
    auto t = Tensor::arange(0.0f, 10.0f);
    auto indices = Tensor::from_vector(std::vector<int>{0, 2, 4, 6, 8}, {5}, Device::CPU);
    
    auto result = t.index_select(0, indices, BoundaryMode::Assert);

    auto values = result.to_vector();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(values[i], indices.to_vector_int()[i]);
    }
}

TEST_F(TensorIndexingAdvancedTest, IndexSelectWithClampMode) {
    auto t = Tensor::arange(0.0f, 5.0f);
    auto indices = Tensor::from_vector(std::vector<int>{-1, 0, 3, 10}, {4}, Device::CPU);
    
    auto result = t.index_select(0, indices, BoundaryMode::Clamp);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 0.0f);  // -1 clamped to 0
    EXPECT_FLOAT_EQ(values[1], 0.0f);  // 0
    EXPECT_FLOAT_EQ(values[2], 3.0f);  // 3
    EXPECT_FLOAT_EQ(values[3], 4.0f);  // 10 clamped to 4
}

TEST_F(TensorIndexingAdvancedTest, IndexSelectWithWrapMode) {
    auto t = Tensor::arange(0.0f, 5.0f);
    auto indices = Tensor::from_vector(std::vector<int>{-1, 0, 5, 7}, {4}, Device::CPU);
    
    auto result = t.index_select(0, indices, BoundaryMode::Wrap);

    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 4.0f);  // -1 wraps to 4
    EXPECT_FLOAT_EQ(values[1], 0.0f);  // 0
    EXPECT_FLOAT_EQ(values[2], 0.0f);  // 5 wraps to 0
    EXPECT_FLOAT_EQ(values[3], 2.0f);  // 7 wraps to 2
}

TEST_F(TensorIndexingAdvancedTest, GatherWithClampMode) {
    auto t = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        {2, 3},
        Device::CPU
    );
    auto indices = Tensor::from_vector(std::vector<int>{-1, 1, 10}, {3}, Device::CPU);
    
    auto result = t.gather(1, indices, BoundaryMode::Clamp);

    // Should clamp: -1->0, 1->1, 10->2
    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 1.0f);  // t[0,0]
    EXPECT_FLOAT_EQ(values[1], 2.0f);  // t[0,1]
    EXPECT_FLOAT_EQ(values[2], 3.0f);  // t[0,2]
}

TEST_F(TensorIndexingAdvancedTest, GatherWithWrapMode) {
    auto t = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        {2, 3},
        Device::CPU
    );
    auto indices = Tensor::from_vector(std::vector<int>{-1, 0, 3}, {3}, Device::CPU);
    
    auto result = t.gather(1, indices, BoundaryMode::Wrap);

    // Should wrap: -1->2, 0->0, 3->0
    auto values = result.to_vector();
    EXPECT_FLOAT_EQ(values[0], 3.0f);  // t[0,2]
    EXPECT_FLOAT_EQ(values[1], 1.0f);  // t[0,0]
    EXPECT_FLOAT_EQ(values[2], 1.0f);  // t[0,0]
}

// ============= nonzero Tests =============

TEST_F(TensorIndexingAdvancedTest, NonzeroBasic) {
    auto t = Tensor::from_vector(std::vector<float>{0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f}, {6}, Device::CPU);
    auto indices = t.nonzero();

    ASSERT_TRUE(indices.is_valid());
    EXPECT_EQ(indices.dtype(), DataType::Int64);
    EXPECT_EQ(indices.numel(), 3);

    // Should be indices 1, 3, 5
    auto values = indices.to_vector();
    // Cast to int64_t for comparison
    int64_t* idx_ptr = reinterpret_cast<int64_t*>(const_cast<float*>(values.data()));
    
    // Note: This is a bit hacky - better to have to_vector_int64()
    // For now, just verify count
    EXPECT_EQ(indices.numel(), 3);
}

TEST_F(TensorIndexingAdvancedTest, NonzeroBool) {
    std::vector<bool> data = {true, false, true, false, true};
    auto t = Tensor::from_vector(data, {5}, Device::CPU);
    auto indices = t.nonzero();

    EXPECT_EQ(indices.numel(), 3);
}

TEST_F(TensorIndexingAdvancedTest, NonzeroAllZeros) {
    auto t = Tensor::zeros({10}, Device::CPU);
    auto indices = t.nonzero();

    EXPECT_EQ(indices.numel(), 0);
}

TEST_F(TensorIndexingAdvancedTest, NonzeroAllOnes) {
    auto t = Tensor::ones({10}, Device::CPU);
    auto indices = t.nonzero();

    EXPECT_EQ(indices.numel(), 10);
}

TEST_F(TensorIndexingAdvancedTest, NonzeroCUDA) {
    auto t = Tensor::from_vector(std::vector<float>{0.0f, 1.0f, 0.0f, 2.0f}, {4}, Device::CUDA);
    auto indices = t.nonzero();

    EXPECT_EQ(indices.device(), Device::CUDA);
    EXPECT_EQ(indices.numel(), 2);
}

// ============= Pythonic Indexing [] Operator Tests =============

TEST_F(TensorIndexingAdvancedTest, IndexingWithTensor) {
    auto t = Tensor::arange(0.0f, 10.0f);
    auto indices = Tensor::from_vector(std::vector<int>{0, 2, 4, 6, 8}, {5}, Device::CPU);
    
    auto result = t[indices];
    Tensor selected = result;  // Convert proxy to tensor

    auto values = selected.to_vector();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(values[i], i * 2.0f);
    }
}

TEST_F(TensorIndexingAdvancedTest, IndexingWithBoolMask) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5}, Device::CPU);
    std::vector<bool> mask_data = {true, false, true, false, true};
    auto mask = Tensor::from_vector(mask_data, {5}, Device::CPU);
    
    auto result = t[mask];
    Tensor selected = result;

    auto values = selected.to_vector();
    EXPECT_EQ(values.size(), 3);
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[1], 3.0f);
    EXPECT_FLOAT_EQ(values[2], 5.0f);
}

TEST_F(TensorIndexingAdvancedTest, IndexingAssignment) {
    auto t = Tensor::zeros({5}, Device::CPU);
    auto indices = Tensor::from_vector(std::vector<int>{0, 2, 4}, {3}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{10.0f, 20.0f, 30.0f}, {3}, Device::CPU);
    
    t[indices] = values;

    auto result = t.to_vector();
    EXPECT_FLOAT_EQ(result[0], 10.0f);
    EXPECT_FLOAT_EQ(result[2], 20.0f);
    EXPECT_FLOAT_EQ(result[4], 30.0f);
}

TEST_F(TensorIndexingAdvancedTest, MaskedAssignment) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5}, Device::CPU);
    std::vector<bool> mask_data = {true, false, true, false, true};
    auto mask = Tensor::from_vector(mask_data, {5}, Device::CPU);
    
    t[mask] = 100.0f;

    auto result = t.to_vector();
    EXPECT_FLOAT_EQ(result[0], 100.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
    EXPECT_FLOAT_EQ(result[2], 100.0f);
    EXPECT_FLOAT_EQ(result[3], 4.0f);
    EXPECT_FLOAT_EQ(result[4], 100.0f);
}

// ============= Integration Tests =============

TEST_F(TensorIndexingAdvancedTest, ComplexIndexingChain) {
    auto t = Tensor::arange(0.0f, 100.0f).reshape({10, 10});
    
    // Select specific rows
    auto row_idx = Tensor::from_vector(std::vector<int>{0, 5, 9}, {3}, Device::CPU);
    auto rows = t.index_select(0, row_idx);
    
    // Then select specific columns from those rows
    auto col_idx = Tensor::from_vector(std::vector<int>{0, 5, 9}, {3}, Device::CPU);
    auto result = rows.index_select(1, col_idx);

    EXPECT_EQ(result.shape(), TensorShape({3, 3}));
    
    // Check corner values
    EXPECT_FLOAT_EQ(result.at({0, 0}), 0.0f);    // (0, 0)
    EXPECT_FLOAT_EQ(result.at({0, 2}), 9.0f);    // (0, 9)
    EXPECT_FLOAT_EQ(result.at({2, 2}), 99.0f);   // (9, 9)
}

TEST_F(TensorIndexingAdvancedTest, IndexingWithArithmetic) {
    auto t = Tensor::arange(0.0f, 10.0f);
    auto indices = Tensor::from_vector(std::vector<int>{1, 3, 5, 7, 9}, {5}, Device::CPU);
    
    auto selected = t.index_select(0, indices);
    auto doubled = selected.mul(2.0f);

    auto values = doubled.to_vector();
    EXPECT_FLOAT_EQ(values[0], 2.0f);   // 1*2
    EXPECT_FLOAT_EQ(values[1], 6.0f);   // 3*2
    EXPECT_FLOAT_EQ(values[2], 10.0f);  // 5*2
}

TEST_F(TensorIndexingAdvancedTest, ScatterGatherRoundtrip) {
    auto original = Tensor::arange(0.0f, 10.0f);
    auto indices = Tensor::from_vector(std::vector<int>{0, 2, 4, 6, 8}, {5}, Device::CPU);
    
    // Gather
    auto gathered = original.gather(0, indices);
    
    // Scatter back
    auto result = Tensor::zeros({10}, Device::CPU);
    result.scatter_(0, indices, gathered);
    
    // Check that scattered values match original at those indices
    auto result_vals = result.to_vector();
    for (int i : {0, 2, 4, 6, 8}) {
        EXPECT_FLOAT_EQ(result_vals[i], static_cast<float>(i));
    }
}
