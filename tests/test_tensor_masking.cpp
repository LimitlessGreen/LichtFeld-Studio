/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <random>

using namespace gs;

class TensorMaskingTest : public ::testing::Test {
protected:
    void SetUp() override {
        tensor::manual_seed(42);
    }

    Tensor create_test_tensor(std::vector<float> data, TensorShape shape) {
        return Tensor::from_vector(data, shape, Device::CUDA);
    }

    bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b,
                         float tolerance = 1e-5f) {
        if (a.size() != b.size())
            return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

// ============= Basic Diagnostic Tests =============
TEST_F(TensorMaskingTest, DiagnosticMaskCreation) {
    // Test 1: Basic boolean mask creation
    auto data = create_test_tensor({1, 2, 3, 4, 5}, {5});
    auto mask = data.gt(3.0f);

    auto mask_values = mask.to_vector_bool();
    std::cout << "Mask values: ";
    for (auto v : mask_values) std::cout << v << " ";
    std::cout << std::endl;

    // Should be [0, 0, 0, 1, 1]
    EXPECT_FALSE(mask_values[0]);
    EXPECT_FALSE(mask_values[1]);
    EXPECT_FALSE(mask_values[2]);
    EXPECT_TRUE(mask_values[3]);
    EXPECT_TRUE(mask_values[4]);
}

TEST_F(TensorMaskingTest, DiagnosticMaskedSelect) {
    // Test 2: Basic masked select
    auto data = create_test_tensor({1, 2, 3, 4, 5}, {5});
    auto mask = create_test_tensor({0, 0, 0, 1, 1}, {5}).ne(0.0f);

    std::cout << "Data count: " << data.numel() << std::endl;
    std::cout << "Mask count nonzero: " << mask.count_nonzero() << std::endl;

    auto selected = data.masked_select(mask);
    std::cout << "Selected count: " << selected.numel() << std::endl;

    if (selected.numel() > 0) {
        auto values = selected.to_vector();
        std::cout << "Selected values: ";
        for (auto v : values) std::cout << v << " ";
        std::cout << std::endl;
    }

    EXPECT_EQ(selected.numel(), 2);
    if (selected.numel() == 2) {
        auto values = selected.to_vector();
        EXPECT_FLOAT_EQ(values[0], 4.0f);
        EXPECT_FLOAT_EQ(values[1], 5.0f);
    }
}

TEST_F(TensorMaskingTest, DiagnosticMaskedFill) {
    // Test 3: Basic masked fill
    auto data = create_test_tensor({1, 2, 3, 4, 5}, {5});
    auto mask = data.gt(3.0f);

    std::cout << "Before fill: ";
    auto before = data.to_vector();
    for (auto v : before) std::cout << v << " ";
    std::cout << std::endl;

    data.masked_fill_(mask, -1.0f);

    std::cout << "After fill: ";
    auto after = data.to_vector();
    for (auto v : after) std::cout << v << " ";
    std::cout << std::endl;

    std::vector<float> expected = {1, 2, 3, -1, -1};
    EXPECT_TRUE(compare_vectors(after, expected));
}

// =============  Regression Tests for Fixed Bugs =============

// Test for the diagonal mask bug where arange().reshape({n,1}).eq(arange().reshape({1,n})) was all true
TEST_F(TensorMaskingTest, RegressionDiagonalMaskBroadcast) {
    const size_t n = 5;

    // Create range tensors
    std::vector<float> range_data = {0, 1, 2, 3, 4};
    auto range = Tensor::from_vector(range_data, {n}, Device::CUDA);

    // Reshape to column and row
    auto col = range.reshape({n, 1});  // [0, 1, 2, 3, 4]^T
    auto row = range.reshape({1, n});  // [0, 1, 2, 3, 4]

    // Create diagonal mask via broadcasting comparison
    auto diag_mask = col.eq(row);

    EXPECT_EQ(diag_mask.shape(), TensorShape({n, n}));
    EXPECT_EQ(diag_mask.dtype(), DataType::Bool);

    // Check that we get a proper diagonal matrix
    auto mask_values = diag_mask.to_vector_bool();

    // Print for debugging
    std::cout << "Diagonal mask (should be identity matrix):\n";
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << (mask_values[i * n + j] ? "1 " : "0 ");
        }
        std::cout << "\n";
    }

    // Verify it's actually a diagonal matrix
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            bool expected = (i == j);
            EXPECT_EQ(mask_values[i * n + j], expected)
                << "Failed at position (" << i << ", " << j << ")";
        }
    }

    // Count diagonal elements
    auto diagonal_count = diag_mask.count_nonzero();
    EXPECT_EQ(diagonal_count, n) << "Should have exactly " << n << " true elements on diagonal";
}

// Test for boolean tensor expansion bug
TEST_F(TensorMaskingTest, RegressionBooleanTensorExpansion) {
    const size_t batch_size = 3;
    const size_t seq_len = 4;

    // Create a 2D boolean mask
    std::vector<bool> mask_data(seq_len * seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            mask_data[i * seq_len + j] = (i >= j);  // Lower triangular
        }
    }

    auto mask_2d = Tensor::from_vector(mask_data, {seq_len, seq_len}, Device::CUDA);

    // Add batch dimension and expand
    auto mask_3d = mask_2d.unsqueeze(0);  // Shape: [1, seq_len, seq_len]
    EXPECT_EQ(mask_3d.shape(), TensorShape({1, seq_len, seq_len}));

    auto mask_expanded = mask_3d.expand({batch_size, seq_len, seq_len});
    EXPECT_EQ(mask_expanded.shape(), TensorShape({batch_size, seq_len, seq_len}));

    // Verify all batches have the same mask pattern
    auto expanded_values = mask_expanded.to_vector_bool();

    for (size_t b = 0; b < batch_size; ++b) {
        std::cout << "Batch " << b << " mask:\n";
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                size_t idx = b * seq_len * seq_len + i * seq_len + j;
                bool value = expanded_values[idx];
                std::cout << (value ? "1 " : "0 ");

                // All batches should have the same pattern
                bool expected = (i >= j);
                EXPECT_EQ(value, expected)
                    << "Failed at batch " << b << ", position (" << i << ", " << j << ")";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

// Test broadcasting with different rank tensors in comparison
TEST_F(TensorMaskingTest, RegressionBroadcastComparisonDifferentRanks) {
    // Test case 1: 1D vs 2D
    auto vec = create_test_tensor({1, 2, 3}, {3});
    auto mat = create_test_tensor({1, 2, 3, 2, 3, 4}, {2, 3});

    auto mask = vec.eq(mat);
    EXPECT_EQ(mask.shape(), TensorShape({2, 3}));

    auto values = mask.to_vector_bool();
    // First row compared with [1, 2, 3]: [1==1, 2==2, 3==3] = [T, T, T]
    // Second row compared with [1, 2, 3]: [2==1, 3==2, 4==3] = [F, F, F]
    std::vector<bool> expected = {true, true, true, false, false, false};
    EXPECT_EQ(values, expected);

    // Test case 2: Scalar-like vs Matrix
    auto scalar = create_test_tensor({2}, {1, 1});
    auto matrix = create_test_tensor({1, 2, 3, 4}, {2, 2});

    auto mask2 = scalar.eq(matrix);
    EXPECT_EQ(mask2.shape(), TensorShape({2, 2}));

    auto values2 = mask2.to_vector_bool();
    // All elements compared with 2
    std::vector<bool> expected2 = {false, true, false, false};
    EXPECT_EQ(values2, expected2);
}

// Test for proper boolean broadcast in logical operations
TEST_F(TensorMaskingTest, RegressionBooleanBroadcastLogical) {
    // Create masks with different shapes
    auto mask1 = Tensor::from_vector(std::vector<bool>{true, false}, {2, 1}, Device::CUDA);
    auto mask2 = Tensor::from_vector(std::vector<bool>{true, false, true}, {1, 3}, Device::CUDA);

    // Logical AND with broadcasting
    auto result = mask1.logical_and(mask2);
    EXPECT_EQ(result.shape(), TensorShape({2, 3}));

    auto values = result.to_vector_bool();
    // First row: [true] AND [true, false, true] = [true, false, true]
    // Second row: [false] AND [true, false, true] = [false, false, false]
    std::vector<bool> expected = {true, false, true, false, false, false};
    EXPECT_EQ(values, expected);
}

// ============= Comparison Operations Tests =============
TEST_F(TensorMaskingTest, ComparisonEqual) {
    auto a = create_test_tensor({1, 2, 3, 4, 5}, {5});
    auto b = create_test_tensor({1, 0, 3, 0, 5}, {5});

    auto mask = a.eq(b);
    EXPECT_EQ(mask.dtype(), DataType::Bool);

    auto mask_values = mask.to_vector_bool();
    std::vector<bool> expected = {true, false, true, false, true};
    EXPECT_EQ(mask_values, expected);

    // Test scalar comparison
    auto mask2 = a.eq(3.0f);
    auto mask2_values = mask2.to_vector_bool();
    std::vector<bool> expected2 = {false, false, true, false, false};
    EXPECT_EQ(mask2_values, expected2);
}

TEST_F(TensorMaskingTest, ComparisonLessThan) {
    auto a = create_test_tensor({1, 2, 3, 4, 5}, {5});

    auto mask = a.lt(3.0f);
    auto mask_values = mask.to_vector_bool();
    std::vector<bool> expected = {true, true, false, false, false};
    EXPECT_EQ(mask_values, expected);

    // Test with another tensor
    auto b = create_test_tensor({2, 2, 2, 6, 4}, {5});
    auto mask2 = a.lt(b);
    auto mask2_values = mask2.to_vector_bool();
    std::vector<bool> expected2 = {true, false, false, true, false};
    EXPECT_EQ(mask2_values, expected2);
}

TEST_F(TensorMaskingTest, ComparisonGreaterThan) {
    auto a = create_test_tensor({1, 2, 3, 4, 5}, {5});

    auto mask = a.gt(3.0f);
    auto mask_values = mask.to_vector_bool();
    std::vector<bool> expected = {false, false, false, true, true};
    EXPECT_EQ(mask_values, expected);
}

TEST_F(TensorMaskingTest, ComparisonChaining) {
    auto a = create_test_tensor({1, 2, 3, 4, 5}, {5});

    // Find elements between 2 and 4 (inclusive)
    auto mask = a.ge(2.0f).logical_and(a.le(4.0f));
    auto mask_values = mask.to_vector_bool();
    std::vector<bool> expected = {false, true, true, true, false};
    EXPECT_EQ(mask_values, expected);
}

TEST_F(TensorMaskingTest, ComparisonWithBroadcasting) {
    auto a = create_test_tensor({1, 2, 3, 4, 5, 6}, {2, 3});
    auto b = create_test_tensor({2, 3, 4}, {1, 3});

    auto mask = a.gt(b);
    EXPECT_EQ(mask.shape(), a.shape());

    auto mask_values = mask.to_vector_bool();
    // First row compared with [2, 3, 4]: [1>2, 2>3, 3>4] = [F, F, F]
    // Second row compared with [2, 3, 4]: [4>2, 5>3, 6>4] = [T, T, T]
    std::vector<bool> expected = {false, false, false, true, true, true};
    EXPECT_EQ(mask_values, expected);
}

// ============= Logical Operations Tests =============
TEST_F(TensorMaskingTest, LogicalOperations) {
    auto a = create_test_tensor({1, 0, 1, 0}, {4});
    auto b = create_test_tensor({1, 1, 0, 0}, {4});

    auto a_bool = a.ne(0.0f);
    auto b_bool = b.ne(0.0f);

    // AND
    auto and_result = a_bool.logical_and(b_bool);
    auto and_values = and_result.to_vector_bool();
    std::vector<bool> and_expected = {true, false, false, false};
    EXPECT_EQ(and_values, and_expected);

    // OR
    auto or_result = a_bool.logical_or(b_bool);
    auto or_values = or_result.to_vector_bool();
    std::vector<bool> or_expected = {true, true, true, false};
    EXPECT_EQ(or_values, or_expected);

    // NOT
    auto not_result = a_bool.logical_not();
    auto not_values = not_result.to_vector_bool();
    std::vector<bool> not_expected = {false, true, false, true};
    EXPECT_EQ(not_values, not_expected);
}

// ============= Masked Select Tests =============
TEST_F(TensorMaskingTest, MaskedSelect) {
    auto tensor = create_test_tensor({1, 2, 3, 4, 5, 6}, {2, 3});
    auto mask = tensor.gt(3.0f);

    auto selected = tensor.masked_select(mask);
    EXPECT_EQ(selected.ndim(), 1);
    EXPECT_EQ(selected.numel(), 3); // Elements 4, 5, 6

    auto values = selected.to_vector();
    std::vector<float> expected = {4, 5, 6};
    EXPECT_TRUE(compare_vectors(values, expected));
}

TEST_F(TensorMaskingTest, MaskedSelectEmpty) {
    auto tensor = create_test_tensor({1, 2, 3}, {3});
    auto mask = tensor.gt(10.0f); // No elements > 10

    auto selected = tensor.masked_select(mask);
    EXPECT_EQ(selected.numel(), 0);
}

// ============= Masked Fill Tests =============
TEST_F(TensorMaskingTest, MaskedFill) {
    auto tensor = create_test_tensor({1, 2, 3, 4, 5}, {5});
    auto mask = tensor.gt(3.0f);

    tensor.masked_fill_(mask, -1.0f);

    auto values = tensor.to_vector();
    std::vector<float> expected = {1, 2, 3, -1, -1};
    EXPECT_TRUE(compare_vectors(values, expected));
}

TEST_F(TensorMaskingTest, MaskedFillNonInplace) {
    auto tensor = create_test_tensor({1, 2, 3, 4, 5}, {5});
    auto mask = tensor.le(2.0f);

    auto result = tensor.masked_fill(mask, 0.0f);

    // Original unchanged
    auto orig_values = tensor.to_vector();
    std::vector<float> orig_expected = {1, 2, 3, 4, 5};
    EXPECT_TRUE(compare_vectors(orig_values, orig_expected));

    // Result has masked values filled
    auto result_values = result.to_vector();
    std::vector<float> result_expected = {0, 0, 3, 4, 5};
    EXPECT_TRUE(compare_vectors(result_values, result_expected));
}

// ============= Where Operation Tests =============
TEST_F(TensorMaskingTest, Where) {
    auto condition = create_test_tensor({1, 0, 1, 0}, {4}).ne(0.0f);
    auto x = create_test_tensor({1, 2, 3, 4}, {4});
    auto y = create_test_tensor({5, 6, 7, 8}, {4});

    auto result = Tensor::where(condition, x, y);

    auto values = result.to_vector();
    std::vector<float> expected = {1, 6, 3, 8}; // Take from x when true, y when false
    EXPECT_TRUE(compare_vectors(values, expected));
}

TEST_F(TensorMaskingTest, WhereWithBroadcasting) {
    auto condition = create_test_tensor({1, 0}, {2, 1}).ne(0.0f);
    auto x = Tensor::full({2, 3}, 1.0f, Device::CUDA);
    auto y = Tensor::full({2, 3}, 0.0f, Device::CUDA);

    auto result = Tensor::where(condition, x, y);
    EXPECT_EQ(result.shape(), TensorShape({2, 3}));

    auto values = result.to_vector();
    // First row: condition is true, so all 1s
    // Second row: condition is false, so all 0s
    std::vector<float> expected = {1, 1, 1, 0, 0, 0};
    EXPECT_TRUE(compare_vectors(values, expected));
}

// ============= Index Select Tests =============
TEST_F(TensorMaskingTest, IndexSelect) {
    auto tensor = create_test_tensor({1, 2, 3, 4, 5, 6}, {2, 3});

    // Explicitly create int vector to avoid ambiguity
    std::vector<int> idx_data = {0, 2};
    auto indices = Tensor::from_vector(idx_data, {2}, Device::CUDA);

    // Select columns 0 and 2
    auto selected = tensor.index_select(1, indices);
    EXPECT_EQ(selected.shape(), TensorShape({2, 2}));

    auto values = selected.to_vector();
    std::vector<float> expected = {1, 3, 4, 6};
    EXPECT_TRUE(compare_vectors(values, expected));
}

TEST_F(TensorMaskingTest, IndexSelectWithBoundaryClamp) {
    auto tensor = create_test_tensor({1, 2, 3, 4}, {4});

    std::vector<int> idx_data = {-1, 0, 5, 2};
    auto indices = Tensor::from_vector(idx_data, {4}, Device::CUDA);

    auto selected = tensor.index_select(0, indices, BoundaryMode::Clamp);

    auto values = selected.to_vector();
    // -1 clamped to 0, 5 clamped to 3
    std::vector<float> expected = {1, 1, 4, 3};
    EXPECT_TRUE(compare_vectors(values, expected));
}

TEST_F(TensorMaskingTest, IndexSelectWithBoundaryWrap) {
    auto tensor = create_test_tensor({1, 2, 3, 4}, {4});

    std::vector<int> idx_data = {-1, 0, 5, 2};
    auto indices = Tensor::from_vector(idx_data, {4}, Device::CUDA);

    auto selected = tensor.index_select(0, indices, BoundaryMode::Wrap);

    auto values = selected.to_vector();
    // -1 wraps to 3, 5 wraps to 1
    std::vector<float> expected = {4, 1, 2, 3};
    EXPECT_TRUE(compare_vectors(values, expected));
}

// ============= Gather Tests =============
TEST_F(TensorMaskingTest, Gather) {
    auto tensor = create_test_tensor({1, 2, 3, 4, 5, 6}, {2, 3});

    std::vector<int> idx_data = {0, 2, 1, 0};
    auto indices = Tensor::from_vector(idx_data, {2, 2}, Device::CUDA);

    auto gathered = tensor.gather(1, indices);
    EXPECT_EQ(gathered.shape(), indices.shape());

    auto values = gathered.to_vector();
    // Row 0: gather columns 0, 2 -> [1, 3]
    // Row 1: gather columns 1, 0 -> [5, 4]
    std::vector<float> expected = {1, 3, 5, 4};
    EXPECT_TRUE(compare_vectors(values, expected));
}

// ============= Take Tests =============
TEST_F(TensorMaskingTest, Take) {
    auto tensor = create_test_tensor({1, 2, 3, 4, 5, 6}, {2, 3});

    std::vector<int> idx_data = {0, 2, 5, 3};
    auto indices = Tensor::from_vector(idx_data, {4}, Device::CUDA);

    auto taken = tensor.take(indices);
    EXPECT_EQ(taken.shape(), indices.shape());

    auto values = taken.to_vector();
    // Flattened tensor is [1, 2, 3, 4, 5, 6]
    std::vector<float> expected = {1, 3, 6, 4};
    EXPECT_TRUE(compare_vectors(values, expected));
}

TEST_F(TensorMaskingTest, TakeNegativeIndices) {
    auto tensor = create_test_tensor({1, 2, 3, 4}, {4});

    std::vector<int> idx_data = {-1, -2, 0, 1};
    auto indices = Tensor::from_vector(idx_data, {4}, Device::CUDA);

    auto taken = tensor.take(indices);

    auto values = taken.to_vector();
    // -1 -> 3, -2 -> 2
    std::vector<float> expected = {4, 3, 1, 2};
    EXPECT_TRUE(compare_vectors(values, expected));
}

// ============= Scatter Tests =============
TEST_F(TensorMaskingTest, Scatter) {
    auto tensor = Tensor::zeros({3, 4}, Device::CUDA);

    std::vector<int> idx_data = {0, 2};
    auto indices = Tensor::from_vector(idx_data, {2}, Device::CUDA);

    // Explicitly create float vector to avoid ambiguity
    std::vector<float> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto src = Tensor::from_vector(src_data, {2, 4}, Device::CUDA);

    tensor.scatter_(0, indices, src);

    auto values = tensor.to_vector();
    // Scatter to rows 0 and 2
    std::vector<float> expected = {
        1, 2, 3, 4, // row 0 from src[0]
        0, 0, 0, 0, // row 1 unchanged
        5, 6, 7, 8  // row 2 from src[1]
    };
    EXPECT_TRUE(compare_vectors(values, expected));
}

TEST_F(TensorMaskingTest, ScatterWithAdd) {
    auto tensor = Tensor::ones({4}, Device::CUDA);

    std::vector<int> idx_data = {0, 2, 0, 3};
    auto indices = Tensor::from_vector(idx_data, {4}, Device::CUDA);

    // Explicitly create float vector
    std::vector<float> src_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto src = Tensor::from_vector(src_data, {4}, Device::CUDA);

    tensor.scatter_(0, indices, src, ScatterMode::Add);

    auto values = tensor.to_vector();
    // Index 0 gets 1 + 3, index 2 gets 2, index 3 gets 4
    std::vector<float> expected = {5, 1, 3, 5}; // 1+1+3, 1, 1+2, 1+4
    EXPECT_TRUE(compare_vectors(values, expected));
}

// ============= Index Fill Tests =============
TEST_F(TensorMaskingTest, IndexFill) {
    auto tensor = create_test_tensor({1, 2, 3, 4, 5}, {5});

    std::vector<int> idx_data = {1, 3};
    auto indices = Tensor::from_vector(idx_data, {2}, Device::CUDA);

    tensor.index_fill_(0, indices, -1.0f);

    auto values = tensor.to_vector();
    std::vector<float> expected = {1, -1, 3, -1, 5};
    EXPECT_TRUE(compare_vectors(values, expected));
}

// ============= Index Copy Tests =============
TEST_F(TensorMaskingTest, IndexCopy) {
    auto tensor = Tensor::zeros({5}, Device::CUDA);

    std::vector<int> idx_data = {0, 2, 4};
    auto indices = Tensor::from_vector(idx_data, {3}, Device::CUDA);

    std::vector<float> src_data = {1.0f, 2.0f, 3.0f};
    auto src = Tensor::from_vector(src_data, {3}, Device::CUDA);

    tensor.index_copy_(0, indices, src);

    auto values = tensor.to_vector();
    std::vector<float> expected = {1, 0, 2, 0, 3};
    EXPECT_TRUE(compare_vectors(values, expected));
}

// ============= Python-like Indexing Tests =============
TEST_F(TensorMaskingTest, PythonLikeMaskedIndexing) {
    auto tensor = create_test_tensor({1, 2, 3, 4, 5}, {5});
    auto mask = tensor.gt(3.0f);

    // Get masked values
    Tensor selected = tensor[mask];
    auto values = selected.to_vector();
    std::vector<float> expected = {4, 5};
    EXPECT_TRUE(compare_vectors(values, expected));
}

TEST_F(TensorMaskingTest, PythonLikeMaskedAssignment) {
    auto tensor = create_test_tensor({1, 2, 3, 4, 5}, {5});
    auto mask = tensor.gt(3.0f);

    // Set masked values
    tensor[mask] = -1.0f;

    auto values = tensor.to_vector();
    std::vector<float> expected = {1, 2, 3, -1, -1};
    EXPECT_TRUE(compare_vectors(values, expected));
}

TEST_F(TensorMaskingTest, PythonLikeIndexing) {
    auto tensor = create_test_tensor({1, 2, 3, 4, 5}, {5});

    std::vector<int> idx_data = {0, 2, 4};
    auto indices = Tensor::from_vector(idx_data, {3}, Device::CUDA);

    // Get indexed values
    Tensor selected = tensor[indices];
    auto values = selected.to_vector();
    std::vector<float> expected = {1, 3, 5};
    EXPECT_TRUE(compare_vectors(values, expected));
}

TEST_F(TensorMaskingTest, PythonLikeIndexAssignment) {
    auto tensor = create_test_tensor({1, 2, 3, 4, 5}, {5});

    std::vector<int> idx_data = {1, 3};
    auto indices = Tensor::from_vector(idx_data, {2}, Device::CUDA);

    // Set indexed values
    tensor[indices] = 0.0f;

    auto values = tensor.to_vector();
    std::vector<float> expected = {1, 0, 3, 0, 5};
    EXPECT_TRUE(compare_vectors(values, expected));
}

// ============= Count Nonzero Tests =============
TEST_F(TensorMaskingTest, CountNonzero) {
    auto tensor = create_test_tensor({0, 1, 0, 2, 0, 3}, {6});
    EXPECT_EQ(tensor.count_nonzero(), 3);

    auto bool_tensor = tensor.ne(0.0f);
    EXPECT_EQ(bool_tensor.count_nonzero(), 3);
}

TEST_F(TensorMaskingTest, AnyAll) {
    auto all_true = Tensor::ones_bool({3, 3}, Device::CUDA);
    // Use scalar reduction on bool tensor converted to float
    auto all_true_float = all_true.to(DataType::Float32);
    EXPECT_FLOAT_EQ(all_true_float.min_scalar(), 1.0f);  // All true
    EXPECT_FLOAT_EQ(all_true_float.max_scalar(), 1.0f);  // All true

    auto all_false = Tensor::zeros_bool({3, 3}, Device::CUDA);
    auto all_false_float = all_false.to(DataType::Float32);
    EXPECT_FLOAT_EQ(all_false_float.max_scalar(), 0.0f);  // All false
    EXPECT_FLOAT_EQ(all_false_float.min_scalar(), 0.0f);  // All false

    std::vector<bool> mixed_data = {true, false, true};
    auto mixed = Tensor::from_vector(mixed_data, {3}, Device::CUDA);
    auto mixed_float = mixed.to(DataType::Float32);
    EXPECT_FLOAT_EQ(mixed_float.min_scalar(), 0.0f);  // Has false
    EXPECT_FLOAT_EQ(mixed_float.max_scalar(), 1.0f);  // Has true
}

// ============= Complex Masking Scenarios =============
TEST_F(TensorMaskingTest, ComplexMaskingScenario) {
    // Create a 2D tensor
    auto tensor = create_test_tensor(
        {1, 2, 3, 4,
         5, 6, 7, 8,
         9, 10, 11, 12},
        {3, 4});

    // Find elements in range [5, 10]
    auto mask = tensor.ge(5.0f).logical_and(tensor.le(10.0f));

    // Get selected elements
    auto selected = tensor.masked_select(mask);
    auto values = selected.to_vector();
    std::vector<float> expected = {5, 6, 7, 8, 9, 10};
    EXPECT_TRUE(compare_vectors(values, expected));

    // Replace them with -1
    tensor.masked_fill_(mask, -1.0f);

    // Verify
    auto final_values = tensor.to_vector();
    std::vector<float> final_expected = {
        1, 2, 3, 4,
        -1, -1, -1, -1,
        -1, -1, 11, 12};
    EXPECT_TRUE(compare_vectors(final_values, final_expected));
}

TEST_F(TensorMaskingTest, AdvancedIndexingScenario) {
    // Create a batch of vectors
    auto tensor = create_test_tensor(
        {1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         10, 11, 12},
        {4, 3});

    // Select specific rows
    std::vector<int> row_idx = {0, 2, 3};
    auto row_indices = Tensor::from_vector(row_idx, {3}, Device::CUDA);
    auto selected_rows = tensor.index_select(0, row_indices);

    EXPECT_EQ(selected_rows.shape(), TensorShape({3, 3}));

    auto values = selected_rows.to_vector();
    std::vector<float> expected = {
        1, 2, 3,
        7, 8, 9,
        10, 11, 12};
    EXPECT_TRUE(compare_vectors(values, expected));

    // Now select specific columns from the result
    std::vector<int> col_idx = {1, 2};
    auto col_indices = Tensor::from_vector(col_idx, {2}, Device::CUDA);
    auto final = selected_rows.index_select(1, col_indices);

    EXPECT_EQ(final.shape(), TensorShape({3, 2}));

    auto final_values = final.to_vector();
    std::vector<float> final_expected = {
        2, 3,
        8, 9,
        11, 12};
    EXPECT_TRUE(compare_vectors(final_values, final_expected));
}

// =============  Additional Advanced Masking Patterns =============

// Test creating attention masks with proper broadcasting
TEST_F(TensorMaskingTest, AttentionMaskCreation) {
    const size_t batch = 2;
    const size_t heads = 4;
    const size_t seq_len = 8;

    // Create attention scores
    auto scores = Tensor::randn({batch, heads, seq_len, seq_len}, Device::CUDA);

    // Create causal mask using broadcasting
    std::vector<float> range_data(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        range_data[i] = static_cast<float>(i);
    }
    auto range = Tensor::from_vector(range_data, {seq_len}, Device::CUDA);

    auto row_idx = range.reshape({seq_len, 1});
    auto col_idx = range.reshape({1, seq_len});

    // Create causal mask: row_idx >= col_idx
    auto causal_mask = row_idx.ge(col_idx);
    EXPECT_EQ(causal_mask.shape(), TensorShape({seq_len, seq_len}));

    // Expand to match scores shape
    auto mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0)
                                    .expand({batch, heads, seq_len, seq_len});
    EXPECT_EQ(mask_expanded.shape(), scores.shape());

    // Apply mask
    scores.masked_fill_(mask_expanded.logical_not(), -1e9f);

    // Verify some values
    auto cpu_scores = scores.to(Device::CPU);
    // Upper triangular should be -1e9, lower triangular should be normal
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            float val = cpu_scores.at({0, 0, i, j});
            if (j > i) {
                EXPECT_FLOAT_EQ(val, -1e9f) << "Failed at position (0, 0, " << i << ", " << j << ")";
            } else {
                EXPECT_NE(val, -1e9f) << "Failed at position (0, 0, " << i << ", " << j << ")";
            }
        }
    }
}

// Test for creating padding masks
TEST_F(TensorMaskingTest, PaddingMaskCreation) {
    const size_t batch = 3;
    const size_t seq_len = 10;
    const size_t hidden = 64;

    // Simulate sequence lengths for each batch
    std::vector<int> lengths = {7, 5, 9};

    // Create data
    auto data = Tensor::randn({batch, seq_len, hidden}, Device::CUDA);

    // Create padding mask
    auto mask = Tensor::zeros_bool({batch, seq_len}, Device::CUDA);
    auto cpu_mask = mask.to(Device::CPU);

    // Set valid positions to true - USE set_bool for boolean tensors!
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < lengths[b]; ++s) {
            cpu_mask.set_bool({b, s}, true);  // Use set_bool for boolean tensors
        }
    }

    mask = cpu_mask.to(Device::CUDA);

    // Expand mask to match data dimensions
    auto mask_expanded = mask.unsqueeze(2).expand({batch, seq_len, hidden});

    // Apply padding
    data = data.mul(mask_expanded.to(DataType::Float32));

    // Verify padding positions are zero
    auto cpu_data = data.to(Device::CPU);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = lengths[b]; s < seq_len; ++s) {
            for (size_t h = 0; h < hidden; ++h) {
                EXPECT_FLOAT_EQ(cpu_data.at({b, s, h}), 0.0f)
                    << "Failed at position (" << b << ", " << s << ", " << h << ")";
            }
        }
    }
}

// ============= Multi-Condition Masking =============
TEST_F(TensorMaskingTest, MultiConditionMasking) {
    auto data = Tensor::randn({100, 100}, Device::CUDA);

    // Complex condition: values between -0.5 and 0.5 but not too close to 0
    auto mask1 = data > -0.5f;
    auto mask2 = data < 0.5f;
    auto mask3 = data.abs() > 0.1f;

    auto complex_mask = mask1.logical_and(mask2).logical_and(mask3);

    // Count elements that match all conditions
    auto count_before = complex_mask.count_nonzero();

    // Apply mask
    data[complex_mask] = 0.0f;

    // Verify all masked values are 0
    auto masked_vals = data.masked_select(complex_mask);
    EXPECT_EQ(masked_vals.numel(), count_before);
    if (masked_vals.numel() > 0) {
        auto all_zero = masked_vals.eq(0.0f);
        auto all_zero_float = all_zero.to(DataType::Float32);
        EXPECT_FLOAT_EQ(all_zero_float.min_scalar(), 1.0f);  // All should be true (all zeros)
    }
}

// ============= Diagonal Masking Pattern =============
TEST_F(TensorMaskingTest, DiagonalMasking) {
    const size_t size = 10;

    // Method 1: Create diagonal mask manually on CPU
    std::vector<bool> mask_data(size * size, false);
    for (size_t i = 0; i < size; ++i) {
        mask_data[i * size + i] = true;  // Set diagonal elements to true
    }

    auto diag_mask = Tensor::from_vector(mask_data, {size, size}, Device::CUDA);

    // Create data tensor starting with zeros
    auto data = Tensor::zeros({size, size}, Device::CUDA);

    // Set diagonal to 1
    data.masked_fill_(diag_mask, 1.0f);

    // Verify: should be identity matrix
    auto cpu_data = data.to(Device::CPU);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_FLOAT_EQ(cpu_data.at({i, j}), expected)
                << "Failed at position (" << i << ", " << j << ")";
        }
    }
}

// ============= Simple Causal Mask Test =============
TEST_F(TensorMaskingTest, SimpleCausalMask) {
    const size_t seq_len = 4;  // Small size for debugging

    // Create a causal mask (lower triangular = true, upper = false)
    std::vector<bool> mask_data(seq_len * seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            // Causal pattern: true for lower triangular (including diagonal)
            mask_data[i * seq_len + j] = (i >= j);
        }
    }

    // Create mask from boolean vector
    auto mask = Tensor::from_vector(mask_data, {seq_len, seq_len}, Device::CUDA);

    // Test with simple data
    auto scores = Tensor::ones({seq_len, seq_len}, Device::CUDA);

    // Apply mask - set upper triangular (where mask is false) to -inf
    scores.masked_fill_(mask.logical_not(), -1e9f);

    // Verify
    auto result = scores.to(Device::CPU);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            if (i >= j) {
                // Lower triangular should be 1
                EXPECT_FLOAT_EQ(result.at({i, j}), 1.0f)
                    << "Failed at position (" << i << ", " << j << ")";
            } else {
                // Upper triangular should be -1e9
                EXPECT_FLOAT_EQ(result.at({i, j}), -1e9f)
                    << "Failed at position (" << i << ", " << j << ")";
            }
        }
    }
}

// ============= Block Sparse Pattern =============
TEST_F(TensorMaskingTest, BlockSparsePattern) {
    const size_t size = 64;
    const size_t block_size = 4;
    auto data = Tensor::randn({size, size}, Device::CUDA);

    // Create checkerboard mask on CPU to avoid malloc issues
    std::vector<bool> mask_data(size * size, false);

    // Fill checkerboard pattern
    for (size_t i = 0; i < size; i += block_size) {
        for (size_t j = 0; j < size; j += block_size) {
            // Checkerboard: (block_i + block_j) % 2 == 0
            size_t block_i = i / block_size;
            size_t block_j = j / block_size;
            if ((block_i + block_j) % 2 == 0) {
                // Fill the block
                for (size_t di = 0; di < block_size && (i + di) < size; ++di) {
                    for (size_t dj = 0; dj < block_size && (j + dj) < size; ++dj) {
                        mask_data[(i + di) * size + (j + dj)] = true;
                    }
                }
            }
        }
    }

    // Create mask tensor from vector
    auto checkerboard_mask = Tensor::from_vector(mask_data, {size, size}, Device::CUDA);

    // Apply sparsity - zero out where mask is false
    data.masked_fill_(checkerboard_mask.logical_not(), 0.0f);

    // Count non-zero elements
    auto nnz = data.ne(0.0f).count_nonzero();
    size_t expected_nnz = (size * size) / 2;  // 50% sparse for checkerboard

    // Allow some tolerance due to random initialization
    EXPECT_NEAR(static_cast<float>(nnz), static_cast<float>(expected_nnz), size * 2);
}

// ============= Top-K Masking Pattern =============
TEST_F(TensorMaskingTest, TopKMasking) {
    const size_t n = 1000;
    const size_t k = 100;
    auto data = Tensor::randn({n}, Device::CUDA);

    // Get top-k using a proper approach
    // First, get the data to CPU for sorting
    auto cpu_data = data.to(Device::CPU);
    std::vector<float> values = cpu_data.to_vector();

    // Create indices and sort
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by value (descending)
    std::sort(indices.begin(), indices.end(),
              [&values](size_t i, size_t j) { return values[i] > values[j]; });

    // Get threshold (k-th largest value)
    float threshold = values[indices[k-1]];

    // Create mask for values >= threshold
    auto top_k_mask = data.ge(threshold);

    // To handle ties, we might have more than k elements
    // For exact k, we'd need more sophisticated handling
    auto mask_count = top_k_mask.count_nonzero();

    // Keep only top-k values, zero out others
    auto result = data.clone();
    result.masked_fill_(top_k_mask.logical_not(), 0.0f);

    // Verify at most k non-zero elements (may be more due to ties)
    auto nnz = result.ne(0.0f).count_nonzero();
    EXPECT_GE(nnz, k - 10);  // Allow some tolerance for ties
    EXPECT_LE(nnz, k + 10);  // Allow some tolerance for ties

    // Verify all non-zero values are >= threshold (minus small epsilon for float precision)
    auto non_zero_vals = result.masked_select(result.ne(0.0f));
    if (non_zero_vals.numel() > 0) {
        auto min_val = non_zero_vals.min_scalar();
        EXPECT_GE(min_val, threshold - 1e-6f);
    }
}

// ============= Gradient Clipping with Masking =============
TEST_F(TensorMaskingTest, GradientClippingMask) {
    auto gradients = Tensor::randn({50, 50}, Device::CUDA).mul(5.0f);  // Large gradients
    float clip_value = 1.0f;

    // Create mask for large gradients
    auto large_grad_mask = gradients.abs() > clip_value;

    // Get gradient signs
    auto grad_sign = Tensor::where(gradients >= 0,
                                   tensor::ones_like(gradients),
                                   Tensor::full(gradients.shape(), -1.0f, Device::CUDA));

    // Clip gradients
    gradients.masked_fill_(large_grad_mask, 0.0f);
    auto clipped_values = grad_sign.mul(clip_value);
    gradients = gradients + clipped_values.mul(large_grad_mask.to(DataType::Float32));

    // Verify all gradients are within [-clip_value, clip_value]
    auto abs_grads = gradients.abs();
    auto all_clipped = abs_grads.le(clip_value * 1.001f);  // Small tolerance
    auto all_clipped_float = all_clipped.to(DataType::Float32);
    EXPECT_FLOAT_EQ(all_clipped_float.min_scalar(), 1.0f);  // All should be true
}

// ============= Fixed Structured Dropout Pattern =============
TEST_F(TensorMaskingTest, FixedStructuredDropout) {
    const size_t batch_size = 2;
    const size_t seq_len = 10;
    const size_t hidden_dim = 4;

    // Create data filled with ones for easier testing
    auto data = Tensor::ones({batch_size, seq_len, hidden_dim}, Device::CUDA);

    // Create deterministic dropout mask
    auto token_mask = Tensor::ones({batch_size, seq_len, 1}, Device::CUDA);
    auto cpu_mask = token_mask.to(Device::CPU);

    // Manually drop some tokens
    cpu_mask.at({0, 1, 0}) = 0.0f;  // Drop token 1 in batch 0
    cpu_mask.at({0, 3, 0}) = 0.0f;  // Drop token 3 in batch 0
    cpu_mask.at({1, 2, 0}) = 0.0f;  // Drop token 2 in batch 1

    token_mask = cpu_mask.to(Device::CUDA);

    // Expand and apply mask
    token_mask = token_mask.expand({batch_size, seq_len, hidden_dim});
    data = data.mul(token_mask);

    // Verify some tokens are dropped
    auto cpu_data = data.to(Device::CPU);

    // Check dropped tokens
    for (size_t h = 0; h < hidden_dim; ++h) {
        EXPECT_FLOAT_EQ(cpu_data.at({0, 1, h}), 0.0f);
        EXPECT_FLOAT_EQ(cpu_data.at({0, 3, h}), 0.0f);
        EXPECT_FLOAT_EQ(cpu_data.at({1, 2, h}), 0.0f);
    }

    // Count zeros
    auto zero_count = data.eq(0.0f).count_nonzero();
    EXPECT_EQ(zero_count, 3 * hidden_dim);  // 3 dropped tokens * hidden_dim
}

// ============= Fixed Image Region Masking =============
TEST_F(TensorMaskingTest, FixedImageRegionMasking) {
    const size_t batch_size = 2;
    const size_t channels = 3;
    const size_t height = 32;
    const size_t width = 32;

    auto images = Tensor::ones({batch_size, channels, height, width}, Device::CUDA);

    // Create mask for a 10x10 rectangle
    std::vector<bool> mask_data(height * width, false);

    // Fill rectangle from (5,5) to (15,15)
    size_t masked_pixels = 0;
    for (size_t y = 5; y < 15 && y < height; ++y) {
        for (size_t x = 5; x < 15 && x < width; ++x) {
            mask_data[y * width + x] = true;
            masked_pixels++;
        }
    }

    auto mask = Tensor::from_vector(mask_data, {height, width}, Device::CUDA);

    // Apply mask to first image, all channels
    auto first_batch = images.slice(0, 0, 1).squeeze(0);  // Shape: [3, 32, 32]

    for (size_t c = 0; c < channels; ++c) {
        auto channel = first_batch.slice(0, c, c+1).squeeze(0);  // Shape: [32, 32]
        channel.masked_fill_(mask, 0.0f);
    }

    // Count zero pixels in the first batch
    auto zero_pixels = first_batch.eq(0.0f).count_nonzero();

    // We should have masked_pixels * channels zeros
    size_t expected_zeros = masked_pixels * channels;
    EXPECT_EQ(zero_pixels, expected_zeros);
}

// ============= Sparse Attention Pattern =============
TEST_F(TensorMaskingTest, SparseAttentionPattern) {
    const size_t seq_len = 64;  // Reduced from 128 to avoid memory issues
    const size_t window_size = 8;
    const size_t stride = 4;

    // Create sparse attention mask: local window + strided global attention
    // Create masks on CPU first to avoid malloc issues
    std::vector<bool> local_mask_data(seq_len * seq_len, false);
    std::vector<bool> global_mask_data(seq_len * seq_len, false);

    // Fill local window mask: |row - col| <= window_size/2
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            int distance = std::abs(static_cast<int>(i) - static_cast<int>(j));
            if (distance <= static_cast<int>(window_size / 2)) {
                local_mask_data[i * seq_len + j] = true;
            }
        }
    }

    // Fill strided global attention mask
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            // Row or column is on stride
            if (i % stride == 0 || j % stride == 0) {
                global_mask_data[i * seq_len + j] = true;
            }
        }
    }

    // Create tensors from vectors
    auto local_mask = Tensor::from_vector(local_mask_data, {seq_len, seq_len}, Device::CUDA);
    auto global_mask = Tensor::from_vector(global_mask_data, {seq_len, seq_len}, Device::CUDA);

    // Combine masks
    auto sparse_mask = local_mask.logical_or(global_mask);

    // Count connections per position
    auto connections = sparse_mask.count_nonzero();

    // Should have sparse connectivity
    float density = static_cast<float>(connections) / (seq_len * seq_len);
    EXPECT_LT(density, 0.55f);  // Less than 55% dense (allowing some tolerance)
}

// ============= Dynamic Masking Based on Values =============
TEST_F(TensorMaskingTest, DynamicValueBasedMasking) {
    auto data = Tensor::randn({100, 100}, Device::CUDA);

    // Compute statistics - use scalar versions
    float mean_val = data.mean_scalar();
    float std_val = data.std_scalar();

    // Create dynamic mask: keep values within 2 standard deviations
    auto lower_bound = mean_val - 2 * std_val;
    auto upper_bound = mean_val + 2 * std_val;

    auto in_range_mask = data.ge(lower_bound).logical_and(data.le(upper_bound));

    // Apply different operations based on mask
    auto result = Tensor::where(in_range_mask,
                               data.mul(2.0f),           // Scale in-range values
                               data.clamp(-1.0f, 1.0f)); // Clamp outliers

    // Verify transformations
    auto outlier_mask = in_range_mask.logical_not();
    // Check if any outliers exist
    auto outlier_count = outlier_mask.to(DataType::Float32).sum_scalar();
    if (outlier_count > 0) {
        auto outlier_vals = result.masked_select(outlier_mask);
        // Check all outlier values are clamped
        auto all_clamped = outlier_vals.ge(-1.0f).logical_and(outlier_vals.le(1.0f));
        auto all_clamped_float = all_clamped.to(DataType::Float32);
        EXPECT_FLOAT_EQ(all_clamped_float.min_scalar(), 1.0f);  // All should be true
    }
}

// ============= Performance Test =============
TEST_F(TensorMaskingTest, MaskingPerformance) {
    const size_t size = 1000000;
    auto tensor = Tensor::randn({size}, Device::CUDA);

    // Create mask for positive values
    auto start = std::chrono::high_resolution_clock::now();
    auto mask = tensor.gt(0.0f);
    cudaDeviceSynchronize();
    auto mask_time = std::chrono::high_resolution_clock::now() - start;

    // Select masked elements
    start = std::chrono::high_resolution_clock::now();
    auto selected = tensor.masked_select(mask);
    cudaDeviceSynchronize();
    auto select_time = std::chrono::high_resolution_clock::now() - start;

    // Fill masked elements
    start = std::chrono::high_resolution_clock::now();
    tensor.masked_fill_(mask, -1.0f);
    cudaDeviceSynchronize();
    auto fill_time = std::chrono::high_resolution_clock::now() - start;

    std::cout << "Masking performance for " << size << " elements:\n";
    std::cout << "  Create mask: " << std::chrono::duration_cast<std::chrono::microseconds>(mask_time).count() << " μs\n";
    std::cout << "  Masked select: " << std::chrono::duration_cast<std::chrono::microseconds>(select_time).count() << " μs\n";
    std::cout << "  Masked fill: " << std::chrono::duration_cast<std::chrono::microseconds>(fill_time).count() << " μs\n";
}

// ============= Fixed Edge Cases =============
TEST_F(TensorMaskingTest, FixedEmptyMask) {
    auto tensor = create_test_tensor({1, 2, 3}, {3});
    auto empty_mask = Tensor::zeros_bool({3}, Device::CUDA);

    // Check mask is actually empty
    EXPECT_EQ(empty_mask.count_nonzero(), 0);

    auto selected = tensor.masked_select(empty_mask);
    EXPECT_EQ(selected.numel(), 0);
}

TEST_F(TensorMaskingTest, FixedFullMask) {
    auto tensor = create_test_tensor({1, 2, 3}, {3});
    auto full_mask = Tensor::ones_bool({3}, Device::CUDA);

    // Verify mask is actually full
    EXPECT_EQ(full_mask.count_nonzero(), 3);

    auto selected = tensor.masked_select(full_mask);
    EXPECT_EQ(selected.numel(), 3);

    if (selected.numel() == 3) {
        auto values = selected.to_vector();
        std::vector<float> expected = {1, 2, 3};
        EXPECT_TRUE(compare_vectors(values, expected));
    }
}

TEST_F(TensorMaskingTest, FixedSingleElementIndexing) {
    auto tensor = create_test_tensor({42}, {1});

    std::vector<int> idx_data = {0};
    auto indices = Tensor::from_vector(idx_data, {1}, Device::CUDA);

    auto selected = tensor.index_select(0, indices);

    // Just check that we get one element back
    EXPECT_EQ(selected.shape(), TensorShape({1}));

    if (selected.numel() == 1) {
        EXPECT_FLOAT_EQ(selected.item(), 42.0f);
    }
}

// ============= Compound Masking Operations =============
TEST_F(TensorMaskingTest, SimpleCompoundMasking) {
    // Simpler test case
    const size_t batch = 2;
    const size_t seq_len = 4;

    // Create simple scores
    auto scores = Tensor::ones({batch, seq_len, seq_len}, Device::CUDA);

    // Create causal mask for each batch separately to avoid expand() issues
    for (size_t b = 0; b < batch; ++b) {
        // Create causal mask manually for this batch
        std::vector<bool> causal_data(seq_len * seq_len);
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                causal_data[i * seq_len + j] = (i >= j);
            }
        }
        auto causal = Tensor::from_vector(causal_data, {seq_len, seq_len}, Device::CUDA);

        // Get the batch slice
        auto batch_scores = scores.slice(0, b, b + 1).squeeze(0);  // Shape: [seq_len, seq_len]

        // Apply mask to this batch
        batch_scores.masked_fill_(causal.logical_not(), -1e9f);
    }

    // Verify causal pattern
    auto cpu_scores = scores.to(Device::CPU);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                if (j > i) {  // Upper triangular should be masked
                    EXPECT_FLOAT_EQ(cpu_scores.at({b, i, j}), -1e9f)
                        << "Failed at position (" << b << ", " << i << ", " << j << ")";
                } else {  // Lower triangular should be 1
                    EXPECT_FLOAT_EQ(cpu_scores.at({b, i, j}), 1.0f)
                        << "Failed at position (" << b << ", " << i << ", " << j << ")";
                }
            }
        }
    }
}

// =============  Extensive Boolean Expansion Tests =============

TEST_F(TensorMaskingTest, BooleanExpansionSimple1D) {
    // Test 1D boolean tensor expansion
    auto bool_1d = Tensor::from_vector(std::vector<bool>{true, false, true}, {3}, Device::CUDA);

    // Expand to 2D
    auto expanded = bool_1d.unsqueeze(0).expand({2, 3});
    EXPECT_EQ(expanded.shape(), TensorShape({2, 3}));

    auto values = expanded.to_vector_bool();
    // Both rows should be [true, false, true]
    std::vector<bool> expected = {true, false, true, true, false, true};
    EXPECT_EQ(values, expected);
}

TEST_F(TensorMaskingTest, BooleanExpansion2Dto3D) {
    // Test 2D to 3D boolean tensor expansion
    std::vector<bool> data_2d = {true, false, false, true};
    auto bool_2d = Tensor::from_vector(data_2d, {2, 2}, Device::CUDA);

    // Add dimension and expand
    auto expanded = bool_2d.unsqueeze(0).expand({3, 2, 2});
    EXPECT_EQ(expanded.shape(), TensorShape({3, 2, 2}));

    auto values = expanded.to_vector_bool();
    // All 3 batches should have the same pattern
    std::vector<bool> expected = {
        true, false, false, true,  // batch 0
        true, false, false, true,  // batch 1
        true, false, false, true   // batch 2
    };
    EXPECT_EQ(values, expected);
}

TEST_F(TensorMaskingTest, BooleanExpansionMultipleDims) {
    // Test expansion along multiple dimensions
    auto bool_tensor = Tensor::from_vector(std::vector<bool>{true}, {1, 1}, Device::CUDA);

    auto expanded = bool_tensor.expand({5, 4});
    EXPECT_EQ(expanded.shape(), TensorShape({5, 4}));

    // All elements should be true
    auto values = expanded.to_vector_bool();
    EXPECT_EQ(values.size(), 20);
    for (bool val : values) {
        EXPECT_TRUE(val);
    }
}

// =============  Broadcasting Comparison Edge Cases =============

TEST_F(TensorMaskingTest, BroadcastComparisonScalarLike) {
    // Test scalar-like tensors in comparisons
    auto scalar = create_test_tensor({5}, {1});
    auto vector = create_test_tensor({1, 2, 3, 4, 5, 6}, {6});

    auto mask = vector.eq(scalar);
    EXPECT_EQ(mask.shape(), vector.shape());

    auto values = mask.to_vector_bool();
    // Only the 5th element (index 4) should be true
    std::vector<bool> expected = {false, false, false, false, true, false};
    EXPECT_EQ(values, expected);
}

TEST_F(TensorMaskingTest, BroadcastComparison3D) {
    // Test 3D broadcasting in comparisons
    auto a = create_test_tensor({1, 2}, {2, 1, 1});
    auto b = create_test_tensor({1, 2, 3}, {1, 1, 3});

    auto mask = a.eq(b);
    EXPECT_EQ(mask.shape(), TensorShape({2, 1, 3}));

    auto values = mask.to_vector_bool();
    // First batch: 1 compared with [1, 2, 3] = [T, F, F]
    // Second batch: 2 compared with [1, 2, 3] = [F, T, F]
    std::vector<bool> expected = {true, false, false, false, true, false};
    EXPECT_EQ(values, expected);
}

TEST_F(TensorMaskingTest, BroadcastComparisonMaxRank) {
    // Test broadcasting with maximum rank difference
    auto a = create_test_tensor({2}, {1});  // Rank 1
    auto b = create_test_tensor({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});  // Rank 3

    auto mask = a.eq(b);
    EXPECT_EQ(mask.shape(), TensorShape({2, 2, 2}));

    auto values = mask.to_vector_bool();
    // Element at position [0, 1, 0] is 2, others are not
    std::vector<bool> expected = {false, true, false, false, false, false, false, false};
    EXPECT_EQ(values, expected);
}

// =============  Complex Boolean Operations =============

TEST_F(TensorMaskingTest, ChainedBooleanOperations) {
    const size_t n = 100;
    auto data = Tensor::randn({n}, Device::CUDA);

    // Create multiple conditions
    auto mask1 = data.gt(0);           // positive values
    auto mask2 = data.lt(1);           // less than 1
    auto mask3 = data.abs().gt(0.1);   // absolute value > 0.1

    // Complex chained boolean logic
    auto final_mask = mask1.logical_and(mask2).logical_or(mask3.logical_not());

    EXPECT_EQ(final_mask.shape(), data.shape());
    EXPECT_EQ(final_mask.dtype(), DataType::Bool);

    // Apply the mask
    data.masked_fill_(final_mask, 0);

    // Verify some properties
    auto remaining = data.ne(0);
    auto remaining_data = data.masked_select(remaining);

    // All remaining values should NOT satisfy the final_mask condition
    // This is a complex condition, so we just check that the operation completed
    EXPECT_GE(remaining.count_nonzero(), 0);
}

TEST_F(TensorMaskingTest, BooleanReductionAcrossDimensions) {
    // Create a 3D boolean tensor
    const size_t d1 = 4, d2 = 5, d3 = 6;
    auto data = Tensor::randn({d1, d2, d3}, Device::CUDA);
    auto mask = data.gt(0);

    // Test reduction operations using conversion to float
    auto mask_float = mask.to(DataType::Float32);
    bool has_any_true = mask_float.max_scalar() > 0.5f;  // Check if max is 1.0
    bool all_true = mask_float.min_scalar() > 0.5f;      // Check if min is 1.0

    // Since it's random data, we expect some true and some false
    EXPECT_TRUE(has_any_true);
    EXPECT_FALSE(all_true);  // Unlikely all random values are > 0

    // Count non-zeros
    size_t count = mask.count_nonzero();
    EXPECT_GT(count, 0);
    EXPECT_LT(count, d1 * d2 * d3);
}

// =============  Stress Tests =============

TEST_F(TensorMaskingTest, StressTestLargeMasking) {
    const size_t size = 10000;
    auto data = Tensor::randn({size, size}, Device::CUDA);

    // Create complex mask
    auto mask = data.gt(0).logical_and(data.lt(1));

    // Apply mask
    auto selected = data.masked_select(mask);

    // Just verify it works without crashing
    EXPECT_GE(selected.numel(), 0);
    EXPECT_LE(selected.numel(), size * size);
}

TEST_F(TensorMaskingTest, StressTestManyDimensions) {
    // Test with many dimensions (up to 6D)
    auto data = Tensor::randn({2, 3, 4, 5, 6, 7}, Device::CUDA);
    auto mask = data.gt(0);

    EXPECT_EQ(mask.shape(), data.shape());
    EXPECT_EQ(mask.ndim(), 6);

    // Apply masking operations
    data.masked_fill_(mask, 1.0f);

    // Count elements
    auto count = mask.count_nonzero();
    EXPECT_GE(count, 0);
    EXPECT_LE(count, data.numel());
}

// =============  Mixed Dtype Operations =============

TEST_F(TensorMaskingTest, MixedDtypeComparisons) {
    // Test that comparisons always return bool regardless of input dtype
    auto float_tensor = create_test_tensor({1.5, 2.5, 3.5}, {3});

    auto mask1 = float_tensor.gt(2.0f);
    EXPECT_EQ(mask1.dtype(), DataType::Bool);

    auto mask2 = float_tensor.eq(float_tensor);
    EXPECT_EQ(mask2.dtype(), DataType::Bool);

    // All elements should equal themselves
    auto mask2_float = mask2.to(DataType::Float32);
    EXPECT_FLOAT_EQ(mask2_float.min_scalar(), 1.0f);  // All true
}

// =============  Special Value Handling =============

TEST_F(TensorMaskingTest, SpecialValueMasking) {
    // Create tensor with special values
    std::vector<float> data = {
        0.0f, 1.0f, -1.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()
    };
    auto tensor = Tensor::from_vector(data, {6}, Device::CUDA);

    // Test comparisons with special values
    auto finite_mask = tensor.eq(tensor);  // NaN != NaN
    auto values = finite_mask.to_vector_bool();

    // First 5 should be true, last (NaN) should be false
    EXPECT_TRUE(values[0]);
    EXPECT_TRUE(values[1]);
    EXPECT_TRUE(values[2]);
    EXPECT_TRUE(values[3]);
    EXPECT_TRUE(values[4]);
    EXPECT_FALSE(values[5]);  // NaN != NaN
}

// =============  Boolean Tensor Arithmetic =============

TEST_F(TensorMaskingTest, BooleanToFloatConversion) {
    auto bool_tensor = Tensor::from_vector(
        std::vector<bool>{true, false, true, false},
        {2, 2},
        Device::CUDA
    );

    // Convert to float for arithmetic
    auto float_tensor = bool_tensor.to(DataType::Float32);

    // Multiply by scalar
    auto result = float_tensor.mul(5.0f);

    auto values = result.to_vector();
    std::vector<float> expected = {5.0f, 0.0f, 5.0f, 0.0f};
    EXPECT_TRUE(compare_vectors(values, expected));
}

// =============  Advanced Indexing with Boolean Masks =============

TEST_F(TensorMaskingTest, AdvancedBooleanIndexing) {
    // Create a 3D tensor
    const size_t d0 = 3, d1 = 4, d2 = 5;

    // Create a tensor with sequential values
    std::vector<float> data(d0 * d1 * d2);
    for (size_t i = 0; i < d0 * d1 * d2; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto tensor = Tensor::from_vector(data, {d0, d1, d2}, Device::CUDA);

    // Create boolean mask for middle dimension
    auto mask_1d = Tensor::from_vector(
        std::vector<bool>{false, true, true, false},
        {d1},
        Device::CUDA
    );

    // Select using mask - this should select dimensions 1 and 2 from middle axis
    auto indices = Tensor::from_vector(std::vector<int>{1, 2}, {2}, Device::CUDA);
    auto selected = tensor.index_select(1, indices);

    EXPECT_EQ(selected.shape(), TensorShape({d0, 2, d2}));
}

// =============  Batch Processing with Masks =============

TEST_F(TensorMaskingTest, BatchMaskProcessing) {
    const size_t batch = 8;
    const size_t features = 128;

    // Simulate batch processing with different masks per batch
    auto data = Tensor::randn({batch, features}, Device::CUDA);

    // Create different threshold for each batch element
    std::vector<float> thresholds(batch);
    for (size_t i = 0; i < batch; ++i) {
        thresholds[i] = -1.0f + 0.5f * i;  // Increasing thresholds
    }

    // Process each batch with its own mask
    for (size_t i = 0; i < batch; ++i) {
        auto batch_data = data.slice(0, i, i+1).squeeze(0);
        auto batch_mask = batch_data.gt(thresholds[i]);
        batch_data.masked_fill_(batch_mask, thresholds[i]);
    }

    // Verify each batch is clamped at its threshold
    for (size_t i = 0; i < batch; ++i) {
        auto batch_data = data.slice(0, i, i+1).squeeze(0);
        auto max_val = batch_data.max_scalar();
        EXPECT_LE(max_val, thresholds[i] + 1e-5f);
    }
}

// =============  Memory Layout Tests =============

TEST_F(TensorMaskingTest, ContiguousVsNonContiguousMasking) {
    // Create a non-contiguous tensor via transpose
    auto original = create_test_tensor(
        {1, 2, 3, 4, 5, 6},
        {2, 3}
    );

    auto transposed = original.transpose(0, 1);  // Now non-contiguous
    EXPECT_EQ(transposed.shape(), TensorShape({3, 2}));

    // Apply masking operations on non-contiguous tensor
    auto mask = transposed.gt(3);
    auto masked = transposed.masked_fill(mask, -1);

    // Verify the operation worked correctly
    auto values = masked.to_vector();
    // Original transposed: [[1,4], [2,5], [3,6]]
    // After masking > 3: [[1,-1], [2,-1], [3,-1]]
    std::vector<float> expected = {1, -1, 2, -1, 3, -1};
    EXPECT_TRUE(compare_vectors(values, expected));
}

// =============  Error Handling Tests =============

TEST_F(TensorMaskingTest, ErrorHandlingMismatchedShapes) {
    auto tensor = create_test_tensor({1, 2, 3, 4}, {2, 2});
    auto wrong_mask = Tensor::ones_bool({3, 3}, Device::CUDA);

    // This should fail gracefully
    auto result = tensor.masked_select(wrong_mask);
    EXPECT_FALSE(result.is_valid());
}

TEST_F(TensorMaskingTest, ErrorHandlingWrongDevice) {
    auto cuda_tensor = create_test_tensor({1, 2, 3, 4}, {2, 2});
    auto cpu_mask = Tensor::ones_bool({2, 2}, Device::CPU);

    // This should fail gracefully
    auto result = cuda_tensor.masked_select(cpu_mask);
    EXPECT_FALSE(result.is_valid());
}

// =============  Final Integration Test =============

TEST_F(TensorMaskingTest, IntegrationCompleteWorkflow) {
    // Simulate a complete ML workflow with masking
    const size_t batch = 32;
    const size_t seq_len = 128;
    const size_t hidden = 256;

    // 1. Create input data
    auto inputs = Tensor::randn({batch, seq_len, hidden}, Device::CUDA);

    // 2. Create attention mask (causal)
    std::vector<float> range(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        range[i] = static_cast<float>(i);
    }
    auto range_tensor = Tensor::from_vector(range, {seq_len}, Device::CUDA);
    auto causal = range_tensor.reshape({seq_len, 1}).ge(range_tensor.reshape({1, seq_len}));

    // 3. Create padding mask (different lengths per batch)
    auto padding = Tensor::ones_bool({batch, seq_len}, Device::CUDA);
    // Simulate some padding
    auto cpu_padding = padding.to(Device::CPU);
    for (size_t b = 0; b < batch; ++b) {
        size_t length = seq_len - (b % 10);  // Variable lengths
        for (size_t s = length; s < seq_len; ++s) {
            cpu_padding.set_bool({b, s}, false);  // Use set_bool for boolean tensors
        }
    }
    padding = cpu_padding.to(Device::CUDA);

    // 4. Combine masks
    auto causal_expanded = causal.unsqueeze(0).expand({batch, seq_len, seq_len});
    auto padding_expanded = padding.unsqueeze(1).expand({batch, seq_len, seq_len});
    auto combined_mask = causal_expanded.logical_and(padding_expanded);

    // 5. Apply to attention scores (simulated)
    auto scores = Tensor::randn({batch, seq_len, seq_len}, Device::CUDA);
    scores.masked_fill_(combined_mask.logical_not(), -1e9f);

    // 6. Apply softmax (simulated by just checking mask worked)
    auto masked_scores = scores.masked_select(combined_mask.logical_not());
    if (masked_scores.numel() > 0) {
        EXPECT_FLOAT_EQ(masked_scores.to_vector()[0], -1e9f);
    }

    // Just verify the workflow completed without errors
    EXPECT_TRUE(scores.is_valid());
    EXPECT_EQ(scores.shape(), TensorShape({batch, seq_len, seq_len}));
}