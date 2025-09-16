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
    EXPECT_TRUE(selected.is_valid());
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
    EXPECT_TRUE(all_true.all());
    EXPECT_TRUE(all_true.any());

    auto all_false = Tensor::zeros_bool({3, 3}, Device::CUDA);
    EXPECT_FALSE(all_false.all());
    EXPECT_FALSE(all_false.any());

    std::vector<bool> mixed_data = {true, false, true};
    auto mixed = Tensor::from_vector(mixed_data, {3}, Device::CUDA);
    EXPECT_FALSE(mixed.all());
    EXPECT_TRUE(mixed.any());
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

// ============= Edge Cases =============
TEST_F(TensorMaskingTest, EmptyMask) {
    auto tensor = create_test_tensor({1, 2, 3}, {3});
    auto empty_mask = Tensor::zeros_bool({3}, Device::CUDA);

    auto selected = tensor.masked_select(empty_mask);
    EXPECT_EQ(selected.numel(), 0);
    EXPECT_TRUE(selected.is_valid());
}

TEST_F(TensorMaskingTest, FullMask) {
    auto tensor = create_test_tensor({1, 2, 3}, {3});
    auto full_mask = Tensor::ones_bool({3}, Device::CUDA);

    auto selected = tensor.masked_select(full_mask);
    EXPECT_EQ(selected.numel(), 3);

    auto values = selected.to_vector();
    std::vector<float> expected = {1, 2, 3};
    EXPECT_TRUE(compare_vectors(values, expected));
}

TEST_F(TensorMaskingTest, SingleElementIndexing) {
    auto tensor = create_test_tensor({42}, {1});

    std::vector<int> idx_data = {0};
    auto indices = Tensor::from_vector(idx_data, {1}, Device::CUDA);

    auto selected = tensor.index_select(0, indices);
    EXPECT_EQ(selected.numel(), 1);
    EXPECT_FLOAT_EQ(selected.item(), 42.0f);
}
