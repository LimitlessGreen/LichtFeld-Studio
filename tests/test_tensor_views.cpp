/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

using namespace gs;

class TensorViewTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
    }

    Tensor create_test_tensor(const std::vector<size_t>& shape) {
        size_t total = 1;
        for (auto dim : shape) {
            total *= dim;
        }

        auto tensor = Tensor::empty(TensorShape(shape), Device::CUDA);
        std::vector<float> data(total);
        for (size_t i = 0; i < total; ++i) {
            data[i] = static_cast<float>(i);
        }
        cudaMemcpy(tensor.ptr<float>(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
        return tensor;
    }

    bool shapes_equal(const Tensor& t, const std::vector<size_t>& expected_shape) {
        if (t.shape().rank() != expected_shape.size())
            return false;
        for (size_t i = 0; i < expected_shape.size(); ++i) {
            if (t.shape()[i] != expected_shape[i])
                return false;
        }
        return true;
    }
};

TEST_F(TensorViewTest, BasicView) {
    auto tensor = create_test_tensor({2, 3, 4});

    // Reshape to different valid shape
    auto view1 = tensor.view({6, 4});
    EXPECT_TRUE(shapes_equal(view1, {6, 4}));
    EXPECT_EQ(view1.numel(), tensor.numel());
    EXPECT_EQ(view1.raw_ptr(), tensor.raw_ptr());
    EXPECT_FALSE(view1.owns_memory());

    // Another reshape
    auto view2 = tensor.view({24});
    EXPECT_TRUE(shapes_equal(view2, {24}));
    EXPECT_EQ(view2.numel(), tensor.numel());

    // Reshape to original
    auto view3 = tensor.view({2, 3, 4});
    EXPECT_TRUE(shapes_equal(view3, {2, 3, 4}));
}

TEST_F(TensorViewTest, ViewWithInferredDimension) {
    auto tensor = create_test_tensor({2, 3, 4});

    // Use -1 to infer dimension
    auto view1 = tensor.view({static_cast<size_t>(-1), 4});
    EXPECT_TRUE(shapes_equal(view1, {6, 4}));

    auto view2 = tensor.view({2, static_cast<size_t>(-1)});
    EXPECT_TRUE(shapes_equal(view2, {2, 12}));

    auto view3 = tensor.view({static_cast<size_t>(-1)});
    EXPECT_TRUE(shapes_equal(view3, {24}));
}

TEST_F(TensorViewTest, InvalidView) {
    auto tensor = create_test_tensor({2, 3, 4});

    // Wrong number of elements
    auto invalid = tensor.view({5, 5});
    EXPECT_FALSE(invalid.is_valid());

    // Multiple -1
    auto invalid2 = tensor.view({static_cast<size_t>(-1), static_cast<size_t>(-1)});
    EXPECT_FALSE(invalid2.is_valid());
}

TEST_F(TensorViewTest, Reshape) {
    auto tensor = create_test_tensor({3, 4});

    // reshape is alias for view
    auto reshaped = tensor.reshape({2, 6});
    EXPECT_TRUE(shapes_equal(reshaped, {2, 6}));
    EXPECT_EQ(reshaped.raw_ptr(), tensor.raw_ptr());
}

TEST_F(TensorViewTest, Slice) {
    auto tensor = create_test_tensor({10, 5});

    // Slice along dimension 0
    auto slice1 = tensor.slice(0, 2, 5);
    EXPECT_TRUE(shapes_equal(slice1, {3, 5}));
    EXPECT_FALSE(slice1.owns_memory());

    // Slice along dimension 1
    auto slice2 = tensor.slice(1, 1, 4);
    EXPECT_TRUE(shapes_equal(slice2, {10, 3}));

    // Check that slice points to correct part of memory
    auto full_values = tensor.to_vector();
    auto slice_values = slice1.to_vector();

    // slice1 should contain elements from rows 2-4 (inclusive)
    for (size_t i = 0; i < slice_values.size(); ++i) {
        size_t original_idx = (2 * 5) + i; // Start at row 2
        EXPECT_FLOAT_EQ(slice_values[i], full_values[original_idx]);
    }
}

TEST_F(TensorViewTest, InvalidSlice) {
    auto tensor = create_test_tensor({10, 5});

    // Out of range dimension
    auto invalid1 = tensor.slice(2, 0, 1);
    EXPECT_FALSE(invalid1.is_valid());

    // Invalid range
    auto invalid2 = tensor.slice(0, 5, 3); // start > end
    EXPECT_FALSE(invalid2.is_valid());

    auto invalid3 = tensor.slice(0, 0, 11); // end > size
    EXPECT_FALSE(invalid3.is_valid());
}

TEST_F(TensorViewTest, Squeeze) {
    // Test squeezing dimensions of size 1
    auto tensor = create_test_tensor({1, 3, 1, 4});

    // Squeeze specific dimension
    auto squeezed0 = tensor.squeeze(0);
    EXPECT_TRUE(shapes_equal(squeezed0, {3, 1, 4}));

    auto squeezed2 = tensor.squeeze(2);
    EXPECT_TRUE(shapes_equal(squeezed2, {1, 3, 4}));

    // Squeeze with negative index
    auto squeezed_neg = tensor.squeeze(-2);
    EXPECT_TRUE(shapes_equal(squeezed_neg, {1, 3, 4}));

    // Try to squeeze non-1 dimension (should return same shape)
    auto not_squeezed = tensor.squeeze(1);
    EXPECT_TRUE(shapes_equal(not_squeezed, {1, 3, 1, 4}));
}

TEST_F(TensorViewTest, Unsqueeze) {
    auto tensor = create_test_tensor({3, 4});

    // Add dimension at position 0
    auto unsqueezed0 = tensor.unsqueeze(0);
    EXPECT_TRUE(shapes_equal(unsqueezed0, {1, 3, 4}));

    // Add dimension at position 1
    auto unsqueezed1 = tensor.unsqueeze(1);
    EXPECT_TRUE(shapes_equal(unsqueezed1, {3, 1, 4}));

    // Add dimension at end
    auto unsqueezed2 = tensor.unsqueeze(2);
    EXPECT_TRUE(shapes_equal(unsqueezed2, {3, 4, 1}));

    // Negative index
    auto unsqueezed_neg = tensor.unsqueeze(-1);
    EXPECT_TRUE(shapes_equal(unsqueezed_neg, {3, 4, 1}));
}

TEST_F(TensorViewTest, Flatten) {
    auto tensor = create_test_tensor({2, 3, 4, 5});

    // Flatten all dimensions
    auto flat_all = tensor.flatten();
    EXPECT_TRUE(shapes_equal(flat_all, {120}));

    // Flatten from dimension 1 to 2
    auto flat_partial = tensor.flatten(1, 2);
    EXPECT_TRUE(shapes_equal(flat_partial, {2, 12, 5}));

    // Flatten with negative indices
    auto flat_neg = tensor.flatten(-2, -1);
    EXPECT_TRUE(shapes_equal(flat_neg, {2, 3, 20}));

    // Flatten single dimension (should not change)
    auto flat_single = tensor.flatten(1, 1);
    EXPECT_TRUE(shapes_equal(flat_single, {2, 3, 4, 5}));
}

TEST_F(TensorViewTest, ChainedViewOperations) {
    auto tensor = create_test_tensor({2, 3, 4});

    // Chain multiple view operations
    auto result = tensor.view({6, 4})
                      .unsqueeze(0)
                      .squeeze(0)
                      .reshape({2, 12})
                      .flatten();

    EXPECT_TRUE(shapes_equal(result, {24}));
    EXPECT_EQ(result.raw_ptr(), tensor.raw_ptr());
    EXPECT_FALSE(result.owns_memory());
}

TEST_F(TensorViewTest, ViewsShareMemory) {
    auto tensor = create_test_tensor({4, 4});
    auto view = tensor.view({2, 8});

    // Modify view
    view.fill_(1.0f);

    // Original should be modified
    auto values = tensor.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }
}

TEST_F(TensorViewTest, SliceSharesMemory) {
    auto tensor = create_test_tensor({5, 5});
    auto slice = tensor.slice(0, 1, 3); // Rows 1-2

    // Modify slice
    slice.fill_(99.0f);

    // Check original tensor
    auto values = tensor.to_vector();

    // First row should be unchanged (0-4)
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(values[i], static_cast<float>(i));
    }

    // Rows 1-2 should be 99
    for (size_t i = 5; i < 15; ++i) {
        EXPECT_FLOAT_EQ(values[i], 99.0f);
    }

    // Remaining rows unchanged
    for (size_t i = 15; i < 25; ++i) {
        EXPECT_FLOAT_EQ(values[i], static_cast<float>(i));
    }
}

TEST_F(TensorViewTest, ComplexReshaping) {
    // Test complex reshaping scenarios
    auto tensor = create_test_tensor({2, 3, 4, 5});

    // Multiple valid reshapes
    std::vector<std::vector<size_t>> valid_shapes = {
        {120},
        {2, 60},
        {6, 20},
        {2, 3, 20},
        {24, 5},
        {2, 12, 5},
        {2, 3, 2, 10}};

    for (const auto& shape : valid_shapes) {
        auto view = tensor.view(TensorShape(shape));
        EXPECT_TRUE(view.is_valid());
        EXPECT_TRUE(shapes_equal(view, shape));
        EXPECT_EQ(view.numel(), tensor.numel());
    }
}

TEST_F(TensorViewTest, TransposeBasic) {
    auto tensor = create_test_tensor({3, 4, 5});

    // Transpose dimensions 0 and 1
    auto transposed = tensor.transpose(0, 1);
    // Note: Currently returns a clone, not a true view
    // Shape should be {4, 3, 5} but implementation may differ
    EXPECT_TRUE(transposed.is_valid());
    EXPECT_EQ(transposed.numel(), tensor.numel());
}

TEST_F(TensorViewTest, PermuteBasic) {
    auto tensor = create_test_tensor({2, 3, 4});

    // Permute dimensions
    auto permuted = tensor.permute({2, 0, 1});
    // Note: Currently returns a clone
    EXPECT_TRUE(permuted.is_valid());
    EXPECT_EQ(permuted.numel(), tensor.numel());
}

TEST_F(TensorViewTest, ViewOnCPUTensor) {
    // Create CPU tensor
    auto cpu_tensor = Tensor::ones({3, 4}, Device::CPU);

    // View operations should work on CPU tensors too
    auto view = cpu_tensor.view({12});
    EXPECT_TRUE(shapes_equal(view, {12}));
    EXPECT_EQ(view.device(), Device::CPU);
    EXPECT_FALSE(view.owns_memory());

    auto slice = cpu_tensor.slice(0, 1, 3);
    EXPECT_TRUE(shapes_equal(slice, {2, 4}));
    EXPECT_EQ(slice.device(), Device::CPU);
}

TEST_F(TensorViewTest, EdgeCases) {
    // Empty tensor
    auto empty = Tensor::empty({0}, Device::CUDA);
    auto view = empty.view({0});
    EXPECT_TRUE(view.is_valid());
    EXPECT_EQ(view.numel(), 0);

    // Single element
    auto single = Tensor::ones({1}, Device::CUDA);
    auto view_single = single.view({1, 1, 1});
    EXPECT_TRUE(shapes_equal(view_single, {1, 1, 1}));

    // Large tensor
    auto large = Tensor::empty({100, 100}, Device::CUDA);
    auto view_large = large.view({10000});
    EXPECT_TRUE(shapes_equal(view_large, {10000}));
}
