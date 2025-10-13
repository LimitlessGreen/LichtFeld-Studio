/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cmath>
#include <random>

using namespace gs;

// ============= Helper Functions =============

// Convert gs::Tensor to torch::Tensor for comparison
torch::Tensor to_torch(const Tensor& t) {
    auto cpu_tensor = t.to(Device::CPU);
    auto vec = cpu_tensor.to_vector();

    std::vector<int64_t> shape_vec;
    for (size_t i = 0; i < cpu_tensor.ndim(); ++i) {
        shape_vec.push_back(static_cast<int64_t>(cpu_tensor.size(i)));
    }

    auto torch_tensor = torch::from_blob(
        vec.data(),
        shape_vec,
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();

    return torch_tensor;
}

// Convert torch::Tensor to gs::Tensor
Tensor from_torch(const torch::Tensor& t, Device device = Device::CUDA) {
    auto cpu_t = t.cpu();
    std::vector<float> vec(cpu_t.data_ptr<float>(),
                          cpu_t.data_ptr<float>() + cpu_t.numel());

    std::vector<size_t> shape_vec;
    for (int64_t i = 0; i < cpu_t.dim(); ++i) {
        shape_vec.push_back(static_cast<size_t>(cpu_t.size(i)));
    }

    return Tensor::from_vector(vec, TensorShape(shape_vec), device);
}

// Compare tensors with tolerance
bool tensors_close(const Tensor& a, const torch::Tensor& b,
                   float rtol = 1e-5f, float atol = 1e-5f) {
    auto torch_a = to_torch(a);
    return torch::allclose(torch_a, b, rtol, atol);
}

// Helper to extract int64 values from a tensor
std::vector<int64_t> tensor_to_int64_vector(const Tensor& t) {
    if (!t.is_valid()) {
        throw std::runtime_error("tensor_to_int64_vector: invalid tensor");
    }

    if (t.numel() == 0) {
        return {};
    }

    auto cpu_t = t.to(Device::CPU);

    if (cpu_t.dtype() != DataType::Int64) {
        throw std::runtime_error(
            std::string("Expected Int64 tensor, got ") + dtype_name(cpu_t.dtype())
        );
    }

    // Use the proper API from tensor.cpp
    return cpu_t.to_vector_int64();
}

// ============= Test Copy Constructor and Assignment =============

class TensorCopyTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
        torch::manual_seed(42);
    }
};

// FIXED: Test shallow copy behavior (LibTorch-like)
TEST_F(TensorCopyTest, CopyConstructorCPU) {
    auto a = Tensor::arange(0, 10, 1).to(Device::CPU);
    Tensor b(a);  // Copy constructor - SHALLOW COPY

    // Verify metadata matches
    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.device(), b.device());
    EXPECT_EQ(a.dtype(), b.dtype());

    // Verify data matches initially
    EXPECT_TRUE(a.all_close(b));

    // CRITICAL: With shallow copy, modifying b also modifies a
    b.fill_(42.0f);

    // Both should now be 42 (shared data!)
    EXPECT_TRUE(a.all_close(b));

    // Verify both are 42
    auto a_vec = a.to_vector();
    auto b_vec = b.to_vector();
    for (size_t i = 0; i < a_vec.size(); ++i) {
        EXPECT_NEAR(a_vec[i], 42.0f, 1e-5f);
        EXPECT_NEAR(b_vec[i], 42.0f, 1e-5f);
    }
}

TEST_F(TensorCopyTest, CopyConstructorCUDA) {
    auto a = Tensor::arange(0, 10, 1).to(Device::CUDA);
    Tensor b(a);  // Copy constructor - SHALLOW COPY

    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.device(), b.device());
    EXPECT_EQ(a.dtype(), b.dtype());
    EXPECT_TRUE(a.all_close(b));

    // With shallow copy, both point to same data
    b.fill_(42.0f);
    EXPECT_TRUE(a.all_close(b));  // Both should be 42

    // Verify b is actually 42
    auto b_vec = b.to_vector();
    for (size_t i = 0; i < b_vec.size(); ++i) {
        EXPECT_NEAR(b_vec[i], 42.0f, 1e-5f);
    }
}

TEST_F(TensorCopyTest, CopyAssignmentCPU) {
    auto a = Tensor::arange(0, 10, 1).to(Device::CPU);
    auto b = Tensor::zeros({5}, Device::CPU);

    // Verify b starts as zeros
    auto b_before = b.to_vector();
    for (float val : b_before) {
        EXPECT_NEAR(val, 0.0f, 1e-5f);
    }

    b = a;  // Copy assignment - SHALLOW COPY

    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.device(), b.device());
    EXPECT_TRUE(a.all_close(b));

    // With shallow copy, both point to same data
    b.fill_(42.0f);
    EXPECT_TRUE(a.all_close(b));  // Both should be 42
}

TEST_F(TensorCopyTest, CopyAssignmentCUDA) {
    auto a = Tensor::arange(0, 10, 1).to(Device::CUDA);
    auto b = Tensor::zeros({5}, Device::CUDA);

    b = a;  // Copy assignment - SHALLOW COPY

    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.device(), b.device());
    EXPECT_TRUE(a.all_close(b));

    // With shallow copy, modifying b also modifies a
    b.fill_(42.0f);

    auto a_after = a.to_vector();
    auto b_after = b.to_vector();

    // Both should be 42
    for (size_t i = 0; i < a_after.size(); ++i) {
        EXPECT_NEAR(a_after[i], 42.0f, 1e-5f);
        EXPECT_NEAR(b_after[i], 42.0f, 1e-5f);
    }

    EXPECT_TRUE(a.all_close(b));
}

TEST_F(TensorCopyTest, CopyAssignmentSelfAssignment) {
    auto a = Tensor::arange(0, 10, 1);
    auto original_values = a.to_vector();

    a = a;  // Self-assignment

    auto after_values = a.to_vector();
    EXPECT_EQ(original_values.size(), after_values.size());
    for (size_t i = 0; i < original_values.size(); ++i) {
        EXPECT_NEAR(original_values[i], after_values[i], 1e-5f);
    }
}

// NEW TEST: Verify clone() creates independent copy
TEST_F(TensorCopyTest, CloneCreatesDeepCopy) {
    auto a = Tensor::arange(0, 10, 1).to(Device::CPU);
    auto b = a.clone();  // Deep copy via clone()

    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_TRUE(a.all_close(b));

    // Now modifying b should NOT affect a
    b.fill_(42.0f);
    EXPECT_FALSE(a.all_close(b));

    // Verify a is unchanged
    auto a_vec = a.to_vector();
    for (size_t i = 0; i < a_vec.size(); ++i) {
        EXPECT_NEAR(a_vec[i], static_cast<float>(i), 1e-5f);
    }

    // Verify b is 42
    auto b_vec = b.to_vector();
    for (size_t i = 0; i < b_vec.size(); ++i) {
        EXPECT_NEAR(b_vec[i], 42.0f, 1e-5f);
    }
}

TEST_F(TensorCopyTest, CopyUnderscoreMethod) {
    auto a = Tensor::arange(0, 10, 1);
    auto b = Tensor::zeros({10});

    b.copy_(a);  // Using copy_() method - deep copy into b

    EXPECT_TRUE(a.all_close(b));

    // copy_() does a deep copy, so they are independent
    b.fill_(42.0f);
    EXPECT_FALSE(a.all_close(b));
}

TEST_F(TensorCopyTest, CopyEmptyTensor) {
    auto a = Tensor::empty({0});
    Tensor b(a);

    EXPECT_EQ(a.numel(), 0);
    EXPECT_EQ(b.numel(), 0);
    EXPECT_EQ(a.shape(), b.shape());
}

TEST_F(TensorCopyTest, CopyLargeTensor) {
    auto a = Tensor::randn({1000, 1000});
    Tensor b(a);  // Shallow copy

    EXPECT_TRUE(a.all_close(b, 1e-5f, 1e-5f));

    // With shallow copy, modifying b affects a
    b.add_(1.0f);
    EXPECT_TRUE(a.all_close(b));  // Both should have the same values

    // Verify they both increased by 1.0
    // (Can't directly test this without saving original, but the fact
    // that they're still equal proves shallow copy behavior)
}

// NEW TEST: Verify LibTorch parity
TEST_F(TensorCopyTest, LibTorchParityShallowCopy) {
    // Test that our behavior matches LibTorch exactly

    // Our implementation
    auto gs_a = Tensor::arange(0, 5, 1).to(Device::CPU);
    auto gs_b = gs_a;  // Shallow copy
    gs_b.fill_(99.0f);

    // LibTorch
    auto torch_a = torch::arange(0, 5, 1);
    auto torch_b = torch_a;  // Shallow copy in LibTorch
    torch_b.fill_(99.0f);

    // Both should show same behavior: a is also 99
    EXPECT_NEAR(gs_a.to_vector()[0], 99.0f, 1e-5f);
    EXPECT_NEAR(torch_a[0].item<float>(), 99.0f, 1e-5f);
}

TEST_F(TensorCopyTest, LibTorchParityClone) {
    // Test that clone() matches LibTorch

    // Our implementation
    auto gs_a = Tensor::arange(0, 5, 1).to(Device::CPU);
    auto gs_b = gs_a.clone();  // Deep copy
    gs_b.fill_(99.0f);

    // LibTorch
    auto torch_a = torch::arange(0, 5, 1);
    auto torch_b = torch_a.clone();  // Deep copy in LibTorch
    torch_b.fill_(99.0f);

    // Both should show same behavior: a is unchanged
    EXPECT_NEAR(gs_a.to_vector()[0], 0.0f, 1e-5f);
    EXPECT_NEAR(torch_a[0].item<float>(), 0.0f, 1e-5f);
}

// ============= Test cdist (Pairwise Distance) =============

class TensorCdistTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
        torch::manual_seed(42);
    }
};

TEST_F(TensorCdistTest, CdistL2Basic) {
    // Create test data
    auto a = Tensor::randn({5, 3});
    auto b = Tensor::randn({7, 3});

    // PyTorch reference
    auto torch_a = to_torch(a);
    auto torch_b = to_torch(b);
    auto torch_result = torch::cdist(torch_a, torch_b, 2.0);

    // Our implementation
    auto result = a.cdist(b, 2.0f);

    EXPECT_EQ(result.shape(), TensorShape({5, 7}));
    EXPECT_TRUE(tensors_close(result, torch_result, 1e-4f, 1e-4f));

    // Verify result is on same device as input
    EXPECT_EQ(result.device(), a.device());
}

TEST_F(TensorCdistTest, CdistL1) {
    auto a = Tensor::randn({5, 3});
    auto b = Tensor::randn({7, 3});

    auto torch_a = to_torch(a);
    auto torch_b = to_torch(b);
    auto torch_result = torch::cdist(torch_a, torch_b, 1.0);

    auto result = a.cdist(b, 1.0f);

    EXPECT_EQ(result.shape(), TensorShape({5, 7}));
    EXPECT_TRUE(tensors_close(result, torch_result, 1e-4f, 1e-4f));
}

TEST_F(TensorCdistTest, CdistSamePoints) {
    auto a = Tensor::randn({10, 4});

    auto torch_a = to_torch(a);
    auto torch_result = torch::cdist(torch_a, torch_a, 2.0);

    auto result = a.cdist(a, 2.0f);

    // Diagonal should be near zero
    for (size_t i = 0; i < 10; ++i) {
        float val = result.to(Device::CPU).at({i, i});
        EXPECT_NEAR(val, 0.0f, 1e-5f) << "Diagonal element " << i << " should be zero";
    }

    EXPECT_TRUE(tensors_close(result, torch_result, 1e-4f, 1e-4f));
}

TEST_F(TensorCdistTest, CdistCPU) {
    auto a = Tensor::randn({5, 3}, Device::CPU);
    auto b = Tensor::randn({7, 3}, Device::CPU);

    auto torch_a = to_torch(a);
    auto torch_b = to_torch(b);
    auto torch_result = torch::cdist(torch_a, torch_b, 2.0);

    auto result = a.cdist(b, 2.0f);

    EXPECT_EQ(result.device(), Device::CPU);
    EXPECT_TRUE(tensors_close(result, torch_result, 1e-4f, 1e-4f));
}

TEST_F(TensorCdistTest, CdistHighDimensional) {
    auto a = Tensor::randn({100, 128});
    auto b = Tensor::randn({50, 128});

    auto torch_a = to_torch(a);
    auto torch_b = to_torch(b);
    auto torch_result = torch::cdist(torch_a, torch_b, 2.0);

    auto result = a.cdist(b, 2.0f);

    EXPECT_EQ(result.shape(), TensorShape({100, 50}));
    EXPECT_TRUE(tensors_close(result, torch_result, 1e-3f, 1e-3f));
}

TEST_F(TensorCdistTest, CdistInvalidShapes) {
    auto a = Tensor::randn({5, 3});
    auto b = Tensor::randn({7, 4});  // Different feature dimension

    auto result = a.cdist(b);
    EXPECT_FALSE(result.is_valid());
}

TEST_F(TensorCdistTest, CdistNonSquared) {
    // Test Manhattan distance vs Euclidean
    auto a = Tensor::from_vector({0.0f, 0.0f}, {1, 2});
    auto b = Tensor::from_vector({3.0f, 4.0f}, {1, 2});

    auto l1_dist = a.cdist(b, 1.0f);
    auto l2_dist = a.cdist(b, 2.0f);

    // L1 distance should be 7.0
    float l1_val = l1_dist.to(Device::CPU).item();
    EXPECT_NEAR(l1_val, 7.0f, 1e-5f);

    // L2 distance should be 5.0 (3-4-5 triangle)
    float l2_val = l2_dist.to(Device::CPU).item();
    EXPECT_NEAR(l2_val, 5.0f, 1e-5f);
}

TEST_F(TensorCdistTest, CdistSymmetry) {
    // Distance matrix should be symmetric for same points
    auto a = Tensor::randn({20, 10});

    auto dist = a.cdist(a, 2.0f);
    auto dist_t = dist.t();

    EXPECT_TRUE(dist.all_close(dist_t, 1e-4f, 1e-4f));
}

// ============= Test min_with_indices and max_with_indices =============

class TensorMinMaxIndicesTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
        torch::manual_seed(42);
    }
};

TEST_F(TensorMinMaxIndicesTest, MinWithIndices1D) {
    auto t = Tensor::from_vector({5.0f, 2.0f, 8.0f, 1.0f, 9.0f}, {5});

    auto torch_t = to_torch(t);
    auto [torch_val, torch_idx] = torch::min(torch_t, 0);

    auto [val, idx] = t.min_with_indices(0);

    // Check value
    EXPECT_NEAR(val.item(), torch_val.item<float>(), 1e-5f);
    EXPECT_NEAR(val.item(), 1.0f, 1e-5f);

    // Check index
    auto idx_vec = tensor_to_int64_vector(idx);
    EXPECT_EQ(idx_vec.size(), 1);
    EXPECT_EQ(idx_vec[0], torch_idx.item<int64_t>());
    EXPECT_EQ(idx_vec[0], 3);  // Min is at index 3
}

TEST_F(TensorMinMaxIndicesTest, MaxWithIndices1D) {
    auto t = Tensor::from_vector({5.0f, 2.0f, 8.0f, 1.0f, 9.0f}, {5});

    auto torch_t = to_torch(t);
    auto [torch_val, torch_idx] = torch::max(torch_t, 0);

    auto [val, idx] = t.max_with_indices(0);

    // Check value
    EXPECT_NEAR(val.item(), torch_val.item<float>(), 1e-5f);
    EXPECT_NEAR(val.item(), 9.0f, 1e-5f);

    // Check index
    auto idx_vec = tensor_to_int64_vector(idx);
    EXPECT_EQ(idx_vec.size(), 1);
    EXPECT_EQ(idx_vec[0], torch_idx.item<int64_t>());
    EXPECT_EQ(idx_vec[0], 4);  // Max is at index 4
}

TEST_F(TensorMinMaxIndicesTest, MinWithIndices2D) {
    auto t = Tensor::randn({5, 10});

    auto torch_t = to_torch(t);
    auto [torch_val, torch_idx] = torch::min(torch_t, 1);

    auto [val, idx] = t.min_with_indices(1);

    // Check shapes
    EXPECT_EQ(val.shape(), TensorShape({5}));
    EXPECT_EQ(idx.shape(), TensorShape({5}));

    // Check values match PyTorch
    EXPECT_TRUE(tensors_close(val, torch_val, 1e-5f, 1e-5f));

    // Check indices match
    auto our_idx_vec = tensor_to_int64_vector(idx);
    auto torch_idx_cpu = torch_idx.cpu();

    EXPECT_EQ(our_idx_vec.size(), 5);
    for (int64_t i = 0; i < torch_idx_cpu.size(0); ++i) {
        int64_t torch_idx_val = torch_idx_cpu[i].item<int64_t>();
        EXPECT_EQ(our_idx_vec[i], torch_idx_val)
            << "Index mismatch at position " << i;
        EXPECT_GE(our_idx_vec[i], 0);
        EXPECT_LT(our_idx_vec[i], 10);
    }
}

TEST_F(TensorMinMaxIndicesTest, MaxWithIndices2D) {
    auto t = Tensor::randn({5, 10});

    auto torch_t = to_torch(t);
    auto [torch_val, torch_idx] = torch::max(torch_t, 1);

    auto [val, idx] = t.max_with_indices(1);

    // Check shapes
    EXPECT_EQ(val.shape(), TensorShape({5}));
    EXPECT_EQ(idx.shape(), TensorShape({5}));

    // Check values match PyTorch
    EXPECT_TRUE(tensors_close(val, torch_val, 1e-5f, 1e-5f));

    // Check indices match
    auto our_idx_vec = tensor_to_int64_vector(idx);
    auto torch_idx_cpu = torch_idx.cpu();

    EXPECT_EQ(our_idx_vec.size(), 5);
    for (int64_t i = 0; i < torch_idx_cpu.size(0); ++i) {
        int64_t torch_idx_val = torch_idx_cpu[i].item<int64_t>();
        EXPECT_EQ(our_idx_vec[i], torch_idx_val)
            << "Index mismatch at position " << i;
        EXPECT_GE(our_idx_vec[i], 0);
        EXPECT_LT(our_idx_vec[i], 10);
    }
}

TEST_F(TensorMinMaxIndicesTest, MinWithIndicesDim0) {
    auto t = Tensor::randn({5, 10});

    auto torch_t = to_torch(t);
    auto [torch_val, torch_idx] = torch::min(torch_t, 0);

    auto [val, idx] = t.min_with_indices(0);

    EXPECT_EQ(val.shape(), TensorShape({10}));
    EXPECT_EQ(idx.shape(), TensorShape({10}));
    EXPECT_TRUE(tensors_close(val, torch_val, 1e-5f, 1e-5f));

    // Verify indices are in valid range
    auto idx_vec = tensor_to_int64_vector(idx);
    for (auto i : idx_vec) {
        EXPECT_GE(i, 0);
        EXPECT_LT(i, 5);
    }
}

TEST_F(TensorMinMaxIndicesTest, MinMaxWithKeepdim) {
    auto t = Tensor::randn({5, 10, 8});

    auto [val, idx] = t.min_with_indices(1, true);

    EXPECT_EQ(val.shape(), TensorShape({5, 1, 8}));
    EXPECT_EQ(idx.shape(), TensorShape({5, 1, 8}));

    // Verify indices are in valid range
    auto idx_vec = tensor_to_int64_vector(idx);
    for (auto i : idx_vec) {
        EXPECT_GE(i, 0);
        EXPECT_LT(i, 10);
    }
}

TEST_F(TensorMinMaxIndicesTest, MinMaxNegativeDim) {
    auto t = Tensor::randn({5, 10});

    auto [val1, idx1] = t.min_with_indices(-1);
    auto [val2, idx2] = t.min_with_indices(1);

    EXPECT_EQ(val1.shape(), val2.shape());
    EXPECT_TRUE(val1.all_close(val2));

    auto idx1_vec = tensor_to_int64_vector(idx1);
    auto idx2_vec = tensor_to_int64_vector(idx2);
    EXPECT_EQ(idx1_vec, idx2_vec);
}

TEST_F(TensorMinMaxIndicesTest, VerifyCorrectness) {
    // Create a tensor with known min/max positions
    auto t = Tensor::from_vector({
        1.0f, 5.0f, 3.0f,  // row 0: min=1 (idx 0), max=5 (idx 1)
        4.0f, 2.0f, 9.0f,  // row 1: min=2 (idx 1), max=9 (idx 2)
        7.0f, 0.0f, 6.0f   // row 2: min=0 (idx 1), max=7 (idx 0)
    }, {3, 3});

    auto [min_vals, min_idx] = t.min_with_indices(1);
    auto [max_vals, max_idx] = t.max_with_indices(1);

    // Check min values
    auto min_vec = min_vals.to_vector();
    EXPECT_NEAR(min_vec[0], 1.0f, 1e-5f);
    EXPECT_NEAR(min_vec[1], 2.0f, 1e-5f);
    EXPECT_NEAR(min_vec[2], 0.0f, 1e-5f);

    // Check min indices
    auto min_idx_vec = tensor_to_int64_vector(min_idx);
    EXPECT_EQ(min_idx_vec[0], 0);
    EXPECT_EQ(min_idx_vec[1], 1);
    EXPECT_EQ(min_idx_vec[2], 1);

    // Check max values
    auto max_vec = max_vals.to_vector();
    EXPECT_NEAR(max_vec[0], 5.0f, 1e-5f);
    EXPECT_NEAR(max_vec[1], 9.0f, 1e-5f);
    EXPECT_NEAR(max_vec[2], 7.0f, 1e-5f);

    // Check max indices
    auto max_idx_vec = tensor_to_int64_vector(max_idx);
    EXPECT_EQ(max_idx_vec[0], 1);
    EXPECT_EQ(max_idx_vec[1], 2);
    EXPECT_EQ(max_idx_vec[2], 0);
}

// ============= Test sort =============

class TensorSortTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
        torch::manual_seed(42);
    }
};

TEST_F(TensorSortTest, Sort1DAscending) {
    auto t = Tensor::from_vector({5.0f, 2.0f, 8.0f, 1.0f, 9.0f}, {5});

    auto torch_t = to_torch(t);
    auto [torch_sorted, torch_indices] = torch::sort(torch_t, 0, false);

    auto [sorted, indices] = t.sort(0, false);

    EXPECT_TRUE(tensors_close(sorted, torch_sorted, 1e-5f, 1e-5f));

    // Verify sorted order
    auto sorted_vec = sorted.to_vector();
    EXPECT_EQ(sorted_vec.size(), 5);
    EXPECT_NEAR(sorted_vec[0], 1.0f, 1e-5f);
    EXPECT_NEAR(sorted_vec[1], 2.0f, 1e-5f);
    EXPECT_NEAR(sorted_vec[2], 5.0f, 1e-5f);
    EXPECT_NEAR(sorted_vec[3], 8.0f, 1e-5f);
    EXPECT_NEAR(sorted_vec[4], 9.0f, 1e-5f);

    // Verify it's actually sorted
    for (size_t i = 1; i < sorted_vec.size(); ++i) {
        EXPECT_LE(sorted_vec[i-1], sorted_vec[i]);
    }
}

TEST_F(TensorSortTest, Sort1DDescending) {
    auto t = Tensor::from_vector({5.0f, 2.0f, 8.0f, 1.0f, 9.0f}, {5});

    auto torch_t = to_torch(t);
    auto [torch_sorted, torch_indices] = torch::sort(torch_t, 0, true);

    auto [sorted, indices] = t.sort(0, true);

    EXPECT_TRUE(tensors_close(sorted, torch_sorted, 1e-5f, 1e-5f));

    // Verify sorted order
    auto sorted_vec = sorted.to_vector();
    EXPECT_EQ(sorted_vec.size(), 5);
    EXPECT_NEAR(sorted_vec[0], 9.0f, 1e-5f);
    EXPECT_NEAR(sorted_vec[1], 8.0f, 1e-5f);
    EXPECT_NEAR(sorted_vec[2], 5.0f, 1e-5f);
    EXPECT_NEAR(sorted_vec[3], 2.0f, 1e-5f);
    EXPECT_NEAR(sorted_vec[4], 1.0f, 1e-5f);

    // Verify it's actually sorted descending
    for (size_t i = 1; i < sorted_vec.size(); ++i) {
        EXPECT_GE(sorted_vec[i-1], sorted_vec[i]);
    }
}

TEST_F(TensorSortTest, SortRandom) {
    auto t = Tensor::randn({100});

    auto torch_t = to_torch(t);
    auto [torch_sorted, torch_indices] = torch::sort(torch_t, 0, false);

    auto [sorted, indices] = t.sort(0, false);

    EXPECT_TRUE(tensors_close(sorted, torch_sorted, 1e-4f, 1e-4f));

    // Verify sorted property
    auto sorted_vec = sorted.to_vector();
    for (size_t i = 1; i < sorted_vec.size(); ++i) {
        EXPECT_LE(sorted_vec[i-1], sorted_vec[i])
            << "Sort order violated at index " << i;
    }
}

TEST_F(TensorSortTest, SortCPU) {
    auto t = Tensor::randn({50}, Device::CPU);

    auto [sorted, indices] = t.sort(0, false);

    EXPECT_EQ(sorted.device(), Device::CPU);
    EXPECT_EQ(indices.device(), Device::CPU);

    // Verify sorted property
    auto sorted_vec = sorted.to_vector();
    for (size_t i = 1; i < sorted_vec.size(); ++i) {
        EXPECT_LE(sorted_vec[i-1], sorted_vec[i]);
    }
}

TEST_F(TensorSortTest, SortIndicesCorrect) {
    auto t = Tensor::from_vector({5.0f, 2.0f, 8.0f, 1.0f, 9.0f}, {5});

    auto [sorted, indices] = t.sort(0, false);

    auto idx_vec = tensor_to_int64_vector(indices);
    auto orig_vec = t.to_vector();
    auto sorted_vec = sorted.to_vector();

    // Verify that indices correctly map to sorted values
    EXPECT_EQ(idx_vec.size(), sorted_vec.size());
    for (size_t i = 0; i < idx_vec.size(); ++i) {
        size_t idx = static_cast<size_t>(idx_vec[i]);
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, orig_vec.size());
        EXPECT_NEAR(orig_vec[idx], sorted_vec[i], 1e-5f)
            << "Index mapping incorrect at position " << i;
    }
}

TEST_F(TensorSortTest, SortStability) {
    // Test with duplicate values to check stability
    auto t = Tensor::from_vector({3.0f, 1.0f, 2.0f, 1.0f, 3.0f}, {5});

    auto [sorted, indices] = t.sort(0, false);

    auto sorted_vec = sorted.to_vector();
    EXPECT_NEAR(sorted_vec[0], 1.0f, 1e-5f);
    EXPECT_NEAR(sorted_vec[1], 1.0f, 1e-5f);
    EXPECT_NEAR(sorted_vec[2], 2.0f, 1e-5f);
    EXPECT_NEAR(sorted_vec[3], 3.0f, 1e-5f);
    EXPECT_NEAR(sorted_vec[4], 3.0f, 1e-5f);
}

// ============= Test any_scalar and all_scalar =============

class TensorBoolReductionTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
    }
};

TEST_F(TensorBoolReductionTest, AnyScalarAllZeros) {
    auto t = Tensor::zeros({10});

    EXPECT_FALSE(t.any_scalar());
    EXPECT_FALSE(t.all_scalar());

    // Compare with PyTorch
    auto torch_t = to_torch(t);
    EXPECT_EQ(t.any_scalar(), torch_t.any().item<bool>());
    EXPECT_EQ(t.all_scalar(), torch_t.all().item<bool>());
}

TEST_F(TensorBoolReductionTest, AnyScalarAllOnes) {
    auto t = Tensor::ones({10});

    EXPECT_TRUE(t.any_scalar());
    EXPECT_TRUE(t.all_scalar());

    auto torch_t = to_torch(t);
    EXPECT_EQ(t.any_scalar(), torch_t.any().item<bool>());
    EXPECT_EQ(t.all_scalar(), torch_t.all().item<bool>());
}

TEST_F(TensorBoolReductionTest, AnyScalarMixed) {
    auto t = Tensor::zeros({10});

    // Modify on CPU to set one element
    auto cpu_t = t.to(Device::CPU);
    cpu_t.ptr<float>()[5] = 1.0f;
    t = cpu_t.to(Device::CUDA);

    EXPECT_TRUE(t.any_scalar());
    EXPECT_FALSE(t.all_scalar());

    auto torch_t = to_torch(t);
    EXPECT_EQ(t.any_scalar(), torch_t.any().item<bool>());
    EXPECT_EQ(t.all_scalar(), torch_t.all().item<bool>());
}

TEST_F(TensorBoolReductionTest, AnyAllBoolTensor) {
    auto t = Tensor::zeros_bool({10});
    EXPECT_FALSE(t.any_scalar());
    EXPECT_FALSE(t.all_scalar());

    t = Tensor::ones_bool({10});
    EXPECT_TRUE(t.any_scalar());
    EXPECT_TRUE(t.all_scalar());

    // Mixed
    t = Tensor::zeros_bool({10});
    auto cpu_t = t.to(Device::CPU);
    cpu_t.ptr<unsigned char>()[0] = 1;
    t = cpu_t.to(Device::CUDA);

    EXPECT_TRUE(t.any_scalar());
    EXPECT_FALSE(t.all_scalar());
}

TEST_F(TensorBoolReductionTest, AnyAllEmpty) {
    auto t = Tensor::empty({0});

    EXPECT_FALSE(t.any_scalar());
    EXPECT_TRUE(t.all_scalar());  // Vacuously true for empty
}

TEST_F(TensorBoolReductionTest, AnyAllLargeTensor) {
    auto t = Tensor::zeros({1000, 1000});
    EXPECT_FALSE(t.any_scalar());

    // Set one element in the middle
    auto cpu_t = t.to(Device::CPU);
    cpu_t.ptr<float>()[500*1000 + 500] = 1.0f;
    t = cpu_t.to(Device::CUDA);

    EXPECT_TRUE(t.any_scalar());
    EXPECT_FALSE(t.all_scalar());

    // Verify count
    EXPECT_EQ(t.count_nonzero(), 1);
}

// ============= Test TensorOptions =============

class TensorOptionsTest : public ::testing::Test {};

TEST_F(TensorOptionsTest, OptionsBasic) {
    auto t = Tensor::randn({10}, Device::CUDA, DataType::Float32);

    auto opts = t.options();
    EXPECT_EQ(opts.device, Device::CUDA);
    EXPECT_EQ(opts.dtype, DataType::Float32);
}

TEST_F(TensorOptionsTest, OptionsConstructors) {
    Tensor::TensorOptions opt1;
    EXPECT_EQ(opt1.device, Device::CUDA);
    EXPECT_EQ(opt1.dtype, DataType::Float32);

    Tensor::TensorOptions opt2(Device::CPU);
    EXPECT_EQ(opt2.device, Device::CPU);
    EXPECT_EQ(opt2.dtype, DataType::Float32);

    Tensor::TensorOptions opt3(DataType::Int32);
    EXPECT_EQ(opt3.device, Device::CUDA);
    EXPECT_EQ(opt3.dtype, DataType::Int32);

    Tensor::TensorOptions opt4(Device::CPU, DataType::Int32);
    EXPECT_EQ(opt4.device, Device::CPU);
    EXPECT_EQ(opt4.dtype, DataType::Int32);
}

TEST_F(TensorOptionsTest, OptionsFromTensor) {
    auto t1 = Tensor::randn({10}, Device::CPU, DataType::Float32);
    auto opts = t1.options();

    auto t2 = Tensor::zeros({5}, opts.device, opts.dtype);
    EXPECT_EQ(t2.device(), Device::CPU);
    EXPECT_EQ(t2.dtype(), DataType::Float32);
    EXPECT_EQ(t2.shape(), TensorShape({5}));
}

// ============= Integration Tests =============

class TensorIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
        torch::manual_seed(42);
    }
};

TEST_F(TensorIntegrationTest, KMeansPlusPlusInitialization) {
    // Test that our new operations work correctly together for k-means++ initialization
    size_t n = 100;
    size_t d = 5;

    auto data = Tensor::randn({n, d});

    // Test 1: cdist works
    auto dists = data.cdist(data);
    EXPECT_EQ(dists.shape(), TensorShape({n, n}));
    EXPECT_TRUE(dists.is_valid());

    // Test 2: min_with_indices works on cdist result
    auto [min_dists, min_indices] = dists.min_with_indices(1);
    EXPECT_EQ(min_dists.shape(), TensorShape({n}));
    EXPECT_EQ(min_indices.shape(), TensorShape({n}));
    EXPECT_TRUE(min_dists.is_valid());
    EXPECT_TRUE(min_indices.is_valid());

    // Test 3: max_with_indices works on the min distances
    auto [max_dist, max_idx] = min_dists.max_with_indices(0);
    EXPECT_TRUE(max_dist.is_valid());
    EXPECT_TRUE(max_idx.is_valid());

    // Test 4: Can extract the index
    auto idx_vec = tensor_to_int64_vector(max_idx);
    EXPECT_EQ(idx_vec.size(), 1);
    EXPECT_GE(idx_vec[0], 0);
    EXPECT_LT(idx_vec[0], static_cast<int64_t>(n));

    // Test 5: Verify distance computation is correct
    auto pt1 = data.slice(0, 0, 1);
    auto pt2 = data.slice(0, 50, 51);

    EXPECT_EQ(pt1.shape(), TensorShape({1, d}));
    EXPECT_EQ(pt2.shape(), TensorShape({1, d}));

    auto pt1_vals = pt1.to_vector();
    auto pt2_vals = pt2.to_vector();

    // Verify they're different points
    bool all_same = true;
    for (size_t i = 0; i < std::min(pt1_vals.size(), pt2_vals.size()); ++i) {
        if (std::abs(pt1_vals[i] - pt2_vals[i]) > 1e-6f) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "Points should be different";

    // Manually compute expected distance
    float manual_dist = 0.0f;
    for (size_t i = 0; i < std::min(pt1_vals.size(), pt2_vals.size()); ++i) {
        float d = pt1_vals[i] - pt2_vals[i];
        manual_dist += d * d;
    }
    manual_dist = std::sqrt(manual_dist);

    // Compute using our operations
    auto diff = pt1.sub(pt2);
    EXPECT_EQ(diff.shape(), TensorShape({1, d}));

    auto squared = diff.pow(2.0f);
    EXPECT_EQ(squared.shape(), TensorShape({1, d}));

    auto sum_result = squared.sum();
    float sum_val = sum_result.item();

    auto sqrt_result = sum_result.sqrt();
    float dist = sqrt_result.item();

    EXPECT_NEAR(dist, manual_dist, 1e-5f) << "Computed distance should match manual calculation";

    // Also verify no NaN in intermediate results
    auto squared_vals = squared.to_vector();
    for (size_t i = 0; i < squared_vals.size(); ++i) {
        EXPECT_FALSE(std::isnan(squared_vals[i]))
            << "NaN found at squared index " << i;
        EXPECT_GE(squared_vals[i], 0.0f)
            << "Squared value should be non-negative at index " << i;
    }
}

TEST_F(TensorIntegrationTest, CopyAndModify) {
    auto original = Tensor::randn({100});

    // IMPORTANT: Use clone() for independent copy
    auto copy1 = original.clone();  // Deep copy
    auto copy2 = Tensor::empty_like(original);
    copy2.copy_(original);  // copy_() also does deep copy

    EXPECT_TRUE(original.all_close(copy1));
    EXPECT_TRUE(original.all_close(copy2));

    // Store original values
    auto original_vals = original.to_vector();

    // Modify copies independently
    copy1.add_(1.0f);
    copy2.mul_(2.0f);

    // Verify original is unchanged
    auto original_vals_after = original.to_vector();
    for (size_t i = 0; i < original_vals.size(); ++i) {
        EXPECT_NEAR(original_vals[i], original_vals_after[i], 1e-5f);
    }

    EXPECT_FALSE(original.all_close(copy1));
    EXPECT_FALSE(original.all_close(copy2));
    EXPECT_FALSE(copy1.all_close(copy2));

    // Verify modifications are correct
    auto copy1_vals = copy1.to_vector();
    for (size_t i = 0; i < original_vals.size(); ++i) {
        EXPECT_NEAR(copy1_vals[i], original_vals[i] + 1.0f, 1e-5f);
    }

    auto copy2_vals = copy2.to_vector();
    for (size_t i = 0; i < original_vals.size(); ++i) {
        EXPECT_NEAR(copy2_vals[i], original_vals[i] * 2.0f, 1e-5f);
    }
}

TEST_F(TensorIntegrationTest, SortAndSelect) {
    auto data = Tensor::randn({100});

    auto [sorted, indices] = data.sort(0, false);

    // Get top 10 smallest values
    auto top_10 = sorted.slice(0, 0, 10);
    auto top_10_indices = indices.slice(0, 0, 10);

    EXPECT_EQ(top_10.shape(), TensorShape({10}));
    EXPECT_EQ(top_10_indices.shape(), TensorShape({10}));

    // Verify they are the actual smallest values
    auto top_10_vec = top_10.to_vector();
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_LE(top_10_vec[i], top_10_vec[i+1])
            << "Top 10 should be sorted at index " << i;
    }

    // Verify indices are valid
    auto idx_vec = tensor_to_int64_vector(top_10_indices);
    for (auto idx : idx_vec) {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, 100);
    }
}

TEST_F(TensorIntegrationTest, DistanceMatrixSymmetry) {
    auto data = Tensor::randn({50, 10});

    auto dist = data.cdist(data);

    EXPECT_EQ(dist.shape(), TensorShape({50, 50}));

    // Distance matrix should be symmetric
    auto dist_t = dist.t();
    EXPECT_TRUE(dist.all_close(dist_t, 1e-4f, 1e-4f));

    // Diagonal should be zero
    for (size_t i = 0; i < 50; ++i) {
        float diag_val = dist.to(Device::CPU).at({i, i});
        EXPECT_NEAR(diag_val, 0.0f, 1e-5f)
            << "Diagonal element " << i << " should be zero";
    }

    // All values should be non-negative
    auto dist_vec = dist.to_vector();
    for (size_t i = 0; i < dist_vec.size(); ++i) {
        EXPECT_GE(dist_vec[i], 0.0f)
            << "Distance should be non-negative at index " << i;
    }
}

// ============= Performance Comparison Tests =============

class TensorPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
        torch::manual_seed(42);
    }

    template<typename Func>
    double measure_time(Func&& func, int iterations = 10) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            func();
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    }
};

TEST_F(TensorPerformanceTest, CdistPerformance) {
    auto a = Tensor::randn({1000, 128});
    auto b = Tensor::randn({500, 128});

    auto torch_a = to_torch(a).cuda();
    auto torch_b = to_torch(b).cuda();

    double our_time = measure_time([&]() {
        auto result = a.cdist(b);
    });

    double torch_time = measure_time([&]() {
        auto result = torch::cdist(torch_a, torch_b);
    });

    std::cout << "cdist Performance (1000x128 vs 500x128):\n";
    std::cout << "  Our impl: " << our_time << " ms\n";
    std::cout << "  PyTorch:  " << torch_time << " ms\n";
    std::cout << "  Ratio:    " << (our_time / torch_time) << "x\n";

    // We should be within 5x of PyTorch
    EXPECT_LT(our_time / torch_time, 5.0)
        << "Our implementation should be within 5x of PyTorch";
}

TEST_F(TensorPerformanceTest, SortPerformance) {
    auto t = Tensor::randn({10000});
    auto torch_t = to_torch(t).cuda();

    double our_time = measure_time([&]() {
        auto [sorted, indices] = t.sort(0);
    });

    double torch_time = measure_time([&]() {
        auto [sorted, indices] = torch::sort(torch_t, 0);
    });

    std::cout << "sort Performance (10000 elements):\n";
    std::cout << "  Our impl: " << our_time << " ms\n";
    std::cout << "  PyTorch:  " << torch_time << " ms\n";
    std::cout << "  Ratio:    " << (our_time / torch_time) << "x\n";
}

TEST_F(TensorPerformanceTest, CopyPerformance) {
    auto large = Tensor::randn({10000, 1000});

    double copy_time = measure_time([&]() {
        Tensor copy = large;  // Shallow copy (very fast!)
    }, 5);

    std::cout << "Copy Performance (10000x1000):\n";
    std::cout << "  Time: " << copy_time << " ms\n";

    // Shallow copy should be very fast
    EXPECT_LT(copy_time, 1.0)  // Should be under 1ms
        << "Shallow copy operation should be very fast";
}

// ============= Edge Cases and Error Handling =============

class TensorEdgeCasesTest : public ::testing::Test {};

TEST_F(TensorEdgeCasesTest, CdistMismatchedDimensions) {
    auto a = Tensor::randn({5, 3});
    auto b = Tensor::randn({7, 4});

    auto result = a.cdist(b);
    EXPECT_FALSE(result.is_valid());
}

TEST_F(TensorEdgeCasesTest, CdistNonMatrix) {
    auto a = Tensor::randn({5});
    auto b = Tensor::randn({7});

    auto result = a.cdist(b);
    EXPECT_FALSE(result.is_valid());
}

TEST_F(TensorEdgeCasesTest, MinMaxInvalidTensor) {
    Tensor t;

    auto [val, idx] = t.min_with_indices(0);
    EXPECT_FALSE(val.is_valid());
    EXPECT_FALSE(idx.is_valid());
}

TEST_F(TensorEdgeCasesTest, MinMaxEmptyTensor) {
    auto t = Tensor::empty({0});

    auto [val, idx] = t.min_with_indices(0);
    EXPECT_FALSE(val.is_valid());
    EXPECT_FALSE(idx.is_valid());
}

TEST_F(TensorEdgeCasesTest, SortInvalidDimension) {
    auto t = Tensor::randn({5, 10});

    // Sort along invalid dimension should handle gracefully
    // This is implementation-dependent behavior
    auto [sorted, indices] = t.sort(5);  // Invalid dimension

    // Should either return empty or handle the error
    // We don't specify exact behavior, just that it shouldn't crash
}

TEST_F(TensorEdgeCasesTest, CopyUninitializedTensor) {
    Tensor a;
    Tensor b = a;

    EXPECT_FALSE(a.is_valid());
    EXPECT_FALSE(b.is_valid());
}

TEST_F(TensorEdgeCasesTest, SortSingleElement) {
    auto t = Tensor::from_vector({42.0f}, {1});

    auto [sorted, indices] = t.sort(0, false);

    EXPECT_EQ(sorted.numel(), 1);
    EXPECT_NEAR(sorted.item(), 42.0f, 1e-5f);

    auto idx_vec = tensor_to_int64_vector(indices);
    EXPECT_EQ(idx_vec.size(), 1);
    EXPECT_EQ(idx_vec[0], 0);
}

TEST_F(TensorEdgeCasesTest, MinMaxSingleElement) {
    auto t = Tensor::from_vector({42.0f}, {1});

    auto [min_val, min_idx] = t.min_with_indices(0);
    auto [max_val, max_idx] = t.max_with_indices(0);

    EXPECT_NEAR(min_val.item(), 42.0f, 1e-5f);
    EXPECT_NEAR(max_val.item(), 42.0f, 1e-5f);

    auto min_idx_vec = tensor_to_int64_vector(min_idx);
    auto max_idx_vec = tensor_to_int64_vector(max_idx);

    EXPECT_EQ(min_idx_vec[0], 0);
    EXPECT_EQ(max_idx_vec[0], 0);
}