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
    auto cpu_t = t.to(Device::CPU);

    if (cpu_t.dtype() != DataType::Int64) {
        throw std::runtime_error(
            std::string("Expected Int64 tensor, got ") + dtype_name(cpu_t.dtype())
        );
    }

    if (!cpu_t.is_valid() || cpu_t.numel() == 0) {
        return {};
    }

    std::vector<int64_t> result(cpu_t.numel());
    const int64_t* data = reinterpret_cast<const int64_t*>(cpu_t.raw_ptr());

    if (!data) {
        throw std::runtime_error("Null data pointer in Int64 tensor");
    }

    std::memcpy(result.data(), data, cpu_t.bytes());
    return result;
}

// ============= Test Copy Constructor and Assignment =============

class TensorCopyTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
        torch::manual_seed(42);
    }
};

TEST_F(TensorCopyTest, CopyConstructorCPU) {
    auto a = Tensor::arange(0, 10, 1).to(Device::CPU);
    Tensor b(a);  // Copy constructor

    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.device(), b.device());
    EXPECT_EQ(a.dtype(), b.dtype());
    EXPECT_TRUE(a.all_close(b));

    // Verify they are independent
    b.fill_(42.0f);
    EXPECT_FALSE(a.all_close(b));
}

TEST_F(TensorCopyTest, CopyConstructorCUDA) {
    auto a = Tensor::arange(0, 10, 1).to(Device::CUDA);
    Tensor b(a);  // Copy constructor

    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.device(), b.device());
    EXPECT_EQ(a.dtype(), b.dtype());
    EXPECT_TRUE(a.all_close(b));

    // Verify they are independent
    b.fill_(42.0f);
    EXPECT_FALSE(a.all_close(b));
}

TEST_F(TensorCopyTest, CopyAssignmentCPU) {
    auto a = Tensor::arange(0, 10, 1).to(Device::CPU);
    auto b = Tensor::zeros({5}, Device::CPU);

    b = a;  // Copy assignment

    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.device(), b.device());
    EXPECT_TRUE(a.all_close(b));

    // Verify they are independent
    b.fill_(42.0f);
    EXPECT_FALSE(a.all_close(b));
}

TEST_F(TensorCopyTest, CopyAssignmentCUDA) {
    auto a = Tensor::arange(0, 10, 1).to(Device::CUDA);
    auto b = Tensor::zeros({5}, Device::CUDA);

    b = a;  // Copy assignment

    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.device(), b.device());
    EXPECT_TRUE(a.all_close(b));

    // Verify they are independent
    b.fill_(42.0f);
    EXPECT_FALSE(a.all_close(b));
}

TEST_F(TensorCopyTest, CopyAssignmentSelfAssignment) {
    auto a = Tensor::arange(0, 10, 1);
    auto original_values = a.to_vector();

    a = a;  // Self-assignment

    auto after_values = a.to_vector();
    EXPECT_EQ(original_values, after_values);
}

TEST_F(TensorCopyTest, CopyUnderscoreMethod) {
    auto a = Tensor::arange(0, 10, 1);
    auto b = Tensor::zeros({10});

    b.copy_(a);  // Using copy_() method

    EXPECT_TRUE(a.all_close(b));

    // Verify they are independent
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
    Tensor b(a);

    EXPECT_TRUE(a.all_close(b, 1e-5f, 1e-5f));

    // Modify and verify independence
    b.add_(1.0f);
    EXPECT_FALSE(a.all_close(b));
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
}

TEST_F(TensorCdistTest, CdistL1) {
    auto a = Tensor::randn({5, 3});
    auto b = Tensor::randn({7, 3});

    auto torch_a = to_torch(a);
    auto torch_b = to_torch(b);
    auto torch_result = torch::cdist(torch_a, torch_b, 1.0);

    auto result = a.cdist(b, 1.0f);

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
        EXPECT_NEAR(val, 0.0f, 1e-5f);
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
    EXPECT_NEAR(l1_dist.to(Device::CPU).item(), 7.0f, 1e-5f);

    // L2 distance should be 5.0
    EXPECT_NEAR(l2_dist.to(Device::CPU).item(), 5.0f, 1e-5f);
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

    EXPECT_NEAR(val.item(), torch_val.item<float>(), 1e-5f);

    // FIX: Use proper helper for Int64 tensors
    auto idx_val = tensor_to_int64_vector(idx);
    EXPECT_EQ(idx_val[0], torch_idx.item<int64_t>());
}

TEST_F(TensorMinMaxIndicesTest, MaxWithIndices1D) {
    auto t = Tensor::from_vector({5.0f, 2.0f, 8.0f, 1.0f, 9.0f}, {5});

    auto torch_t = to_torch(t);
    auto [torch_val, torch_idx] = torch::max(torch_t, 0);

    auto [val, idx] = t.max_with_indices(0);

    EXPECT_NEAR(val.item(), torch_val.item<float>(), 1e-5f);

    // FIX: Use proper helper for Int64 tensors
    auto idx_val = tensor_to_int64_vector(idx);
    EXPECT_EQ(idx_val[0], torch_idx.item<int64_t>());
}

TEST_F(TensorMinMaxIndicesTest, MinWithIndices2D) {
    auto t = Tensor::randn({5, 10});

    auto torch_t = to_torch(t);
    auto [torch_val, torch_idx] = torch::min(torch_t, 1);

    auto [val, idx] = t.min_with_indices(1);

    EXPECT_TRUE(tensors_close(val, torch_val, 1e-5f, 1e-5f));

    // FIX: Use proper helper for Int64 tensors
    auto our_idx_vec = tensor_to_int64_vector(idx);

    // Check indices match
    auto torch_idx_cpu = torch_idx.cpu();
    for (int64_t i = 0; i < torch_idx_cpu.size(0); ++i) {
        EXPECT_EQ(our_idx_vec[i], torch_idx_cpu[i].item<int64_t>());
    }
}

TEST_F(TensorMinMaxIndicesTest, MaxWithIndices2D) {
    auto t = Tensor::randn({5, 10});

    auto torch_t = to_torch(t);
    auto [torch_val, torch_idx] = torch::max(torch_t, 1);

    auto [val, idx] = t.max_with_indices(1);

    EXPECT_TRUE(tensors_close(val, torch_val, 1e-5f, 1e-5f));

    // FIX: Use proper helper for Int64 tensors
    auto our_idx_vec = tensor_to_int64_vector(idx);

    // Check indices match
    auto torch_idx_cpu = torch_idx.cpu();
    for (int64_t i = 0; i < torch_idx_cpu.size(0); ++i) {
        EXPECT_EQ(our_idx_vec[i], torch_idx_cpu[i].item<int64_t>());
    }
}

TEST_F(TensorMinMaxIndicesTest, MinWithIndicesDim0) {
    auto t = Tensor::randn({5, 10});

    auto torch_t = to_torch(t);
    auto [torch_val, torch_idx] = torch::min(torch_t, 0);

    auto [val, idx] = t.min_with_indices(0);

    EXPECT_EQ(val.shape(), TensorShape({10}));
    EXPECT_TRUE(tensors_close(val, torch_val, 1e-5f, 1e-5f));
}

TEST_F(TensorMinMaxIndicesTest, MinMaxWithKeepdim) {
    auto t = Tensor::randn({5, 10, 8});

    auto [val, idx] = t.min_with_indices(1, true);

    EXPECT_EQ(val.shape(), TensorShape({5, 1, 8}));
    EXPECT_EQ(idx.shape(), TensorShape({5, 1, 8}));
}

TEST_F(TensorMinMaxIndicesTest, MinMaxNegativeDim) {
    auto t = Tensor::randn({5, 10});

    auto [val1, idx1] = t.min_with_indices(-1);
    auto [val2, idx2] = t.min_with_indices(1);

    EXPECT_TRUE(val1.all_close(val2));
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
    EXPECT_EQ(sorted_vec[0], 1.0f);
    EXPECT_EQ(sorted_vec[1], 2.0f);
    EXPECT_EQ(sorted_vec[2], 5.0f);
    EXPECT_EQ(sorted_vec[3], 8.0f);
    EXPECT_EQ(sorted_vec[4], 9.0f);
}

TEST_F(TensorSortTest, Sort1DDescending) {
    auto t = Tensor::from_vector({5.0f, 2.0f, 8.0f, 1.0f, 9.0f}, {5});

    auto torch_t = to_torch(t);
    auto [torch_sorted, torch_indices] = torch::sort(torch_t, 0, true);

    auto [sorted, indices] = t.sort(0, true);

    EXPECT_TRUE(tensors_close(sorted, torch_sorted, 1e-5f, 1e-5f));

    // Verify sorted order
    auto sorted_vec = sorted.to_vector();
    EXPECT_EQ(sorted_vec[0], 9.0f);
    EXPECT_EQ(sorted_vec[1], 8.0f);
    EXPECT_EQ(sorted_vec[2], 5.0f);
    EXPECT_EQ(sorted_vec[3], 2.0f);
    EXPECT_EQ(sorted_vec[4], 1.0f);
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
        EXPECT_LE(sorted_vec[i-1], sorted_vec[i]);
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

    // FIX: Use proper helper for Int64 tensors
    auto idx_vec = tensor_to_int64_vector(indices);
    auto orig_vec = t.to_vector();
    auto sorted_vec = sorted.to_vector();

    // Verify that indices correctly map to sorted values
    for (size_t i = 0; i < idx_vec.size(); ++i) {
        size_t idx = static_cast<size_t>(idx_vec[i]);
        EXPECT_EQ(orig_vec[idx], sorted_vec[i]);
    }
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
    // Modify on CPU
    auto cpu_t = t.to(Device::CPU);
    cpu_t.ptr<float>()[5] = 1.0f;  // Set one element to 1
    // Copy back to CUDA
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
}

TEST_F(TensorBoolReductionTest, AnyAllEmpty) {
    auto t = Tensor::empty({0});
    EXPECT_FALSE(t.any_scalar());
    EXPECT_TRUE(t.all_scalar());  // Vacuously true
}

TEST_F(TensorBoolReductionTest, AnyAllLargeTensor) {
    auto t = Tensor::zeros({1000, 1000});
    EXPECT_FALSE(t.any_scalar());

    // Set one element
    auto cpu_t = t.to(Device::CPU);
    cpu_t.ptr<float>()[500*1000 + 500] = 1.0f;
    t = cpu_t.to(Device::CUDA);

    EXPECT_TRUE(t.any_scalar());
    EXPECT_FALSE(t.all_scalar());
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

    Tensor::TensorOptions opt3(DataType::Int32);
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
    // Test that our new operations work correctly together
    size_t n = 100;
    size_t d = 5;

    auto data = Tensor::randn({n, d});

    // Test 1: cdist works
    auto dists = data.cdist(data);
    EXPECT_EQ(dists.shape(), TensorShape({n, n}));

    // Test 2: min_with_indices works on cdist result
    auto [min_dists, min_indices] = dists.min_with_indices(1);
    EXPECT_EQ(min_dists.shape(), TensorShape({n}));
    EXPECT_EQ(min_indices.shape(), TensorShape({n}));

    // Test 3: max_with_indices works on the min distances
    auto [max_dist, max_idx] = min_dists.max_with_indices(0);
    EXPECT_TRUE(max_dist.is_valid());
    EXPECT_TRUE(max_idx.is_valid());

    // Test 4: Can extract the index
    auto idx_vec = tensor_to_int64_vector(max_idx);
    EXPECT_EQ(idx_vec.size(), 1);
    EXPECT_GE(idx_vec[0], 0);
    EXPECT_LT(idx_vec[0], static_cast<int64_t>(n));

    // Test 5: Verify operations work step by step
    auto pt1 = data.slice(0, 0, 1);
    auto pt2 = data.slice(0, 50, 51);

    std::cout << "pt1 shape: " << pt1.shape().str() << std::endl;
    std::cout << "pt2 shape: " << pt2.shape().str() << std::endl;

    auto pt1_vals = pt1.to_vector();
    auto pt2_vals = pt2.to_vector();

    std::cout << "pt1 values: ";
    for (size_t i = 0; i < pt1_vals.size(); ++i) {
        std::cout << pt1_vals[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "pt2 values: ";
    for (size_t i = 0; i < pt2_vals.size(); ++i) {
        std::cout << pt2_vals[i] << " ";
    }
    std::cout << std::endl;

    // Verify they're different
    bool all_same = true;
    for (size_t i = 0; i < std::min(pt1_vals.size(), pt2_vals.size()); ++i) {
        if (std::abs(pt1_vals[i] - pt2_vals[i]) > 1e-6f) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);

    // Test subtraction
    auto diff = pt1.sub(pt2);
    std::cout << "diff shape: " << diff.shape().str() << std::endl;
    auto diff_vals = diff.to_vector();
    std::cout << "diff values: ";
    for (size_t i = 0; i < diff_vals.size(); ++i) {
        std::cout << diff_vals[i] << " ";
    }
    std::cout << std::endl;

    // Manually compute expected distance
    float manual_dist = 0.0f;
    for (size_t i = 0; i < std::min(pt1_vals.size(), pt2_vals.size()); ++i) {
        float d = pt1_vals[i] - pt2_vals[i];
        manual_dist += d * d;
    }
    manual_dist = std::sqrt(manual_dist);
    std::cout << "Manual distance: " << manual_dist << std::endl;

    // Now test each operation
    auto squared = diff.pow(2.0f);
    std::cout << "squared shape: " << squared.shape().str() << std::endl;
    auto squared_vals = squared.to_vector();
    std::cout << "squared values: ";
    for (size_t i = 0; i < squared_vals.size(); ++i) {
        std::cout << squared_vals[i] << " ";
    }
    std::cout << std::endl;

    auto sum_result = squared.sum();
    std::cout << "sum shape: " << sum_result.shape().str() << std::endl;
    float sum_val = sum_result.item();
    std::cout << "sum value: " << sum_val << std::endl;

    auto sqrt_result = sum_result.sqrt();
    float dist = sqrt_result.item();
    std::cout << "Final distance: " << dist << std::endl;

    EXPECT_NEAR(dist, manual_dist, 1e-5f) << "Computed distance should match manual calculation";
}

TEST_F(TensorIntegrationTest, CopyAndModify) {
    auto original = Tensor::randn({100});
    Tensor copy1 = original;  // Copy constructor
    auto copy2 = Tensor::empty_like(original);
    copy2.copy_(original);  // copy_ method

    EXPECT_TRUE(original.all_close(copy1));
    EXPECT_TRUE(original.all_close(copy2));

    // Modify copies independently
    copy1.add_(1.0f);
    copy2.mul_(2.0f);

    EXPECT_FALSE(original.all_close(copy1));
    EXPECT_FALSE(original.all_close(copy2));
    EXPECT_FALSE(copy1.all_close(copy2));
}

TEST_F(TensorIntegrationTest, SortAndSelect) {
    auto data = Tensor::randn({100});

    auto [sorted, indices] = data.sort(0, false);

    // Get top 10 smallest values
    auto top_10 = sorted.slice(0, 0, 10);
    auto top_10_indices = indices.slice(0, 0, 10);

    // Verify they are the actual smallest values
    auto top_10_vec = top_10.to_vector();
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_LE(top_10_vec[i], top_10_vec[i+1]);
    }
}

TEST_F(TensorIntegrationTest, DistanceMatrixSymmetry) {
    auto data = Tensor::randn({50, 10});

    auto dist = data.cdist(data);

    // Distance matrix should be symmetric
    auto dist_t = dist.t();
    EXPECT_TRUE(dist.all_close(dist_t, 1e-4f, 1e-4f));

    // Diagonal should be zero
    for (size_t i = 0; i < 50; ++i) {
        float diag_val = dist.to(Device::CPU).at({i, i});
        EXPECT_NEAR(diag_val, 0.0f, 1e-5f);
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
    EXPECT_LT(our_time / torch_time, 5.0);
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
        Tensor copy = large;  // Copy constructor
    }, 5);  // Fewer iterations for large tensor

    std::cout << "Copy Performance (10000x1000):\n";
    std::cout << "  Time: " << copy_time << " ms\n";

    // Should complete in reasonable time (< 100ms)
    EXPECT_LT(copy_time, 100.0);
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

TEST_F(TensorEdgeCasesTest, SortInvalidDimension) {
    auto t = Tensor::randn({5, 10});

    auto [sorted, indices] = t.sort(5);  // Invalid dimension

    // Should handle gracefully or return empty
    // Implementation dependent
}

TEST_F(TensorEdgeCasesTest, CopyUninitializedTensor) {
    Tensor a;
    Tensor b = a;

    EXPECT_FALSE(a.is_valid());
    EXPECT_FALSE(b.is_valid());
}