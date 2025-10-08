/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#include "core/tensor.hpp"

namespace {

constexpr float FLOAT_TOLERANCE = 1e-4f;

// Helper to convert torch::Tensor to gs::Tensor
gs::Tensor torch_to_tensor(const torch::Tensor& torch_tensor) {
    auto cpu_tensor = torch_tensor.cpu().contiguous();
    std::vector<size_t> shape;
    for (int i = 0; i < torch_tensor.dim(); ++i) {
        shape.push_back(torch_tensor.size(i));
    }
    
    if (torch_tensor.scalar_type() == torch::kFloat32) {
        std::vector<float> data(cpu_tensor.data_ptr<float>(), 
                               cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
        return gs::Tensor::from_vector(data, gs::TensorShape(shape), gs::Device::CUDA);
    } else if (torch_tensor.scalar_type() == torch::kInt32) {
        std::vector<int> data(cpu_tensor.data_ptr<int>(), 
                             cpu_tensor.data_ptr<int>() + cpu_tensor.numel());
        return gs::Tensor::from_vector(data, gs::TensorShape(shape), gs::Device::CUDA);
    }
    
    return gs::Tensor();
}

// Helper to convert gs::Tensor to torch::Tensor
torch::Tensor tensor_to_torch(const gs::Tensor& gs_tensor) {
    auto cpu_tensor = gs_tensor.cpu();
    std::vector<int64_t> shape;
    for (size_t i = 0; i < cpu_tensor.ndim(); ++i) {
        shape.push_back(cpu_tensor.shape()[i]);
    }
    
    if (gs_tensor.dtype() == gs::DataType::Float32) {
        auto data = cpu_tensor.to_vector();
        auto torch_tensor = torch::from_blob(data.data(), shape, torch::kFloat32).clone();
        return torch_tensor.cuda();
    } else if (gs_tensor.dtype() == gs::DataType::Int32) {
        auto data = cpu_tensor.to_vector_int();
        auto torch_tensor = torch::from_blob(data.data(), shape, torch::kInt32).clone();
        return torch_tensor.cuda();
    }
    
    return torch::Tensor();
}

// Helper to compare tensors
bool tensors_close(const gs::Tensor& a, const torch::Tensor& b, float tol = FLOAT_TOLERANCE) {
    if (a.numel() != b.numel()) {
        std::cout << "Size mismatch: " << a.numel() << " vs " << b.numel() << std::endl;
        return false;
    }
    
    auto a_cpu = a.cpu().to_vector();
    auto b_cpu = b.cpu();
    auto b_ptr = b_cpu.data_ptr<float>();
    
    for (size_t i = 0; i < a.numel(); ++i) {
        float diff = std::abs(a_cpu[i] - b_ptr[i]);
        if (diff > tol) {
            std::cout << "Mismatch at index " << i << ": " << a_cpu[i] << " vs " << b_ptr[i] 
                      << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

} // anonymous namespace

class TensorOpsComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        torch::manual_seed(42);
        torch::cuda::manual_seed(42);
        gs::Tensor::manual_seed(42);
    }
};

// ============================================================================
// Basic Creation and Initialization
// ============================================================================

TEST_F(TensorOpsComparisonTest, Zeros) {
    auto torch_zeros = torch::zeros({10, 5}, torch::kCUDA);
    auto gs_zeros = gs::Tensor::zeros({10, 5}, gs::Device::CUDA);
    
    EXPECT_TRUE(tensors_close(gs_zeros, torch_zeros));
}

TEST_F(TensorOpsComparisonTest, Ones) {
    auto torch_ones = torch::ones({10, 5}, torch::kCUDA);
    auto gs_ones = gs::Tensor::ones({10, 5}, gs::Device::CUDA);
    
    EXPECT_TRUE(tensors_close(gs_ones, torch_ones));
}

TEST_F(TensorOpsComparisonTest, Full) {
    float value = 3.14f;
    auto torch_full = torch::full({10, 5}, value, torch::kCUDA);
    auto gs_full = gs::Tensor::full({10, 5}, value, gs::Device::CUDA);
    
    EXPECT_TRUE(tensors_close(gs_full, torch_full));
}

TEST_F(TensorOpsComparisonTest, Arange) {
    auto torch_arange = torch::arange(0, 100, 1, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto gs_arange = gs::Tensor::arange(0, 100, 1.0f);
    
    EXPECT_TRUE(tensors_close(gs_arange, torch_arange));
}

TEST_F(TensorOpsComparisonTest, Linspace) {
    auto torch_linspace = torch::linspace(0, 100, 50, torch::kCUDA);
    auto gs_linspace = gs::Tensor::linspace(0, 100, 50, gs::Device::CUDA);
    
    EXPECT_TRUE(tensors_close(gs_linspace, torch_linspace, 1e-3f));
}

// ============================================================================
// Shape Operations
// ============================================================================

TEST_F(TensorOpsComparisonTest, Unsqueeze) {
    auto torch_data = torch::randn({10}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    // Unsqueeze at dim 1
    auto torch_unsqueezed = torch_data.unsqueeze(1);
    auto gs_unsqueezed = gs_data.unsqueeze(1);
    
    EXPECT_EQ(torch_unsqueezed.size(0), gs_unsqueezed.shape()[0]);
    EXPECT_EQ(torch_unsqueezed.size(1), gs_unsqueezed.shape()[1]);
    EXPECT_TRUE(tensors_close(gs_unsqueezed, torch_unsqueezed));
}

TEST_F(TensorOpsComparisonTest, Squeeze) {
    auto torch_data = torch::randn({10, 1}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    auto torch_squeezed = torch_data.squeeze(1);
    auto gs_squeezed = gs_data.squeeze(1);
    
    EXPECT_EQ(torch_squeezed.size(0), gs_squeezed.shape()[0]);
    EXPECT_EQ(torch_squeezed.dim(), gs_squeezed.ndim());
    EXPECT_TRUE(tensors_close(gs_squeezed, torch_squeezed));
}

TEST_F(TensorOpsComparisonTest, Reshape) {
    auto torch_data = torch::randn({20}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    auto torch_reshaped = torch_data.reshape({4, 5});
    auto gs_reshaped = gs_data.reshape({4, 5});
    
    EXPECT_EQ(torch_reshaped.size(0), gs_reshaped.shape()[0]);
    EXPECT_EQ(torch_reshaped.size(1), gs_reshaped.shape()[1]);
    EXPECT_TRUE(tensors_close(gs_reshaped, torch_reshaped));
}

TEST_F(TensorOpsComparisonTest, Clone) {
    auto torch_data = torch::randn({10, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    auto torch_cloned = torch_data.clone();
    auto gs_cloned = gs_data.clone();
    
    EXPECT_TRUE(tensors_close(gs_cloned, torch_cloned));
}

TEST_F(TensorOpsComparisonTest, Slice) {
    auto torch_data = torch::randn({100, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    // Slice first dimension
    auto torch_sliced = torch_data.slice(0, 10, 20);
    auto gs_sliced = gs_data.slice(0, 10, 20);
    
    EXPECT_EQ(torch_sliced.size(0), gs_sliced.shape()[0]);
    EXPECT_EQ(torch_sliced.size(1), gs_sliced.shape()[1]);
    EXPECT_TRUE(tensors_close(gs_sliced, torch_sliced));
}

// ============================================================================
// Reduction Operations
// ============================================================================

TEST_F(TensorOpsComparisonTest, Min) {
    auto torch_data = torch::randn({100}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    float torch_min = torch_data.min().item<float>();
    float gs_min = gs_data.min().item();
    
    EXPECT_NEAR(gs_min, torch_min, FLOAT_TOLERANCE);
}

TEST_F(TensorOpsComparisonTest, Max) {
    auto torch_data = torch::randn({100}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    float torch_max = torch_data.max().item<float>();
    float gs_max = gs_data.max().item();
    
    EXPECT_NEAR(gs_max, torch_max, FLOAT_TOLERANCE);
}

TEST_F(TensorOpsComparisonTest, Sum) {
    auto torch_data = torch::randn({100}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    float torch_sum = torch_data.sum().item<float>();
    float gs_sum = gs_data.sum().item();
    
    EXPECT_NEAR(gs_sum, torch_sum, 1e-3f); // Slightly relaxed tolerance for sum
}

TEST_F(TensorOpsComparisonTest, Mean) {
    auto torch_data = torch::randn({100}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    float torch_mean = torch_data.mean().item<float>();
    float gs_mean = gs_data.mean().item();
    
    EXPECT_NEAR(gs_mean, torch_mean, FLOAT_TOLERANCE);
}

TEST_F(TensorOpsComparisonTest, Cumsum) {
    auto torch_data = torch::randn({100}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    auto torch_cumsum = torch_data.cumsum(0);
    auto gs_cumsum = gs_data.cumsum(0);
    
    EXPECT_TRUE(tensors_close(gs_cumsum, torch_cumsum, 1e-3f));
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

TEST_F(TensorOpsComparisonTest, Addition) {
    auto torch_a = torch::randn({10, 5}, torch::kCUDA);
    auto torch_b = torch::randn({10, 5}, torch::kCUDA);
    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);
    
    auto torch_result = torch_a + torch_b;
    auto gs_result = gs_a + gs_b;
    
    EXPECT_TRUE(tensors_close(gs_result, torch_result));
}

TEST_F(TensorOpsComparisonTest, Subtraction) {
    auto torch_a = torch::randn({10, 5}, torch::kCUDA);
    auto torch_b = torch::randn({10, 5}, torch::kCUDA);
    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);
    
    auto torch_result = torch_a - torch_b;
    auto gs_result = gs_a - gs_b;
    
    EXPECT_TRUE(tensors_close(gs_result, torch_result));
}

TEST_F(TensorOpsComparisonTest, Multiplication) {
    auto torch_a = torch::randn({10, 5}, torch::kCUDA);
    auto torch_b = torch::randn({10, 5}, torch::kCUDA);
    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);
    
    auto torch_result = torch_a * torch_b;
    auto gs_result = gs_a * gs_b;
    
    EXPECT_TRUE(tensors_close(gs_result, torch_result));
}

TEST_F(TensorOpsComparisonTest, Division) {
    auto torch_a = torch::randn({10, 5}, torch::kCUDA);
    auto torch_b = torch::randn({10, 5}, torch::kCUDA) + 1.0f; // Avoid division by zero
    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);
    
    auto torch_result = torch_a / torch_b;
    auto gs_result = gs_a / gs_b;
    
    EXPECT_TRUE(tensors_close(gs_result, torch_result, 1e-3f)); // Relaxed tolerance
}

TEST_F(TensorOpsComparisonTest, ScalarAddition) {
    auto torch_a = torch::randn({10, 5}, torch::kCUDA);
    auto gs_a = torch_to_tensor(torch_a);
    
    auto torch_result = torch_a + 5.0f;
    auto gs_result = gs_a + 5.0f;
    
    EXPECT_TRUE(tensors_close(gs_result, torch_result));
}

TEST_F(TensorOpsComparisonTest, ScalarMultiplication) {
    auto torch_a = torch::randn({10, 5}, torch::kCUDA);
    auto gs_a = torch_to_tensor(torch_a);
    
    auto torch_result = torch_a * 3.5f;
    auto gs_result = gs_a * 3.5f;
    
    EXPECT_TRUE(tensors_close(gs_result, torch_result));
}

TEST_F(TensorOpsComparisonTest, ScalarDivision) {
    auto torch_a = torch::randn({10, 5}, torch::kCUDA);
    auto gs_a = torch_to_tensor(torch_a);
    
    auto torch_result = torch_a / 2.5f;
    auto gs_result = gs_a / 2.5f;
    
    EXPECT_TRUE(tensors_close(gs_result, torch_result));
}

// ============================================================================
// Unary Operations
// ============================================================================

TEST_F(TensorOpsComparisonTest, Abs) {
    auto torch_data = torch::randn({10, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    auto torch_result = torch_data.abs();
    auto gs_result = gs_data.abs();
    
    EXPECT_TRUE(tensors_close(gs_result, torch_result));
}

TEST_F(TensorOpsComparisonTest, Square) {
    auto torch_data = torch::randn({10, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    auto torch_result = torch_data.square();
    auto gs_result = gs_data.square();
    
    EXPECT_TRUE(tensors_close(gs_result, torch_result));
}

TEST_F(TensorOpsComparisonTest, Sqrt) {
    auto torch_data = torch::rand({10, 5}, torch::kCUDA); // Positive values for sqrt
    auto gs_data = torch_to_tensor(torch_data);
    
    auto torch_result = torch_data.sqrt();
    auto gs_result = gs_data.sqrt();
    
    EXPECT_TRUE(tensors_close(gs_result, torch_result));
}

// ============================================================================
// Comparison Operations
// ============================================================================

TEST_F(TensorOpsComparisonTest, GreaterThan) {
    auto torch_a = torch::randn({10, 5}, torch::kCUDA);
    auto gs_a = torch_to_tensor(torch_a);
    
    auto torch_result = torch_a > 0.0f;
    auto gs_result = gs_a > 0.0f;
    
    // Convert bool tensors to float for comparison
    auto torch_result_float = torch_result.to(torch::kFloat32);
    auto gs_result_float = gs_result.to(gs::DataType::Float32);
    
    EXPECT_TRUE(tensors_close(gs_result_float, torch_result_float));
}

TEST_F(TensorOpsComparisonTest, Equality) {
    auto torch_a = torch::randint(0, 5, {10, 5}, torch::kCUDA).to(torch::kFloat32);
    auto gs_a = torch_to_tensor(torch_a);
    
    auto torch_result = torch_a == 2.0f;
    auto gs_result = gs_a == 2.0f;
    
    auto torch_result_float = torch_result.to(torch::kFloat32);
    auto gs_result_float = gs_result.to(gs::DataType::Float32);
    
    EXPECT_TRUE(tensors_close(gs_result_float, torch_result_float));
}

// ============================================================================
// Data Type Conversion
// ============================================================================

TEST_F(TensorOpsComparisonTest, ToInt32) {
    auto torch_data = torch::randn({10, 5}, torch::kCUDA) * 10.0f;
    auto gs_data = torch_to_tensor(torch_data);
    
    auto torch_int = torch_data.to(torch::kInt32);
    auto gs_int = gs_data.to(gs::DataType::Int32);
    
    auto torch_int_cpu = torch_int.cpu();
    auto gs_int_cpu = gs_int.cpu();
    
    auto torch_ptr = torch_int_cpu.data_ptr<int>();
    auto gs_vec = gs_int_cpu.to_vector_int();
    
    for (size_t i = 0; i < gs_vec.size(); ++i) {
        EXPECT_EQ(gs_vec[i], torch_ptr[i]);
    }
}

// ============================================================================
// Memory Operations
// ============================================================================

TEST_F(TensorOpsComparisonTest, CopyFrom) {
    auto torch_src = torch::randn({10, 5}, torch::kCUDA);
    auto gs_src = torch_to_tensor(torch_src);
    
    auto torch_dst = torch::zeros({10, 5}, torch::kCUDA);
    auto gs_dst = gs::Tensor::zeros({10, 5}, gs::Device::CUDA);
    
    torch_dst.copy_(torch_src);
    gs_dst.copy_from(gs_src);
    
    EXPECT_TRUE(tensors_close(gs_dst, torch_dst));
}

TEST_F(TensorOpsComparisonTest, CPUToGPU) {
    auto torch_cpu = torch::randn({10, 5});
    auto gs_cpu_vec = std::vector<float>(torch_cpu.data_ptr<float>(), 
                                         torch_cpu.data_ptr<float>() + torch_cpu.numel());
    auto gs_cpu = gs::Tensor::from_vector(gs_cpu_vec, gs::TensorShape({10, 5}), gs::Device::CPU);
    
    auto torch_gpu = torch_cpu.cuda();
    auto gs_gpu = gs_cpu.cuda();
    
    EXPECT_TRUE(tensors_close(gs_gpu, torch_gpu));
}

// ============================================================================
// Indexing and Masking
// ============================================================================

TEST_F(TensorOpsComparisonTest, MaskedSelect) {
    auto torch_data = torch::randn({100}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    auto torch_mask = torch_data > 0.0f;
    auto gs_mask = gs_data > 0.0f;
    
    auto torch_selected = torch_data.masked_select(torch_mask);
    auto gs_selected = gs_data.masked_select(gs_mask);
    
    EXPECT_EQ(torch_selected.numel(), gs_selected.numel());
    EXPECT_TRUE(tensors_close(gs_selected, torch_selected));
}

TEST_F(TensorOpsComparisonTest, Nonzero) {
    auto torch_data = torch::tensor({0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    auto torch_nz = torch_data.nonzero().squeeze();
    auto gs_nz = gs_data.nonzero();
    
    EXPECT_EQ(torch_nz.numel(), gs_nz.numel());
    
    // Compare indices
    auto torch_nz_cpu = torch_nz.cpu();
    auto gs_nz_cpu = gs_nz.cpu();
    auto torch_ptr = torch_nz_cpu.data_ptr<int64_t>();
    auto gs_vec = gs_nz_cpu.to_vector_int();
    
    for (size_t i = 0; i < gs_vec.size(); ++i) {
        EXPECT_EQ(gs_vec[i], static_cast<int>(torch_ptr[i]));
    }
}

// ============================================================================
// Special K-means Related Operations
// ============================================================================

TEST_F(TensorOpsComparisonTest, ExpandBroadcast) {
    auto torch_data = torch::randn({1, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    auto torch_expanded = torch_data.expand({10, 5});
    auto gs_expanded = gs_data.expand({10, 5});
    
    EXPECT_TRUE(tensors_close(gs_expanded, torch_expanded));
}

TEST_F(TensorOpsComparisonTest, MinimumElementwise) {
    auto torch_a = torch::randn({10, 5}, torch::kCUDA);
    auto torch_b = torch::randn({10, 5}, torch::kCUDA);
    auto gs_a = torch_to_tensor(torch_a);
    auto gs_b = torch_to_tensor(torch_b);
    
    auto torch_result = torch::minimum(torch_a, torch_b);
    auto gs_result = gs_a.minimum(gs_b);
    
    EXPECT_TRUE(tensors_close(gs_result, torch_result));
}

TEST_F(TensorOpsComparisonTest, PointerAccess) {
    auto torch_data = torch::randn({100}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    // Copy data using raw pointers
    std::vector<float> torch_vec(100);
    std::vector<float> gs_vec(100);
    
    cudaMemcpy(torch_vec.data(), torch_data.data_ptr<float>(), 100 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gs_vec.data(), gs_data.ptr<float>(), 100 * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_NEAR(gs_vec[i], torch_vec[i], FLOAT_TOLERANCE);
    }
}

// ============================================================================
// K-means Specific Pattern Tests
// ============================================================================

TEST_F(TensorOpsComparisonTest, MemcpyPattern) {
    // Test the pattern: copy a slice of data using cudaMemcpy
    auto torch_data = torch::randn({100, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    auto torch_dest = torch::zeros({10, 5}, torch::kCUDA);
    auto gs_dest = gs::Tensor::zeros({10, 5}, gs::Device::CUDA);
    
    // Copy rows 20-30 to destination
    cudaMemcpy(torch_dest.data_ptr<float>(), 
               torch_data.data_ptr<float>() + 20 * 5,
               10 * 5 * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    cudaMemcpy(gs_dest.ptr<float>(),
               gs_data.ptr<float>() + 20 * 5,
               10 * 5 * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    EXPECT_TRUE(tensors_close(gs_dest, torch_dest));
}

TEST_F(TensorOpsComparisonTest, UnsqueezeLinspace) {
    // Test pattern: linspace then unsqueeze
    auto torch_ls = torch::linspace(0, 100, 256, torch::kCUDA);
    auto gs_ls = gs::Tensor::linspace(0, 100, 256, gs::Device::CUDA);
    
    auto torch_unsq = torch_ls.unsqueeze(1);
    auto gs_unsq = gs_ls.unsqueeze(1);
    
    EXPECT_EQ(torch_unsq.size(0), gs_unsq.shape()[0]);
    EXPECT_EQ(torch_unsq.size(1), gs_unsq.shape()[1]);
    
    // Check that data is preserved
    EXPECT_TRUE(tensors_close(gs_unsq, torch_unsq, 1e-3f));
    
    // Check we can access the data via pointer
    std::vector<float> torch_vec(256);
    std::vector<float> gs_vec(256);
    cudaMemcpy(torch_vec.data(), torch_unsq.data_ptr<float>(), 256 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gs_vec.data(), gs_unsq.ptr<float>(), 256 * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(gs_vec[i], torch_vec[i], 1e-3f) << "Mismatch at index " << i;
    }
}

TEST_F(TensorOpsComparisonTest, SqueezeAfterUnsqueeze) {
    // Test pattern used in 1D k-means
    auto torch_data = torch::linspace(0, 100, 256, torch::kCUDA);
    auto gs_data = gs::Tensor::linspace(0, 100, 256, gs::Device::CUDA);
    
    auto torch_2d = torch_data.unsqueeze(1);
    auto gs_2d = gs_data.unsqueeze(1);
    
    auto torch_1d = torch_2d.squeeze(1);
    auto gs_1d = gs_2d.squeeze(1);
    
    EXPECT_TRUE(tensors_close(gs_1d, torch_1d, 1e-3f));
}

TEST_F(TensorOpsComparisonTest, Int32Labels) {
    // Test creating and accessing integer labels
    auto torch_labels = torch::zeros({100}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto gs_labels = gs::Tensor::zeros({100}, gs::Device::CUDA, gs::DataType::Int32);
    
    // Set some values
    std::vector<int> test_vals(100);
    for (int i = 0; i < 100; ++i) {
        test_vals[i] = i % 10;
    }
    
    cudaMemcpy(torch_labels.data_ptr<int>(), test_vals.data(), 100 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gs_labels.ptr<int>(), test_vals.data(), 100 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Read back
    std::vector<int> torch_readback(100);
    std::vector<int> gs_readback(100);
    cudaMemcpy(torch_readback.data(), torch_labels.data_ptr<int>(), 100 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(gs_readback.data(), gs_labels.ptr<int>(), 100 * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(gs_readback[i], torch_readback[i]) << "Mismatch at index " << i;
    }
}

TEST_F(TensorOpsComparisonTest, CudaMemsetPattern) {
    // Test using cudaMemset on tensor data
    auto torch_data = torch::zeros({100}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto gs_data = gs::Tensor::zeros({100}, gs::Device::CUDA, gs::DataType::Int32);
    
    cudaMemset(torch_data.data_ptr<int>(), 0, 100 * sizeof(int));
    cudaMemset(gs_data.ptr<int>(), 0, 100 * sizeof(int));
    
    std::vector<int> torch_vec(100);
    std::vector<int> gs_vec(100);
    cudaMemcpy(torch_vec.data(), torch_data.data_ptr<int>(), 100 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(gs_vec.data(), gs_data.ptr<int>(), 100 * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(gs_vec[i], 0);
        EXPECT_EQ(torch_vec[i], 0);
    }
}

TEST_F(TensorOpsComparisonTest, TwoDimensionalPointerAccess) {
    // Test accessing 2D tensor as flat array
    auto torch_data = torch::randn({10, 5}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);
    
    // Access element at [3, 2] via flat indexing
    std::vector<float> torch_vec(1);
    std::vector<float> gs_vec(1);
    
    int idx = 3 * 5 + 2;  // row 3, col 2
    cudaMemcpy(torch_vec.data(), torch_data.data_ptr<float>() + idx, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gs_vec.data(), gs_data.ptr<float>() + idx, sizeof(float), cudaMemcpyDeviceToHost);
    
    EXPECT_NEAR(gs_vec[0], torch_vec[0], FLOAT_TOLERANCE);
}

TEST_F(TensorOpsComparisonTest, RandintPattern) {
    // Test randint for index selection
    torch::manual_seed(123);
    gs::Tensor::manual_seed(123);
    
    auto torch_idx = torch::randint(0, 100, {1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto gs_idx = gs::Tensor::randint({1}, 0, 100, gs::Device::CUDA, gs::DataType::Int32);
    
    int torch_val = torch_idx.cpu().item<int>();
    int gs_val = gs_idx.cpu().item<int>();
    
    // Both should be in valid range
    EXPECT_GE(torch_val, 0);
    EXPECT_LT(torch_val, 100);
    EXPECT_GE(gs_val, 0);
    EXPECT_LT(gs_val, 100);
}
