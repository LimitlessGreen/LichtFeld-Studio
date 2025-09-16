/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>

using namespace gs;

class TensorBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";

        // Set random seed for reproducibility
        torch::manual_seed(42);
        gen.seed(42);
    }

    // Helper to compare our tensor with torch tensor
    bool tensors_equal(const Tensor& our_tensor, const torch::Tensor& torch_tensor, float tolerance = 1e-5f) {
        // Check shapes match
        if (our_tensor.shape().rank() != static_cast<size_t>(torch_tensor.dim())) {
            return false;
        }

        for (size_t i = 0; i < our_tensor.shape().rank(); ++i) {
            if (our_tensor.shape()[i] != static_cast<size_t>(torch_tensor.size(i))) {
                return false;
            }
        }

        // Compare values
        auto our_values = our_tensor.to_vector();
        auto torch_cpu = torch_tensor.to(torch::kCPU).contiguous();
        auto torch_data = torch_cpu.data_ptr<float>();

        for (size_t i = 0; i < our_values.size(); ++i) {
            if (std::abs(our_values[i] - torch_data[i]) > tolerance) {
                return false;
            }
        }

        return true;
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};
};

// Test tensor creation
TEST_F(TensorBasicTest, EmptyTensorCreation) {
    // Create empty tensors
    auto our_tensor = Tensor::empty({2, 3, 4}, Device::CUDA);
    auto torch_tensor = torch::empty({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Check properties
    EXPECT_EQ(our_tensor.shape().rank(), 3);
    EXPECT_EQ(our_tensor.shape()[0], 2);
    EXPECT_EQ(our_tensor.shape()[1], 3);
    EXPECT_EQ(our_tensor.shape()[2], 4);
    EXPECT_EQ(our_tensor.numel(), 24);
    EXPECT_EQ(our_tensor.device(), Device::CUDA);
    EXPECT_EQ(our_tensor.dtype(), DataType::Float32);
    EXPECT_TRUE(our_tensor.is_valid());
    EXPECT_FALSE(our_tensor.is_empty());
}

TEST_F(TensorBasicTest, ZerosTensorCreation) {
    // Create zeros tensors
    auto our_tensor = Tensor::zeros({3, 4}, Device::CUDA);
    auto torch_tensor = torch::zeros({3, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Verify all zeros
    auto values = our_tensor.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }

    EXPECT_TRUE(tensors_equal(our_tensor, torch_tensor));
}

TEST_F(TensorBasicTest, OnesTensorCreation) {
    // Create ones tensors
    auto our_tensor = Tensor::ones({5, 2}, Device::CUDA);
    auto torch_tensor = torch::ones({5, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Verify all ones
    auto values = our_tensor.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }

    EXPECT_TRUE(tensors_equal(our_tensor, torch_tensor));
}

TEST_F(TensorBasicTest, FullTensorCreation) {
    float fill_value = 3.14f;

    auto our_tensor = Tensor::full({2, 3, 2}, fill_value, Device::CUDA);
    auto torch_tensor = torch::full({2, 3, 2}, fill_value,
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Verify all values
    auto values = our_tensor.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, fill_value);
    }

    EXPECT_TRUE(tensors_equal(our_tensor, torch_tensor));
}

TEST_F(TensorBasicTest, FromBlobCreation) {
    // Create data on device
    size_t num_elements = 12;
    float* cuda_data;
    cudaMalloc(&cuda_data, num_elements * sizeof(float));

    // Fill with test data
    std::vector<float> host_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        host_data[i] = static_cast<float>(i);
    }
    cudaMemcpy(cuda_data, host_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Create tensor from blob
    auto our_tensor = Tensor::from_blob(cuda_data, {3, 4}, Device::CUDA, DataType::Float32);
    auto torch_tensor = torch::from_blob(cuda_data, {3, 4},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    EXPECT_TRUE(tensors_equal(our_tensor, torch_tensor));
    EXPECT_FALSE(our_tensor.owns_memory()); // Should not own the memory

    // Cleanup
    cudaFree(cuda_data);
}

TEST_F(TensorBasicTest, DeviceTransfer) {
    // Create CUDA tensor
    auto cuda_tensor = Tensor::full({3, 3}, 2.5f, Device::CUDA);

    // Transfer to CPU
    auto cpu_tensor = cuda_tensor.to(Device::CPU);
    EXPECT_EQ(cpu_tensor.device(), Device::CPU);

    // Check values preserved
    auto cuda_values = cuda_tensor.to_vector();
    auto cpu_values = cpu_tensor.to_vector();

    ASSERT_EQ(cuda_values.size(), cpu_values.size());
    for (size_t i = 0; i < cuda_values.size(); ++i) {
        EXPECT_FLOAT_EQ(cuda_values[i], cpu_values[i]);
    }

    // Transfer back to CUDA
    auto cuda_tensor2 = cpu_tensor.to(Device::CUDA);
    EXPECT_EQ(cuda_tensor2.device(), Device::CUDA);
    EXPECT_TRUE(cuda_tensor.all_close(cuda_tensor2));
}

TEST_F(TensorBasicTest, Clone) {
    // Create original tensor with random data
    auto original = Tensor::empty({4, 5}, Device::CUDA);
    std::vector<float> data(20);
    for (auto& val : data) {
        val = dist(gen);
    }
    cudaMemcpy(original.ptr<float>(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Clone it
    auto cloned = original.clone();

    // Check they have same values but different memory
    EXPECT_TRUE(original.all_close(cloned));
    EXPECT_NE(original.raw_ptr(), cloned.raw_ptr());
    EXPECT_TRUE(cloned.owns_memory());

    // Modify clone shouldn't affect original
    cloned.fill_(0.0f);
    EXPECT_FALSE(original.all_close(cloned));
}

TEST_F(TensorBasicTest, CopyFrom) {
    auto tensor1 = Tensor::ones({2, 3}, Device::CUDA);
    auto tensor2 = Tensor::zeros({2, 3}, Device::CUDA);

    // Copy tensor1 to tensor2
    tensor2.copy_from(tensor1);

    // Check values copied
    EXPECT_TRUE(tensor1.all_close(tensor2));

    // But different memory
    EXPECT_NE(tensor1.raw_ptr(), tensor2.raw_ptr());
}

TEST_F(TensorBasicTest, FillOperations) {
    auto tensor = Tensor::empty({3, 4}, Device::CUDA);

    // Fill with value
    tensor.fill_(5.5f);
    auto values = tensor.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 5.5f);
    }

    // Zero
    tensor.zero_();
    values = tensor.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }
}

TEST_F(TensorBasicTest, MoveSemantics) {
    // Test move constructor
    auto tensor1 = Tensor::ones({2, 2}, Device::CUDA);
    void* original_ptr = tensor1.raw_ptr();

    Tensor tensor2(std::move(tensor1));
    EXPECT_EQ(tensor2.raw_ptr(), original_ptr);
    EXPECT_EQ(tensor1.raw_ptr(), nullptr); // tensor1 should be cleared

    // Test move assignment
    auto tensor3 = Tensor::zeros({2, 2}, Device::CUDA);
    void* tensor3_original = tensor3.raw_ptr();

    tensor3 = std::move(tensor2);
    EXPECT_EQ(tensor3.raw_ptr(), original_ptr);
    EXPECT_EQ(tensor2.raw_ptr(), nullptr);
}

TEST_F(TensorBasicTest, Properties) {
    auto tensor = Tensor::full({2, 3, 4}, 1.0f, Device::CUDA);

    EXPECT_EQ(tensor.numel(), 24);
    EXPECT_EQ(tensor.bytes(), 24 * sizeof(float));
    EXPECT_TRUE(tensor.is_valid());
    EXPECT_FALSE(tensor.is_empty());
    EXPECT_TRUE(tensor.is_contiguous());
    EXPECT_TRUE(tensor.owns_memory());

    // Test shape access
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 3);
    EXPECT_EQ(tensor.shape()[2], 4);
    EXPECT_EQ(tensor.shape().rank(), 3);
}

TEST_F(TensorBasicTest, InvalidOperations) {
    // Test operations on invalid tensor
    Tensor invalid_tensor;
    EXPECT_FALSE(invalid_tensor.is_valid());
    EXPECT_TRUE(invalid_tensor.is_empty());

    // These should not crash but return invalid tensors
    auto result = invalid_tensor.clone();
    EXPECT_FALSE(result.is_valid());

    result = invalid_tensor.add(1.0f);
    EXPECT_FALSE(result.is_valid());
}

TEST_F(TensorBasicTest, AllClose) {
    auto tensor1 = Tensor::full({3, 3}, 1.0f, Device::CUDA);
    auto tensor2 = Tensor::full({3, 3}, 1.0f, Device::CUDA);
    auto tensor3 = Tensor::full({3, 3}, 1.00001f, Device::CUDA);
    auto tensor4 = Tensor::full({3, 3}, 2.0f, Device::CUDA);

    EXPECT_TRUE(tensor1.all_close(tensor2));
    EXPECT_TRUE(tensor1.all_close(tensor3, 1e-4f));  // With higher tolerance
    EXPECT_FALSE(tensor1.all_close(tensor3, 1e-6f)); // With lower tolerance
    EXPECT_FALSE(tensor1.all_close(tensor4));
}
