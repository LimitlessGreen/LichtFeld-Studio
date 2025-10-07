/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <torch/torch.h>

using namespace gs;

class TensorMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Record initial CUDA memory
        cudaMemGetInfo(&initial_free_mem, &total_mem);
    }

    void TearDown() override {
        // Check for memory leaks
        size_t current_free_mem, current_total;
        cudaMemGetInfo(&current_free_mem, &current_total);

        // Allow tolerance for CUDA internal allocations and allocator pooling
        // CUDA allocators often allocate in 2MB chunks to reduce overhead
        size_t tolerance = 4 * 1024 * 1024; // 4 MB tolerance
        EXPECT_NEAR(current_free_mem, initial_free_mem, tolerance)
            << "Possible memory leak detected";
    }

    size_t initial_free_mem;
    size_t total_mem;
};

TEST_F(TensorMemoryTest, MemoryOwnership) {
    // Test owning memory
    {
        auto tensor = Tensor::zeros({100, 100}, Device::CUDA);
        EXPECT_TRUE(tensor.owns_memory());
        EXPECT_NE(tensor.raw_ptr(), nullptr);
    }
    // Memory should be freed after scope

    // Test non-owning view
    float* cuda_data;
    cudaMalloc(&cuda_data, 100 * sizeof(float));
    {
        auto tensor = Tensor::from_blob(cuda_data, {10, 10}, Device::CUDA, DataType::Float32);
        EXPECT_FALSE(tensor.owns_memory());
        EXPECT_EQ(tensor.raw_ptr(), cuda_data);
    }
    // Memory should NOT be freed after scope
    cudaFree(cuda_data);
}

TEST_F(TensorMemoryTest, MoveSemantics) {
    // Test move constructor
    void* original_ptr = nullptr;
    {
        auto tensor1 = Tensor::ones({50, 50}, Device::CUDA);
        EXPECT_TRUE(tensor1.owns_memory());
        original_ptr = tensor1.raw_ptr();

        auto tensor2 = std::move(tensor1);
        EXPECT_EQ(tensor2.raw_ptr(), original_ptr);
        EXPECT_TRUE(tensor2.owns_memory());
        EXPECT_EQ(tensor1.raw_ptr(), nullptr);
        EXPECT_FALSE(tensor1.owns_memory());
    }

    // Test move assignment
    {
        auto tensor1 = Tensor::zeros({30, 30}, Device::CUDA);
        auto tensor2 = Tensor::ones({20, 20}, Device::CUDA);

        void* ptr1 = tensor1.raw_ptr();
        void* ptr2 = tensor2.raw_ptr();

        tensor2 = std::move(tensor1);

        EXPECT_EQ(tensor2.raw_ptr(), ptr1);
        EXPECT_EQ(tensor1.raw_ptr(), nullptr);
        // ptr2 should have been freed
    }
}

TEST_F(TensorMemoryTest, ViewDoesNotOwnMemory) {
    auto original = Tensor::ones({4, 5, 6}, Device::CUDA);
    EXPECT_TRUE(original.owns_memory());

    auto view = original.view({20, 6});
    EXPECT_FALSE(view.owns_memory());
    EXPECT_EQ(view.raw_ptr(), original.raw_ptr());

    // Modifying view should affect original
    view.fill_(2.0f);
    auto original_values = original.to_vector();
    for (float val : original_values) {
        EXPECT_FLOAT_EQ(val, 2.0f);
    }
}

TEST_F(TensorMemoryTest, SliceDoesNotOwnMemory) {
    auto original = Tensor::full({10, 10}, 3.0f, Device::CUDA);
    EXPECT_TRUE(original.owns_memory());

    auto slice = original.slice(0, 2, 5);
    EXPECT_FALSE(slice.owns_memory());

    // Slice should point to part of original memory
    EXPECT_GE(slice.raw_ptr(), original.raw_ptr());
    EXPECT_LT(slice.raw_ptr(),
              static_cast<char*>(original.raw_ptr()) + original.bytes());
}

TEST_F(TensorMemoryTest, CloneOwnsMemory) {
    auto original = Tensor::ones({3, 3}, Device::CUDA);
    EXPECT_TRUE(original.owns_memory());

    auto cloned = original.clone();
    EXPECT_TRUE(cloned.owns_memory());
    EXPECT_NE(cloned.raw_ptr(), original.raw_ptr());

    // Modifying clone should not affect original
    cloned.fill_(5.0f);
    auto original_values = original.to_vector();
    for (float val : original_values) {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }
}

TEST_F(TensorMemoryTest, DeviceTransferOwnsMemory) {
    auto cuda_tensor = Tensor::full({5, 5}, 2.5f, Device::CUDA);
    EXPECT_TRUE(cuda_tensor.owns_memory());

    auto cpu_tensor = cuda_tensor.to(Device::CPU);
    EXPECT_TRUE(cpu_tensor.owns_memory());
    EXPECT_NE(cpu_tensor.raw_ptr(), cuda_tensor.raw_ptr());

    auto cuda_tensor2 = cpu_tensor.to(Device::CUDA);
    EXPECT_TRUE(cuda_tensor2.owns_memory());
    EXPECT_NE(cuda_tensor2.raw_ptr(), cuda_tensor.raw_ptr());
    EXPECT_NE(cuda_tensor2.raw_ptr(), cpu_tensor.raw_ptr());
}

TEST_F(TensorMemoryTest, LargeTensorAllocation) {
    // Test large tensor allocation
    const size_t large_size = 1024 * 1024; // 1M elements = 4MB for float32

    {
        auto large_tensor = Tensor::zeros({large_size}, Device::CUDA);
        EXPECT_TRUE(large_tensor.is_valid());
        EXPECT_EQ(large_tensor.numel(), large_size);
        EXPECT_EQ(large_tensor.bytes(), large_size * sizeof(float));
    }
    // Should be freed
}

TEST_F(TensorMemoryTest, MultipleViewsOfSameMemory) {
    auto original = Tensor::ones({24}, Device::CUDA);

    auto view1 = original.view({2, 12});
    auto view2 = original.view({3, 8});
    auto view3 = original.view({4, 6});

    // All views should share the same memory
    EXPECT_EQ(view1.raw_ptr(), original.raw_ptr());
    EXPECT_EQ(view2.raw_ptr(), original.raw_ptr());
    EXPECT_EQ(view3.raw_ptr(), original.raw_ptr());

    // None of the views should own memory
    EXPECT_FALSE(view1.owns_memory());
    EXPECT_FALSE(view2.owns_memory());
    EXPECT_FALSE(view3.owns_memory());

    // Modifying through one view affects all
    view1.fill_(3.0f);

    auto check_values = [](const Tensor& t, float expected) {
        auto values = t.to_vector();
        for (float val : values) {
            EXPECT_FLOAT_EQ(val, expected);
        }
    };

    check_values(original, 3.0f);
    check_values(view2, 3.0f);
    check_values(view3, 3.0f);
}

TEST_F(TensorMemoryTest, CopyFromPreservesOwnership) {
    auto tensor1 = Tensor::ones({3, 3}, Device::CUDA);
    auto tensor2 = Tensor::zeros({3, 3}, Device::CUDA);

    EXPECT_TRUE(tensor1.owns_memory());
    EXPECT_TRUE(tensor2.owns_memory());

    void* ptr1 = tensor1.raw_ptr();
    void* ptr2 = tensor2.raw_ptr();

    tensor2.copy_from(tensor1);

    // Both should still own their memory
    EXPECT_TRUE(tensor1.owns_memory());
    EXPECT_TRUE(tensor2.owns_memory());

    // Pointers should not have changed
    EXPECT_EQ(tensor1.raw_ptr(), ptr1);
    EXPECT_EQ(tensor2.raw_ptr(), ptr2);

    // But values should be copied
    auto values2 = tensor2.to_vector();
    for (float val : values2) {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }
}

TEST_F(TensorMemoryTest, InvalidTensorOperations) {
    Tensor invalid;
    EXPECT_FALSE(invalid.is_valid());
    EXPECT_EQ(invalid.raw_ptr(), nullptr);
    EXPECT_FALSE(invalid.owns_memory());

    // Operations on invalid tensor should return invalid tensors
    auto result = invalid.clone();
    EXPECT_FALSE(result.is_valid());

    result = invalid.to(Device::CPU);
    EXPECT_FALSE(result.is_valid());

    result = invalid.view({2, 2});
    EXPECT_FALSE(result.is_valid());
}

TEST_F(TensorMemoryTest, MemoryAlignmentAndPadding) {
    // Test that memory is properly aligned
    auto tensor = Tensor::empty({17}, Device::CUDA); // Odd size

    // CUDA memory should be aligned to at least 256 bytes
    uintptr_t addr = reinterpret_cast<uintptr_t>(tensor.raw_ptr());
    EXPECT_EQ(addr % 256, 0) << "CUDA memory not properly aligned";
}

TEST_F(TensorMemoryTest, StressTestManyAllocations) {
    // Stress test with many small allocations
    std::vector<Tensor> tensors;

    for (int i = 0; i < 100; ++i) {
        tensors.push_back(Tensor::zeros({10, 10}, Device::CUDA));
    }

    // All should be valid
    for (const auto& t : tensors) {
        EXPECT_TRUE(t.is_valid());
        EXPECT_TRUE(t.owns_memory());
    }

    // Clear them
    tensors.clear();

    // Memory should be freed (checked in TearDown)
}
