/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <torch/torch.h>

using namespace gs;
using namespace gs::tensor;

class TensorAdvancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";

        // Set random seed for reproducibility
        torch::manual_seed(42);
        gen.seed(42);
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};
};

// ============= Utility Functions Tests =============
TEST_F(TensorAdvancedTest, Linspace) {
    // Test basic linspace
    auto t1 = linspace(0, 10, 11);
    EXPECT_EQ(t1.numel(), 11);
    auto vals1 = t1.to_vector();
    for (size_t i = 0; i < 11; ++i) {
        EXPECT_FLOAT_EQ(vals1[i], static_cast<float>(i));
    }

    // Test linspace with 2 points
    auto t2 = linspace(-5, 5, 2);
    EXPECT_EQ(t2.numel(), 2);
    auto vals2 = t2.to_vector();
    EXPECT_FLOAT_EQ(vals2[0], -5.0f);
    EXPECT_FLOAT_EQ(vals2[1], 5.0f);

    // Test single point
    auto t3 = linspace(3.14f, 3.14f, 1);
    EXPECT_EQ(t3.numel(), 1);
    EXPECT_FLOAT_EQ(t3.item(), 3.14f);

    // Test invalid (0 steps)
    auto invalid = linspace(0, 1, 0);
    EXPECT_FALSE(invalid.is_valid());
}

TEST_F(TensorAdvancedTest, Stack) {
    // Create test tensors
    auto t1 = Tensor::full({3, 4}, 1.0f, Device::CUDA);
    auto t2 = Tensor::full({3, 4}, 2.0f, Device::CUDA);
    auto t3 = Tensor::full({3, 4}, 3.0f, Device::CUDA);

    // Stack along dimension 0
    std::vector<Tensor> tensors;
    tensors.push_back(t1.clone());
    tensors.push_back(t2.clone());
    tensors.push_back(t3.clone());

    auto stacked = stack(std::move(tensors), 0);
    EXPECT_EQ(stacked.shape().rank(), 3);
    EXPECT_EQ(stacked.shape()[0], 3);
    EXPECT_EQ(stacked.shape()[1], 3);
    EXPECT_EQ(stacked.shape()[2], 4);

    // Verify values
    auto values = stacked.to_vector();
    // First tensor should be all 1s
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(values[i], 1.0f);
    }
    // Second tensor should be all 2s
    for (size_t i = 12; i < 24; ++i) {
        EXPECT_FLOAT_EQ(values[i], 2.0f);
    }
    // Third tensor should be all 3s
    for (size_t i = 24; i < 36; ++i) {
        EXPECT_FLOAT_EQ(values[i], 3.0f);
    }

    // Test stacking empty list
    std::vector<Tensor> empty_list;
    auto invalid = stack(std::move(empty_list));
    EXPECT_FALSE(invalid.is_valid());
}

TEST_F(TensorAdvancedTest, Concatenate) {
    // Create test tensors with different sizes along dim 0
    auto t1 = Tensor::full({2, 3}, 1.0f, Device::CUDA);
    auto t2 = Tensor::full({3, 3}, 2.0f, Device::CUDA);
    auto t3 = Tensor::full({1, 3}, 3.0f, Device::CUDA);

    // Concatenate along dimension 0
    std::vector<Tensor> tensors;
    tensors.push_back(t1.clone());
    tensors.push_back(t2.clone());
    tensors.push_back(t3.clone());

    auto concatenated = cat(std::move(tensors), 0);
    EXPECT_EQ(concatenated.shape().rank(), 2);
    EXPECT_EQ(concatenated.shape()[0], 6); // 2 + 3 + 1
    EXPECT_EQ(concatenated.shape()[1], 3);

    // Verify values
    auto values = concatenated.to_vector();
    // First 6 values (2x3) should be 1s
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(values[i], 1.0f);
    }
    // Next 9 values (3x3) should be 2s
    for (size_t i = 6; i < 15; ++i) {
        EXPECT_FLOAT_EQ(values[i], 2.0f);
    }
    // Last 3 values (1x3) should be 3s
    for (size_t i = 15; i < 18; ++i) {
        EXPECT_FLOAT_EQ(values[i], 3.0f);
    }

    // Test mismatched shapes (should fail)
    std::vector<Tensor> mismatched;
    mismatched.push_back(Tensor::zeros({2, 3}, Device::CUDA));
    mismatched.push_back(Tensor::zeros({2, 4}, Device::CUDA)); // Different dim 1
    auto invalid = cat(std::move(mismatched), 0);
    EXPECT_FALSE(invalid.is_valid());
}

// ============= Builder Pattern Tests =============
TEST_F(TensorAdvancedTest, TensorBuilder) {
    // Test basic builder
    auto t1 = TensorBuilder()
                  .with_shape({3, 4, 5})
                  .on_device(Device::CUDA)
                  .with_dtype(DataType::Float32)
                  .filled_with(2.5f)
                  .build();

    EXPECT_TRUE(t1.is_valid());
    EXPECT_EQ(t1.shape().rank(), 3);
    EXPECT_EQ(t1.device(), Device::CUDA);
    EXPECT_EQ(t1.dtype(), DataType::Float32);

    auto values = t1.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 2.5f);
    }

    // Test builder without fill value
    auto t2 = TensorBuilder()
                  .with_shape({10})
                  .on_device(Device::CPU)
                  .build();

    EXPECT_TRUE(t2.is_valid());
    EXPECT_EQ(t2.device(), Device::CPU);
    EXPECT_EQ(t2.numel(), 10);
}

// ============= Safe Operations Tests =============
TEST_F(TensorAdvancedTest, SafeOperations) {
    // Test safe division
    auto a = Tensor::full({3, 3}, 10.0f, Device::CUDA);
    auto b = Tensor::full({3, 3}, 0.0f, Device::CUDA); // Zero divisor

    auto result = SafeOps::divide(a, b, 1e-6f);
    EXPECT_TRUE(result.is_valid());
    EXPECT_FALSE(result.has_inf());

    // Values should be large but not infinite
    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FALSE(std::isinf(val));
        EXPECT_GT(std::abs(val), 1e5f); // Should be large
    }

    // Test safe log
    auto negative = Tensor::full({2, 2}, -5.0f, Device::CUDA);
    auto log_result = SafeOps::log(negative, 1e-6f);
    EXPECT_TRUE(log_result.is_valid());
    EXPECT_FALSE(log_result.has_nan());

    // Test safe sqrt
    auto neg_sqrt = Tensor::full({2, 2}, -4.0f, Device::CUDA);
    auto sqrt_result = SafeOps::sqrt(neg_sqrt, 0.0f);
    EXPECT_TRUE(sqrt_result.is_valid());
    EXPECT_FALSE(sqrt_result.has_nan());
}

// ============= Functional Programming Tests =============
// Define function objects outside of test to avoid template instantiation issues
struct SquareFunc {
    float operator()(float x) const { return x * x; }
};

struct PositiveFilter {
    bool operator()(float x) const { return x > 0; }
};

// ============= Memory Info Tests =============
TEST_F(TensorAdvancedTest, MemoryInfo) {
    auto initial_info = MemoryInfo::cuda();

    // Allocate a large tensor
    const size_t large_size = 1024 * 1024; // 1M elements = 4MB
    auto large_tensor = Tensor::zeros({large_size}, Device::CUDA);

    auto after_alloc_info = MemoryInfo::cuda();

    // Should have more allocated memory
    EXPECT_GT(after_alloc_info.allocated_bytes, initial_info.allocated_bytes);

    // Log the info (for debugging)
    initial_info.log();
    after_alloc_info.log();
}

// ============= Error Handling Tests =============
TEST_F(TensorAdvancedTest, ErrorHandling) {
    // Test shape mismatch in operations
    auto t1 = Tensor::ones({3, 4}, Device::CUDA);
    auto t2 = Tensor::ones({4, 3}, Device::CUDA);

    auto result = t1.add(t2);
    EXPECT_FALSE(result.is_valid());

    // Test invalid reshape
    auto t3 = Tensor::ones({12}, Device::CUDA);
    auto reshaped = t3.try_reshape({5, 3}); // 15 != 12
    EXPECT_FALSE(reshaped.has_value());

    // Test valid reshape
    auto valid_reshaped = t3.try_reshape({3, 4}); // 12 = 3*4
    EXPECT_TRUE(valid_reshaped.has_value());
    if (valid_reshaped.has_value()) {
        EXPECT_EQ(valid_reshaped->shape()[0], 3);
        EXPECT_EQ(valid_reshaped->shape()[1], 4);
        EXPECT_EQ(valid_reshaped->numel(), 12);
    }

    // Test operations on invalid tensors
    Tensor invalid;
    EXPECT_FALSE(invalid.is_valid());

    auto result2 = invalid.add(1.0f);
    EXPECT_FALSE(result2.is_valid());

    auto result3 = invalid.clone();
    EXPECT_FALSE(result3.is_valid());
}

// ============= Batch Processing Tests =============
TEST_F(TensorAdvancedTest, BatchProcessing) {
    // Create a large tensor
    auto large = Tensor::ones({100, 10}, Device::CUDA);

    // Split into batches
    auto batches = Tensor::split_batch(large, 32);

    // Should have 4 batches: 32, 32, 32, 4
    EXPECT_EQ(batches.size(), 4);
    EXPECT_EQ(batches[0].shape()[0], 32);
    EXPECT_EQ(batches[1].shape()[0], 32);
    EXPECT_EQ(batches[2].shape()[0], 32);
    EXPECT_EQ(batches[3].shape()[0], 4);

    // All should have same second dimension
    for (const auto& batch : batches) {
        EXPECT_EQ(batch.shape()[1], 10);
    }
}

// ============= Cross-Device Operations Tests =============
TEST_F(TensorAdvancedTest, CrossDeviceOperations) {
    // Create CPU tensor
    auto cpu_tensor = Tensor::full({3, 3}, 5.0f, Device::CPU);
    EXPECT_EQ(cpu_tensor.device(), Device::CPU);

    // Transfer to CUDA
    auto cuda_tensor = cpu_tensor.to(Device::CUDA);
    EXPECT_EQ(cuda_tensor.device(), Device::CUDA);
    EXPECT_TRUE(cpu_tensor.all_close(cuda_tensor));

    // Perform operation on CUDA
    auto result_cuda = cuda_tensor.mul(2.0f);

    // Transfer back to CPU
    auto result_cpu = result_cuda.to(Device::CPU);
    EXPECT_EQ(result_cpu.device(), Device::CPU);

    // Verify values
    auto values = result_cpu.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 10.0f);
    }
}

// ============= Stress Tests =============
TEST_F(TensorAdvancedTest, StressTestLargeTensors) {
    // Test with very large tensors
    const size_t large_dim = 1000;

    auto large1 = Tensor::zeros({large_dim, large_dim}, Device::CUDA);
    auto large2 = Tensor::ones({large_dim, large_dim}, Device::CUDA);

    // Perform operations
    auto sum = large1.add(large2);
    EXPECT_TRUE(sum.is_valid());
    EXPECT_FLOAT_EQ(sum.mean_scalar(), 1.0f);  // Use mean_scalar()

    auto product = large1.mul(large2);
    EXPECT_TRUE(product.is_valid());
    EXPECT_FLOAT_EQ(product.sum_scalar(), 0.0f); // All zeros - use sum_scalar()
}

TEST_F(TensorAdvancedTest, StressTestManyOperations) {
    // Chain many operations
    auto tensor = Tensor::ones({100, 100}, Device::CUDA);

    for (int i = 0; i < 100; ++i) {
        tensor = tensor.add(0.01f).mul(1.01f).sub(0.01f);
    }

    EXPECT_TRUE(tensor.is_valid());
    EXPECT_FALSE(tensor.has_nan());
    EXPECT_FALSE(tensor.has_inf());
}

// ============= Chainable Operations Tests =============
TEST_F(TensorAdvancedTest, ChainableInplace) {
    auto tensor = Tensor::ones({3, 3}, Device::CUDA);

    // Test inplace chaining
    tensor.inplace([](Tensor& t) { t.add_(1.0f); })
        .inplace([](Tensor& t) { t.mul_(2.0f); })
        .inplace([](Tensor& t) { t.sub_(1.0f); });

    // Should be ((1 + 1) * 2) - 1 = 3
    auto values = tensor.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 3.0f);
    }
}

TEST_F(TensorAdvancedTest, ChainableApply) {
    auto tensor = Tensor::ones({3, 3}, Device::CUDA);

    // Test apply (non-mutating)
    auto result = tensor.apply([](const Tensor& t) { return t.add(1.0f); })
                      .apply([](const Tensor& t) { return t.mul(2.0f); })
                      .apply([](const Tensor& t) { return t.sub(1.0f); });

    // Original should be unchanged
    EXPECT_FLOAT_EQ(tensor.to_vector()[0], 1.0f);

    // Result should be ((1 + 1) * 2) - 1 = 3
    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 3.0f);
    }
}

// ============= Edge Cases Tests =============
TEST_F(TensorAdvancedTest, ExtremelySparseOperations) {
    // Test with single element tensors
    auto scalar1 = Tensor::full({1}, 3.14f, Device::CUDA);
    auto scalar2 = Tensor::full({1}, 2.71f, Device::CUDA);

    auto sum = scalar1.add(scalar2);
    EXPECT_NEAR(sum.item(), 5.85f, 1e-5f);

    auto product = scalar1.mul(scalar2);
    EXPECT_NEAR(product.item(), 8.5094f, 1e-4f);

    // Test reshape of scalar
    auto reshaped = scalar1.view({1, 1, 1, 1});
    EXPECT_EQ(reshaped.shape().rank(), 4);
    EXPECT_FLOAT_EQ(reshaped.item(), 3.14f);
}

TEST_F(TensorAdvancedTest, ZeroDimensionalConsistency) {
    // Test that operations on 0-element tensors behave correctly
    auto empty1 = Tensor::empty({0}, Device::CUDA);
    auto empty2 = Tensor::empty({0, 5}, Device::CUDA);

    EXPECT_TRUE(empty1.is_valid());
    EXPECT_TRUE(empty2.is_valid());
    EXPECT_EQ(empty1.numel(), 0);
    EXPECT_EQ(empty2.numel(), 0);

    // Operations should work but produce empty results
    auto sum = empty1.add(1.0f);
    EXPECT_TRUE(sum.is_valid());
    EXPECT_EQ(sum.numel(), 0);

    // Reductions should return 0 - use scalar versions
    EXPECT_FLOAT_EQ(empty1.sum_scalar(), 0.0f);
    EXPECT_FLOAT_EQ(empty1.mean_scalar(), 0.0f);
    EXPECT_FLOAT_EQ(empty1.min_scalar(), 0.0f);
    EXPECT_FLOAT_EQ(empty1.max_scalar(), 0.0f);

    // Clone should work
    auto cloned = empty1.clone();
    EXPECT_TRUE(cloned.is_valid());
    EXPECT_EQ(cloned.numel(), 0);
}

// ============= Performance Monitoring Tests =============
// Define functor for profiling test
struct ProfileOp {
    Tensor operator()(const Tensor& t) const {
        return t.add(1.0f).mul(2.0f).sub(1.0f);
    }
};

TEST_F(TensorAdvancedTest, ProfilingSupport) {
    // Enable profiling
    Tensor::enable_profiling(true);

    auto tensor = Tensor::ones({100, 100}, Device::CUDA);

    // This should log timing information
    auto result = tensor.timed("test_operation", ProfileOp());

    EXPECT_TRUE(result.is_valid());

    // Disable profiling
    Tensor::enable_profiling(false);
}

// ============= Special Values Tests =============
TEST_F(TensorAdvancedTest, SpecialValues) {
    // Test with NaN and Inf
    std::vector<float> special_values = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        0.0f,
        -0.0f,
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::epsilon()};

    auto cpu_tensor = Tensor::empty({8}, Device::CPU);
    std::memcpy(cpu_tensor.ptr<float>(), special_values.data(), 8 * sizeof(float));

    auto cuda_tensor = cpu_tensor.to(Device::CUDA);

    // Check detection
    EXPECT_TRUE(cuda_tensor.has_nan());
    EXPECT_TRUE(cuda_tensor.has_inf());

    // Test that assert_finite throws
    EXPECT_THROW(cuda_tensor.assert_finite(), TensorError);

    // Test clamping removes inf
    auto clamped = cuda_tensor.clamp(-1e10f, 1e10f);
    EXPECT_FALSE(clamped.has_inf());

    // Note: CUDA's fminf/fmaxf handle NaN by returning the non-NaN value,
    // so NaN is effectively replaced by the clamp bounds
    // This is actually useful behavior for removing NaN values
    EXPECT_FALSE(clamped.has_nan());
}

// ============= Thread Safety Tests (Basic) =============
TEST_F(TensorAdvancedTest, ConcurrentTensorCreation) {
    const int num_threads = 10;
    const int tensors_per_thread = 100;

    std::vector<std::thread> threads;
    std::vector<std::vector<Tensor>> thread_tensors(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&thread_tensors, t, tensors_per_thread]() {
            for (int i = 0; i < tensors_per_thread; ++i) {
                thread_tensors[t].push_back(
                    Tensor::full({10, 10}, static_cast<float>(t * 100 + i), Device::CUDA));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all tensors were created correctly
    for (int t = 0; t < num_threads; ++t) {
        EXPECT_EQ(thread_tensors[t].size(), tensors_per_thread);
        for (int i = 0; i < tensors_per_thread; ++i) {
            EXPECT_TRUE(thread_tensors[t][i].is_valid());
            EXPECT_FLOAT_EQ(thread_tensors[t][i].to_vector()[0],
                            static_cast<float>(t * 100 + i));
        }
    }
}

// ============= Compatibility Tests =============
TEST_F(TensorAdvancedTest, LikeOperations) {
    auto original = Tensor::full({3, 4, 5}, 2.5f, Device::CUDA);

    auto zeros = zeros_like(original);
    EXPECT_EQ(zeros.shape(), original.shape());
    EXPECT_EQ(zeros.device(), original.device());
    EXPECT_FLOAT_EQ(zeros.sum_scalar(), 0.0f);

    auto ones = ones_like(original);
    EXPECT_EQ(ones.shape(), original.shape());
    EXPECT_EQ(ones.device(), original.device());
    EXPECT_FLOAT_EQ(ones.sum_scalar(), 60.0f); // 3*4*5 = 60
}
