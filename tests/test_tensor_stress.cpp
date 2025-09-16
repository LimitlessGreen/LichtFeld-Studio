/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <thread>

using namespace gs;
using namespace gs::tensor;

class TensorStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Record initial memory
        cudaMemGetInfo(&initial_free_mem_, &total_mem_);
        gen_.seed(42);
    }

    void TearDown() override {
        // Force synchronization
        cudaDeviceSynchronize();

        // Check for memory leaks
        size_t current_free_mem, current_total;
        cudaMemGetInfo(&current_free_mem, &current_total);

        // Allow 10MB tolerance for CUDA internal allocations
        size_t tolerance = 10 * 1024 * 1024;
        EXPECT_NEAR(current_free_mem, initial_free_mem_, tolerance)
            << "Possible memory leak detected in stress test";
    }

    size_t initial_free_mem_;
    size_t total_mem_;
    std::mt19937 gen_;
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};
};

TEST_F(TensorStressTest, MaxMemoryAllocation) {
    // Try to allocate tensors until we run out of memory
    std::vector<Tensor> tensors;
    const size_t chunk_size = 10 * 1024 * 1024; // 10M floats = 40MB per tensor

    size_t total_allocated = 0;
    bool allocation_failed = false;

    for (int i = 0; i < 100; ++i) { // Max 100 tensors to avoid hanging
        try {
            auto t = Tensor::empty({chunk_size}, Device::CUDA);
            if (!t.is_valid()) {
                allocation_failed = true;
                break;
            }
            tensors.push_back(std::move(t));
            total_allocated += chunk_size * sizeof(float);
        } catch (...) {
            allocation_failed = true;
            break;
        }

        // Check available memory
        size_t free_mem, total;
        cudaMemGetInfo(&free_mem, &total);
        if (free_mem < chunk_size * sizeof(float) * 2) {
            // Stop before we completely run out
            break;
        }
    }

    EXPECT_GT(tensors.size(), 0) << "Should be able to allocate at least one tensor";

    // Clear tensors to free memory
    tensors.clear();

    // Memory should be freed
    size_t free_after, total_after;
    cudaMemGetInfo(&free_after, &total_after);
    EXPECT_GT(free_after, initial_free_mem_ - 100 * 1024 * 1024)
        << "Memory not properly freed after clearing tensors";
}

TEST_F(TensorStressTest, RapidAllocationDeallocation) {
    // Rapidly allocate and deallocate tensors
    const int iterations = 1000;
    const size_t tensor_size = 1024 * 1024; // 1M elements

    for (int i = 0; i < iterations; ++i) {
        auto t = Tensor::zeros({tensor_size}, Device::CUDA);
        EXPECT_TRUE(t.is_valid());
        // Tensor destroyed here
    }

    // Check memory is stable
    size_t free_after, total_after;
    cudaMemGetInfo(&free_after, &total_after);
    EXPECT_NEAR(free_after, initial_free_mem_, 50 * 1024 * 1024)
        << "Memory leak detected after rapid allocation/deallocation";
}

TEST_F(TensorStressTest, DeepOperationChain) {
    // Create a very deep chain of operations
    auto tensor = Tensor::ones({100, 100}, Device::CUDA);

    const int chain_length = 1000;
    for (int i = 0; i < chain_length; ++i) {
        // Alternate operations to avoid overflow/underflow
        if (i % 4 == 0) {
            tensor = tensor.add(0.001f);
        } else if (i % 4 == 1) {
            tensor = tensor.mul(1.001f);
        } else if (i % 4 == 2) {
            tensor = tensor.sub(0.001f);
        } else {
            tensor = tensor.div(1.001f);
        }

        // Every 100 operations, check validity
        if (i % 100 == 99) {
            EXPECT_TRUE(tensor.is_valid());
            EXPECT_FALSE(tensor.has_nan());
            EXPECT_FALSE(tensor.has_inf());
        }
    }

    EXPECT_TRUE(tensor.is_valid());
}

TEST_F(TensorStressTest, LargeMatrixOperations) {
    const size_t dim = 2048; // 2048x2048 = 4M elements = 16MB

    auto a = Tensor::empty({dim, dim}, Device::CUDA);
    auto b = Tensor::empty({dim, dim}, Device::CUDA);

    // Fill with random values
    std::vector<float> data(dim * dim);
    for (auto& val : data) {
        val = dist_(gen_);
    }
    cudaMemcpy(a.ptr<float>(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

    for (auto& val : data) {
        val = dist_(gen_);
    }
    cudaMemcpy(b.ptr<float>(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Perform operations
    auto start = std::chrono::high_resolution_clock::now();

    auto c = a.add(b);
    auto d = c.mul(2.0f);
    auto e = d.sub(a);
    auto result = e.div(b.add(1.0f)); // Add 1 to avoid division by zero

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_TRUE(result.is_valid());
    EXPECT_FALSE(result.has_nan());
    EXPECT_FALSE(result.has_inf());

    std::cout << "Large matrix operations took: " << duration.count() << " ms" << std::endl;
}

TEST_F(TensorStressTest, ManySmallTensors) {
    // Create many small tensors
    const int num_tensors = 10000;
    std::vector<Tensor> tensors;
    tensors.reserve(num_tensors);

    for (int i = 0; i < num_tensors; ++i) {
        tensors.push_back(Tensor::full({10}, static_cast<float>(i), Device::CUDA));
    }

    // Verify all tensors
    for (int i = 0; i < num_tensors; ++i) {
        EXPECT_TRUE(tensors[i].is_valid());
        EXPECT_FLOAT_EQ(tensors[i].to_vector()[0], static_cast<float>(i));
    }

    // Perform operation on all
    for (auto& t : tensors) {
        t = t.add(1.0f);
    }

    // Verify again
    for (int i = 0; i < num_tensors; ++i) {
        EXPECT_FLOAT_EQ(tensors[i].to_vector()[0], static_cast<float>(i + 1));
    }
}

TEST_F(TensorStressTest, ViewStress) {
    // Create complex view hierarchies
    auto original = Tensor::ones({24, 24}, Device::CUDA);

    std::vector<Tensor> views;

    // Create many different views
    views.push_back(original.view({576}));
    views.push_back(original.view({4, 144}));
    views.push_back(original.view({8, 72}));
    views.push_back(original.view({16, 36}));
    views.push_back(original.view({2, 12, 24}));
    views.push_back(original.view({3, 8, 24}));
    views.push_back(original.view({4, 6, 24}));
    views.push_back(original.view({2, 2, 144}));

    // Modify original
    original.fill_(2.0f);

    // All views should see the change
    for (const auto& view : views) {
        EXPECT_FLOAT_EQ(view.to_vector()[0], 2.0f);
    }

    // Create views of views
    auto view_of_view = views[0].view({24, 24});
    EXPECT_TRUE(view_of_view.is_valid());
    EXPECT_FLOAT_EQ(view_of_view.to_vector()[0], 2.0f);
}

TEST_F(TensorStressTest, SliceStress) {
    const size_t size = 1000;
    auto tensor = Tensor::empty({size, size}, Device::CUDA);

    // Fill with row indices for testing
    std::vector<float> data(size * size);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            data[i * size + j] = static_cast<float>(i);
        }
    }
    cudaMemcpy(tensor.ptr<float>(), data.data(), data.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    // Create many overlapping slices
    std::vector<Tensor> slices;
    for (size_t i = 0; i < size - 10; i += 10) {
        slices.push_back(tensor.slice(0, i, std::min(i + 20, size)));
    }

    // Verify slices
    for (size_t i = 0; i < slices.size(); ++i) {
        auto expected_rows = std::min(size_t(20), size - i * 10);
        EXPECT_EQ(slices[i].shape()[0], expected_rows);
        EXPECT_EQ(slices[i].shape()[1], size);

        // Check first value in slice
        auto first_val = slices[i].to_vector()[0];
        EXPECT_FLOAT_EQ(first_val, static_cast<float>(i * 10));
    }
}

TEST_F(TensorStressTest, ReductionStress) {
    // Test reductions on large tensors
    const size_t size = 1024 * 1024; // 1M elements

    auto tensor = Tensor::empty({size}, Device::CUDA);

    // Fill with known pattern
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i % 100); // Repeating pattern 0-99
    }
    cudaMemcpy(tensor.ptr<float>(), data.data(), data.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    // Test reductions
    float sum = tensor.sum();
    float mean = tensor.mean();
    float min_val = tensor.min();
    float max_val = tensor.max();

    // Expected values
    float expected_sum = (size / 100) * (99 * 100 / 2.0f) + ((size % 100) * ((size % 100) - 1) / 2.0f);
    float expected_mean = expected_sum / size;

    EXPECT_NEAR(sum, expected_sum, expected_sum * 1e-5f);
    EXPECT_NEAR(mean, expected_mean, 1e-3f);
    EXPECT_FLOAT_EQ(min_val, 0.0f);
    EXPECT_FLOAT_EQ(max_val, 99.0f);
}

TEST_F(TensorStressTest, MixedDeviceStress) {
    // Stress test device transfers
    const int iterations = 100;
    const size_t size = 1024 * 1024; // 1MB

    for (int i = 0; i < iterations; ++i) {
        // Create on CPU
        auto cpu_tensor = Tensor::full({size}, static_cast<float>(i), Device::CPU);

        // Transfer to CUDA
        auto cuda_tensor = cpu_tensor.to(Device::CUDA);

        // Perform operation
        auto result = cuda_tensor.add(1.0f).mul(2.0f);

        // Transfer back
        auto cpu_result = result.to(Device::CPU);

        // Verify
        EXPECT_FLOAT_EQ(cpu_result.to_vector()[0], (static_cast<float>(i) + 1.0f) * 2.0f);
    }
}

TEST_F(TensorStressTest, NumericalStability) {
    // Test numerical stability with extreme values
    auto tensor = Tensor::empty({1000}, Device::CUDA);

    // Fill with exponentially growing values
    std::vector<float> data(1000);
    for (size_t i = 0; i < 1000; ++i) {
        data[i] = std::pow(1.01f, static_cast<float>(i));
    }
    cudaMemcpy(tensor.ptr<float>(), data.data(), data.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    // Test operations maintain stability
    auto log_tensor = tensor.log();
    EXPECT_FALSE(log_tensor.has_nan());
    EXPECT_FALSE(log_tensor.has_inf());

    auto exp_log = log_tensor.exp();
    EXPECT_TRUE(tensor.all_close(exp_log, 1e-3f, 1e-3f));

    // Test with very small values
    auto small = Tensor::full({1000}, 1e-30f, Device::CUDA);
    auto sqrt_small = small.sqrt();
    EXPECT_FALSE(sqrt_small.has_nan());

    // Test normalization with extreme values
    auto normalized = tensor.normalize();
    EXPECT_FALSE(normalized.has_nan());
    EXPECT_FALSE(normalized.has_inf());
    EXPECT_NEAR(normalized.mean(), 0.0f, 1e-5f);
    EXPECT_NEAR(normalized.std(), 1.0f, 1e-5f);
}

TEST_F(TensorStressTest, ConcurrentOperations) {
    // Test concurrent operations on different tensors
    const int num_threads = 8;
    const int ops_per_thread = 100;

    std::vector<std::thread> threads;
    std::vector<Tensor> results(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&results, t, ops_per_thread]() {
            auto tensor = Tensor::full({100, 100}, static_cast<float>(t), Device::CUDA);

            for (int i = 0; i < ops_per_thread; ++i) {
                tensor = tensor.add(0.1f).mul(1.01f).sub(0.05f);
            }

            // Use move assignment instead of copy
            results[t] = std::move(tensor);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all results are valid
    for (const auto& result : results) {
        EXPECT_TRUE(result.is_valid());
        EXPECT_FALSE(result.has_nan());
        EXPECT_FALSE(result.has_inf());
    }
}
