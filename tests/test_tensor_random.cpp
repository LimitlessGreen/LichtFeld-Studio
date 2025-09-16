/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <torch/torch.h>

using namespace gs;

class TensorRandomTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set same seed for both libraries
        torch::manual_seed(42);
        tensor::manual_seed(42);
    }

    // Helper to check if values are in expected range
    bool check_range(const Tensor& tensor, float min, float max) {
        auto values = tensor.to_vector();
        for (float val : values) {
            if (val < min || val > max) {
                return false;
            }
        }
        return true;
    }

    // Helper to compute mean and std
    std::pair<float, float> compute_stats(const Tensor& tensor) {
        auto values = tensor.to_vector();
        float sum = std::accumulate(values.begin(), values.end(), 0.0f);
        float mean = sum / values.size();

        float sq_sum = 0.0f;
        for (float val : values) {
            float diff = val - mean;
            sq_sum += diff * diff;
        }
        float std = std::sqrt(sq_sum / values.size());

        return {mean, std};
    }

    // Check if distribution parameters are approximately correct
    bool check_distribution(const Tensor& tensor, float expected_mean,
                            float expected_std, float tolerance = 0.1f) {
        auto [mean, std] = compute_stats(tensor);
        return std::abs(mean - expected_mean) < tolerance &&
               std::abs(std - expected_std) < tolerance;
    }
};

// ============= Uniform Distribution Tests =============

TEST_F(TensorRandomTest, Rand) {
    // Test basic rand (uniform [0, 1))
    auto our_tensor = Tensor::rand({100, 100}, Device::CUDA);

    // Check all values are in [0, 1)
    EXPECT_TRUE(check_range(our_tensor, 0.0f, 1.0f));

    // Check mean is approximately 0.5
    auto [mean, std] = compute_stats(our_tensor);
    EXPECT_NEAR(mean, 0.5f, 0.05f);

    // For uniform [0,1], std should be approximately sqrt(1/12) ≈ 0.289
    EXPECT_NEAR(std, 0.289f, 0.05f);
}

TEST_F(TensorRandomTest, UniformCustomRange) {
    // Test uniform with custom range
    float low = -5.0f;
    float high = 10.0f;

    auto our_tensor = Tensor::uniform({1000}, low, high, Device::CUDA);

    // Check all values are in range
    EXPECT_TRUE(check_range(our_tensor, low, high));

    // Check mean and std
    float expected_mean = (low + high) / 2.0f;
    float expected_std = (high - low) / std::sqrt(12.0f);

    auto [mean, std] = compute_stats(our_tensor);
    EXPECT_NEAR(mean, expected_mean, 0.5f);
    EXPECT_NEAR(std, expected_std, 0.5f);
}

TEST_F(TensorRandomTest, UniformInPlace) {
    // Test in-place uniform
    auto tensor = Tensor::zeros({500}, Device::CUDA);
    tensor.uniform_(2.0f, 4.0f);

    // Check range
    EXPECT_TRUE(check_range(tensor, 2.0f, 4.0f));

    // Check distribution
    auto [mean, std] = compute_stats(tensor);
    EXPECT_NEAR(mean, 3.0f, 0.1f);
}

// ============= Normal Distribution Tests =============

TEST_F(TensorRandomTest, Randn) {
    // Test standard normal distribution
    auto our_tensor = Tensor::randn({1000, 10}, Device::CUDA);

    // Check mean and std
    auto [mean, std] = compute_stats(our_tensor);
    EXPECT_NEAR(mean, 0.0f, 0.05f);
    EXPECT_NEAR(std, 1.0f, 0.05f);

    // Most values should be within 3 standard deviations
    auto values = our_tensor.to_vector();
    int outliers = 0;
    for (float val : values) {
        if (std::abs(val) > 3.0f) {
            outliers++;
        }
    }
    // Expect less than 1% outliers beyond 3 sigma
    EXPECT_LT(outliers, values.size() * 0.01);
}

TEST_F(TensorRandomTest, NormalCustomParams) {
    // Test normal with custom mean and std
    float target_mean = 5.0f;
    float target_std = 2.0f;

    auto our_tensor = Tensor::normal({2000}, target_mean, target_std, Device::CUDA);

    auto [mean, std] = compute_stats(our_tensor);
    EXPECT_NEAR(mean, target_mean, 0.1f);
    EXPECT_NEAR(std, target_std, 0.1f);
}

TEST_F(TensorRandomTest, NormalInPlace) {
    // Test in-place normal
    auto tensor = Tensor::empty({1000}, Device::CUDA);
    tensor.normal_(10.0f, 3.0f);

    auto [mean, std] = compute_stats(tensor);
    EXPECT_NEAR(mean, 10.0f, 0.2f);
    EXPECT_NEAR(std, 3.0f, 0.2f);
}

// ============= Integer Random Tests =============

TEST_F(TensorRandomTest, RandInt) {
    // Test random integers
    int low = 0;
    int high = 10;

    // FIXED: Specify DataType::Float32 explicitly since to_vector() only supports float32
    auto our_tensor = Tensor::randint({1000}, low, high, Device::CUDA, DataType::Float32);

    // Check all values are integers in [low, high)
    auto values = our_tensor.to_vector();
    for (float val : values) {
        EXPECT_GE(val, low);
        EXPECT_LT(val, high);
        EXPECT_FLOAT_EQ(val, std::floor(val)); // Check it's an integer
    }

    // Check roughly uniform distribution
    std::vector<int> counts(high - low, 0);
    for (float val : values) {
        counts[static_cast<int>(val) - low]++;
    }

    // Each value should appear roughly 100 times (1000 / 10)
    for (int count : counts) {
        EXPECT_GT(count, 50);  // At least 50
        EXPECT_LT(count, 150); // At most 150
    }
}

TEST_F(TensorRandomTest, RandIntNegative) {
    // Test with negative range
    int low = -5;
    int high = 5;

    // FIXED: Specify DataType::Float32 for consistency with to_vector()
    auto tensor = Tensor::randint({500}, low, high, Device::CUDA, DataType::Float32);

    // Check range
    auto values = tensor.to_vector();
    for (float val : values) {
        EXPECT_GE(val, low);
        EXPECT_LT(val, high);
        EXPECT_FLOAT_EQ(val, std::floor(val));
    }
}

// ============= Bernoulli Tests =============

TEST_F(TensorRandomTest, Bernoulli) {
    // Test Bernoulli distribution
    float p = 0.7f;

    auto tensor = Tensor::bernoulli({10000}, p, Device::CUDA);

    // Check all values are 0 or 1
    auto values = tensor.to_vector();
    for (float val : values) {
        EXPECT_TRUE(val == 0.0f || val == 1.0f);
    }

    // Check probability
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    float observed_p = sum / values.size();
    EXPECT_NEAR(observed_p, p, 0.02f);
}

TEST_F(TensorRandomTest, BernoulliExtreme) {
    // Test extreme probabilities
    auto tensor0 = Tensor::bernoulli({100}, 0.0f, Device::CUDA);
    auto tensor1 = Tensor::bernoulli({100}, 1.0f, Device::CUDA);

    // All should be 0
    auto values0 = tensor0.to_vector();
    for (float val : values0) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }

    // All should be 1
    auto values1 = tensor1.to_vector();
    for (float val : values1) {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }
}

// ============= Seed Tests =============

TEST_F(TensorRandomTest, ManualSeed) {
    // Test that manual seed produces reproducible results
    tensor::manual_seed(12345);
    auto tensor1 = Tensor::randn({100}, Device::CUDA);

    tensor::manual_seed(12345);
    auto tensor2 = Tensor::randn({100}, Device::CUDA);

    // Should produce identical results
    EXPECT_TRUE(tensor1.all_close(tensor2));

    // Different seed should produce different results
    tensor::manual_seed(54321);
    auto tensor3 = Tensor::randn({100}, Device::CUDA);

    EXPECT_FALSE(tensor1.all_close(tensor3));
}

// ============= Like Operations Tests =============

TEST_F(TensorRandomTest, RandLike) {
    auto original = Tensor::zeros({3, 4, 5}, Device::CUDA);

    auto random = tensor::rand_like(original);
    EXPECT_EQ(random.shape(), original.shape());
    EXPECT_EQ(random.device(), original.device());
    EXPECT_TRUE(check_range(random, 0.0f, 1.0f));
}

TEST_F(TensorRandomTest, RandnLike) {
    auto original = Tensor::ones({10, 10}, Device::CUDA);

    auto random = tensor::randn_like(original);
    EXPECT_EQ(random.shape(), original.shape());
    EXPECT_EQ(random.device(), original.device());

    // Should be roughly standard normal
    auto [mean, std] = compute_stats(random);
    EXPECT_NEAR(mean, 0.0f, 0.3f);
    EXPECT_NEAR(std, 1.0f, 0.3f);
}

// ============= CPU vs CUDA Consistency =============

TEST_F(TensorRandomTest, CPUCUDAConsistency) {
    // Test that CPU and CUDA produce similar distributions
    tensor::manual_seed(999);
    auto cpu_tensor = Tensor::randn({1000}, Device::CPU);

    tensor::manual_seed(888);
    auto cuda_tensor = Tensor::randn({1000}, Device::CUDA);

    auto [cpu_mean, cpu_std] = compute_stats(cpu_tensor);
    auto [cuda_mean, cuda_std] = compute_stats(cuda_tensor);

    // Both should be roughly standard normal
    EXPECT_NEAR(cpu_mean, 0.0f, 0.1f);
    EXPECT_NEAR(cpu_std, 1.0f, 0.1f);
    EXPECT_NEAR(cuda_mean, 0.0f, 0.1f);
    EXPECT_NEAR(cuda_std, 1.0f, 0.1f);
}

// ============= Shape Tests =============

TEST_F(TensorRandomTest, VariousShapes) {
    // Test with various tensor shapes
    std::vector<std::vector<size_t>> shapes = {
        {10},
        {5, 5},
        {2, 3, 4},
        {1, 1, 100},
        {10, 10, 10, 10}};

    for (const auto& shape : shapes) {
        auto tensor = Tensor::randn(TensorShape(shape), Device::CUDA);
        EXPECT_EQ(tensor.shape().dims(), shape);

        size_t expected_elements = 1;
        for (size_t dim : shape) {
            expected_elements *= dim;
        }
        EXPECT_EQ(tensor.numel(), expected_elements);
    }
}

// ============= Edge Cases =============

TEST_F(TensorRandomTest, EmptyTensor) {
    // Test with 0-element tensor
    auto empty = Tensor::randn({0}, Device::CUDA);
    EXPECT_TRUE(empty.is_valid());
    EXPECT_EQ(empty.numel(), 0);
}

TEST_F(TensorRandomTest, SingleElement) {
    // Test with single element
    auto single = Tensor::uniform({1}, -1.0f, 1.0f, Device::CUDA);
    EXPECT_EQ(single.numel(), 1);

    float val = single.item();
    EXPECT_GE(val, -1.0f);
    EXPECT_LE(val, 1.0f);
}

TEST_F(TensorRandomTest, LargeTensor) {
    // Test with large tensor
    const size_t large_size = 1000000;
    auto large = Tensor::randn({large_size}, Device::CUDA);

    EXPECT_EQ(large.numel(), large_size);

    // Sample a subset to check distribution
    auto sample_size = 10000;
    std::vector<float> all_values = large.to_vector();
    std::vector<float> sample(all_values.begin(), all_values.begin() + sample_size);

    float sum = std::accumulate(sample.begin(), sample.end(), 0.0f);
    float mean = sum / sample.size();

    EXPECT_NEAR(mean, 0.0f, 0.05f);
}

// ============= Performance Test =============

TEST_F(TensorRandomTest, RandomGenerationSpeed) {
    // Test speed of random generation
    const size_t n = 10000000; // 10M elements

    auto start = std::chrono::high_resolution_clock::now();
    auto tensor = Tensor::randn({n}, Device::CUDA);
    cudaDeviceSynchronize();
    auto duration = std::chrono::high_resolution_clock::now() - start;

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    std::cout << "Generated " << n << " random numbers in " << ms << " ms" << std::endl;

    // Just ensure it completes without error
    EXPECT_TRUE(tensor.is_valid());
    EXPECT_EQ(tensor.numel(), n);
}
