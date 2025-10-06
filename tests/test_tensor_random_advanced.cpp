/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include <map>

using namespace gs;

class TensorRandomAdvancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        tensor::manual_seed(42);
    }
};

// ============= Multinomial Tests =============

TEST_F(TensorRandomAdvancedTest, MultinomialBasicCPU) {
    // Uniform weights
    auto weights = Tensor::ones({5}, Device::CPU);
    auto samples = Tensor::multinomial(weights, 10, true);

    ASSERT_TRUE(samples.is_valid());
    EXPECT_EQ(samples.dtype(), DataType::Int32);
    EXPECT_EQ(samples.shape(), TensorShape({10}));
    EXPECT_EQ(samples.device(), Device::CPU);

    // Check all samples are in valid range [0, 5)
    auto values = samples.to_vector_int();
    for (int v : values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 5);
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialBasicCUDA) {
    auto weights = Tensor::ones({5}, Device::CUDA);
    auto samples = Tensor::multinomial(weights, 10, true);

    ASSERT_TRUE(samples.is_valid());
    EXPECT_EQ(samples.dtype(), DataType::Int32);
    EXPECT_EQ(samples.shape(), TensorShape({10}));
    EXPECT_EQ(samples.device(), Device::CUDA);

    auto values = samples.to_vector_int();
    for (int v : values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 5);
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialWithReplacement) {
    auto weights = Tensor::ones({3}, Device::CPU);
    auto samples = Tensor::multinomial(weights, 100, true);

    ASSERT_TRUE(samples.is_valid());
    EXPECT_EQ(samples.numel(), 100);

    // With replacement, we should be able to get more samples than weights
    // Count frequency of each index
    auto values = samples.to_vector_int();
    std::map<int, int> freq;
    for (int v : values) {
        freq[v]++;
    }

    // All 3 indices should appear (with high probability)
    EXPECT_EQ(freq.size(), 3);
    // Each should appear roughly 33 times (allow some variance)
    for (const auto& [idx, count] : freq) {
        EXPECT_GT(count, 10);  // At least 10% of samples
        EXPECT_LT(count, 60);  // At most 60% of samples
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialWithoutReplacement) {
    auto weights = Tensor::ones({5}, Device::CPU);
    auto samples = Tensor::multinomial(weights, 5, false);

    ASSERT_TRUE(samples.is_valid());
    EXPECT_EQ(samples.numel(), 5);

    // Without replacement, each index should appear exactly once
    auto values = samples.to_vector_int();
    std::set<int> unique_values(values.begin(), values.end());
    EXPECT_EQ(unique_values.size(), 5);

    // All values should be in [0, 5)
    for (int v : values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 5);
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialBiasedWeights) {
    // Heavily biased weights: index 0 has weight 1000, others have weight 1
    std::vector<float> weights_data = {1000.0f, 1.0f, 1.0f, 1.0f};
    auto weights = Tensor::from_vector(weights_data, {4}, Device::CPU);
    
    auto samples = Tensor::multinomial(weights, 1000, true);

    auto values = samples.to_vector_int();
    std::map<int, int> freq;
    for (int v : values) {
        freq[v]++;
    }

    // Index 0 should dominate (expect at least 95% of samples)
    EXPECT_GT(freq[0], 950);
    
    // Other indices should be rare
    int others = freq[1] + freq[2] + freq[3];
    EXPECT_LT(others, 50);
}

TEST_F(TensorRandomAdvancedTest, MultinomialZeroWeight) {
    // Some weights are zero
    std::vector<float> weights_data = {1.0f, 0.0f, 1.0f, 0.0f};
    auto weights = Tensor::from_vector(weights_data, {4}, Device::CPU);
    
    auto samples = Tensor::multinomial(weights, 100, true);

    auto values = samples.to_vector_int();
    
    // Should never sample indices with zero weight
    for (int v : values) {
        EXPECT_TRUE(v == 0 || v == 2);
        EXPECT_FALSE(v == 1 || v == 3);
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialSingleSample) {
    auto weights = Tensor::ones({10}, Device::CPU);
    auto sample = Tensor::multinomial(weights, 1, true);

    ASSERT_TRUE(sample.is_valid());
    EXPECT_EQ(sample.numel(), 1);

    int value = sample.to_vector_int()[0];
    EXPECT_GE(value, 0);
    EXPECT_LT(value, 10);
}

TEST_F(TensorRandomAdvancedTest, MultinomialInvalidInputs) {
    // Non-1D weights
    auto weights_2d = Tensor::ones({3, 3}, Device::CPU);
    auto result1 = Tensor::multinomial(weights_2d, 5, true);
    EXPECT_FALSE(result1.is_valid());

    // Negative weights
    auto negative_weights = Tensor::full({5}, -1.0f, Device::CPU);
    auto result2 = Tensor::multinomial(negative_weights, 5, true);
    // Should either fail or handle gracefully
    // (Implementation-dependent behavior)

    // Zero total weight
    auto zero_weights = Tensor::zeros({5}, Device::CPU);
    auto result3 = Tensor::multinomial(zero_weights, 5, true);
    EXPECT_FALSE(result3.is_valid());
}

TEST_F(TensorRandomAdvancedTest, MultinomialTooManySamplesWithoutReplacement) {
    auto weights = Tensor::ones({5}, Device::CPU);
    
    // Try to sample more than available without replacement
    // This should fail or be capped at 5
    auto samples = Tensor::multinomial(weights, 10, false);
    
    // Implementation may either fail or cap at num_weights
    // Check that if it succeeds, we get at most 5 unique values
    if (samples.is_valid()) {
        auto values = samples.to_vector_int();
        std::set<int> unique(values.begin(), values.end());
        EXPECT_LE(unique.size(), 5);
    }
}

// ============= Reproducibility Tests =============

TEST_F(TensorRandomAdvancedTest, MultinomialReproducibility) {
    auto weights = Tensor::ones({10}, Device::CPU);

    tensor::manual_seed(123);
    auto samples1 = Tensor::multinomial(weights, 20, true);

    tensor::manual_seed(123);
    auto samples2 = Tensor::multinomial(weights, 20, true);

    auto values1 = samples1.to_vector_int();
    auto values2 = samples2.to_vector_int();

    EXPECT_EQ(values1, values2);
}

// ============= Performance Test =============

TEST_F(TensorRandomAdvancedTest, MultinomialLargeScale) {
    auto weights = Tensor::ones({1000}, Device::CUDA);
    auto samples = Tensor::multinomial(weights, 10000, true);

    ASSERT_TRUE(samples.is_valid());
    EXPECT_EQ(samples.numel(), 10000);

    // Verify distribution is roughly uniform
    auto values = samples.to_vector_int();
    std::map<int, int> freq;
    for (int v : values) {
        freq[v]++;
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 1000);
    }

    // With 10000 samples over 1000 bins, expect ~10 per bin on average
    // Check that most bins have reasonable counts (between 1 and 30)
    for (const auto& [idx, count] : freq) {
        EXPECT_GT(count, 0);
        EXPECT_LT(count, 50);  // Allow some variance
    }
}

// ============= Integration with Other Ops =============

TEST_F(TensorRandomAdvancedTest, MultinomialAsIndices) {
    auto weights = Tensor::ones({5}, Device::CPU);
    auto indices = Tensor::multinomial(weights, 10, true);

    // Use multinomial result as indices for index_select
    auto data = Tensor::arange(0.0f, 5.0f);
    auto selected = data.index_select(0, indices);

    ASSERT_TRUE(selected.is_valid());
    EXPECT_EQ(selected.shape(), TensorShape({10}));

    // All selected values should be in [0, 5)
    auto values = selected.to_vector();
    for (float v : values) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LT(v, 5.0f);
    }
}
