/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include "kernels/kmeans.cuh"
#include "kernels/kmeans_new.cuh"
#include "core/tensor.hpp"

namespace {

constexpr float FLOAT_TOLERANCE = 1e-3f;  // Slightly relaxed for comparison
constexpr float INERTIA_TOLERANCE = 0.15f; // 15% tolerance for inertia comparison

// Helper to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error); \
    } while(0)

// Helper to create random test data for torch
torch::Tensor create_random_data_torch(int n, int d, float min_val = 0.0f, float max_val = 1.0f) {
    auto data = torch::rand({n, d}, torch::kCUDA) * (max_val - min_val) + min_val;
    return data;
}

// Helper to create random test data for gs::Tensor
gs::Tensor create_random_data_tensor(int n, int d, float min_val = 0.0f, float max_val = 1.0f) {
    auto data = gs::Tensor::rand({static_cast<size_t>(n), static_cast<size_t>(d)}, 
                                 gs::Device::CUDA, gs::DataType::Float32);
    data = data * (max_val - min_val) + min_val;
    return data;
}

// Helper to create clustered test data for torch
torch::Tensor create_clustered_data_torch(int n, int k, int d, float separation = 5.0f) {
    auto data = torch::zeros({n, d}, torch::kCUDA);
    int points_per_cluster = n / k;
    
    for (int c = 0; c < k; ++c) {
        int start = c * points_per_cluster;
        int end = (c == k - 1) ? n : (c + 1) * points_per_cluster;
        
        auto cluster_center = torch::full({d}, c * separation, torch::kCUDA);
        auto noise = torch::randn({end - start, d}, torch::kCUDA) * 0.5f;
        
        data.slice(0, start, end) = cluster_center + noise;
    }
    
    return data;
}

// Helper to create clustered test data for gs::Tensor
gs::Tensor create_clustered_data_tensor(int n, int k, int d, float separation = 5.0f) {
    auto data = gs::Tensor::zeros({static_cast<size_t>(n), static_cast<size_t>(d)}, 
                                   gs::Device::CUDA, gs::DataType::Float32);
    int points_per_cluster = n / k;
    
    for (int c = 0; c < k; ++c) {
        int start = c * points_per_cluster;
        int end = (c == k - 1) ? n : (c + 1) * points_per_cluster;
        
        auto cluster_center = gs::Tensor::full({static_cast<size_t>(d)}, 
                                               c * separation, 
                                               gs::Device::CUDA, 
                                               gs::DataType::Float32);
        auto noise = gs::Tensor::randn({static_cast<size_t>(end - start), 
                                       static_cast<size_t>(d)}, 
                                       gs::Device::CUDA, 
                                       gs::DataType::Float32) * 0.5f;
        
        // Get CPU versions for easier access
        auto cluster_center_cpu = cluster_center.cpu();
        auto noise_cpu = noise.cpu();
        auto data_cpu = data.cpu();
        
        // Update slice
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < d; ++j) {
                float center_val = static_cast<float>(cluster_center_cpu[j]);
                float noise_val = static_cast<float>(noise_cpu[i - start][j]);
                data_cpu[i][j] = center_val + noise_val;
            }
        }
        
        // Copy back to GPU
        data.copy_from(data_cpu.cuda());
    }
    
    return data;
}

// Convert torch::Tensor to gs::Tensor
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

// Convert gs::Tensor to torch::Tensor
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

// Helper function to compute inertia (sum of squared distances to centroids)
float compute_inertia_torch(const torch::Tensor& data, 
                            const torch::Tensor& centroids,
                            const torch::Tensor& labels) {
    auto n = data.size(0);
    auto d = data.size(1);
    
    float inertia = 0.0f;
    auto data_cpu = data.cpu();
    auto centroids_cpu = centroids.cpu();
    auto labels_cpu = labels.cpu();
    
    for (int i = 0; i < n; ++i) {
        int cluster = labels_cpu[i].item<int>();
        float dist = 0.0f;
        for (int j = 0; j < d; ++j) {
            float diff = data_cpu[i][j].item<float>() - centroids_cpu[cluster][j].item<float>();
            dist += diff * diff;
        }
        inertia += dist;
    }
    
    return inertia;
}

float compute_inertia_tensor(const gs::Tensor& data, 
                             const gs::Tensor& centroids,
                             const gs::Tensor& labels) {
    auto n = data.shape()[0];
    auto d = data.shape()[1];
    
    float inertia = 0.0f;
    auto data_cpu = data.cpu();
    auto centroids_cpu = centroids.cpu();
    auto labels_cpu = labels.cpu();
    
    for (size_t i = 0; i < n; ++i) {
        int cluster = static_cast<int>(static_cast<float>(labels_cpu[i]));
        float dist = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            float data_val = static_cast<float>(data_cpu[i][j]);
            float centroid_val = static_cast<float>(centroids_cpu[cluster][j]);
            float diff = data_val - centroid_val;
            dist += diff * diff;
        }
        inertia += dist;
    }
    
    return inertia;
}

// Helper to verify all labels are valid
bool verify_labels_valid(const torch::Tensor& labels, int k) {
    auto labels_cpu = labels.cpu();
    for (int i = 0; i < labels.size(0); ++i) {
        int label = labels_cpu[i].item<int>();
        if (label < 0 || label >= k) {
            return false;
        }
    }
    return true;
}

bool verify_labels_valid(const gs::Tensor& labels, int k) {
    auto labels_cpu = labels.cpu();
    for (size_t i = 0; i < labels.numel(); ++i) {
        int label = static_cast<int>(static_cast<float>(labels_cpu[i]));
        if (label < 0 || label >= k) {
            return false;
        }
    }
    return true;
}

// Helper to count unique labels
int count_unique_labels(const torch::Tensor& labels) {
    auto labels_cpu = labels.cpu();
    std::vector<int> label_vec;
    for (int i = 0; i < labels.size(0); ++i) {
        label_vec.push_back(labels_cpu[i].item<int>());
    }
    std::sort(label_vec.begin(), label_vec.end());
    auto last = std::unique(label_vec.begin(), label_vec.end());
    return std::distance(label_vec.begin(), last);
}

int count_unique_labels(const gs::Tensor& labels) {
    auto labels_cpu = labels.cpu();
    std::vector<int> label_vec;
    for (size_t i = 0; i < labels.numel(); ++i) {
        int label = static_cast<int>(static_cast<float>(labels_cpu[i]));
        label_vec.push_back(label);
    }
    std::sort(label_vec.begin(), label_vec.end());
    auto last = std::unique(label_vec.begin(), label_vec.end());
    return std::distance(label_vec.begin(), last);
}

} // anonymous namespace

class KMeansComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA availability
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        
        // Set random seed for reproducibility
        torch::manual_seed(42);
        torch::cuda::manual_seed(42);
        gs::Tensor::manual_seed(42);
    }
};

// ============================================================================
// Basic Functionality Comparison Tests
// ============================================================================

TEST_F(KMeansComparisonTest, BasicClustering2D_Comparison) {
    const int n_points = 1000;
    const int n_dims = 2;
    const int k = 3;
    
    // Create shared random data
    auto data_torch = create_clustered_data_torch(n_points, k, n_dims, 10.0f);
    auto data_tensor = torch_to_tensor(data_torch);
    
    // Run both implementations
    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 50, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 50, 1e-4f);
    
    // Verify dimensions match
    EXPECT_EQ(centroids_old.size(0), k);
    EXPECT_EQ(centroids_new.shape()[0], k);
    EXPECT_EQ(labels_old.size(0), n_points);
    EXPECT_EQ(labels_new.shape()[0], n_points);
    
    // Verify labels are valid
    EXPECT_TRUE(verify_labels_valid(labels_old, k));
    EXPECT_TRUE(verify_labels_valid(labels_new, k));
    
    // Compare inertias (should be similar)
    float inertia_old = compute_inertia_torch(data_torch, centroids_old, labels_old);
    float inertia_new = compute_inertia_tensor(data_tensor, centroids_new, labels_new);
    
    float rel_diff = std::abs(inertia_old - inertia_new) / std::max(inertia_old, inertia_new);
    EXPECT_LT(rel_diff, INERTIA_TOLERANCE) 
        << "Inertia mismatch: old=" << inertia_old << " new=" << inertia_new;
}

TEST_F(KMeansComparisonTest, BasicClustering1D_Comparison) {
    const int n_points = 500;
    const int k = 256;
    
    // Create shared random data
    auto data_torch = create_random_data_torch(n_points, 1, 0.0f, 100.0f);
    auto data_tensor = torch_to_tensor(data_torch);
    
    // Run both implementations
    auto [centroids_old, labels_old] = gs::cuda::kmeans_1d(data_torch, k, 20);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_1d_new(data_tensor, k, 20);
    
    // Verify dimensions
    EXPECT_EQ(centroids_old.size(0), k);
    EXPECT_EQ(centroids_new.shape()[0], k);
    EXPECT_EQ(labels_old.size(0), n_points);
    EXPECT_EQ(labels_new.shape()[0], n_points);
    
    // Verify centroids are sorted for both
    auto centroids_old_cpu = centroids_old.squeeze(1).cpu();
    auto centroids_new_cpu = centroids_new.cpu();
    
    for (int i = 1; i < k; ++i) {
        EXPECT_GE(centroids_old_cpu[i].item<float>(), centroids_old_cpu[i-1].item<float>());
        EXPECT_GE(centroids_new_cpu[i][0], centroids_new_cpu[i-1][0]);
    }
    
    // Verify all labels are valid
    EXPECT_TRUE(verify_labels_valid(labels_old, k));
    EXPECT_TRUE(verify_labels_valid(labels_new, k));
}

TEST_F(KMeansComparisonTest, EmptyInput_Comparison) {
    const int k = 3;
    auto empty_data_torch = torch::empty({0, 2}, torch::kCUDA);
    auto empty_data_tensor = gs::Tensor::empty({0, 2}, gs::Device::CUDA, gs::DataType::Float32);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans(empty_data_torch, k, 10, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(empty_data_tensor, k, 10, 1e-4f);
    
    // Both should return empty results
    EXPECT_EQ(centroids_old.size(0), 0);
    EXPECT_EQ(centroids_new.shape()[0], 0);
    EXPECT_EQ(labels_old.size(0), 0);
    EXPECT_EQ(labels_new.shape()[0], 0);
}

TEST_F(KMeansComparisonTest, FewerPointsThanClusters_Comparison) {
    const int n_points = 5;
    const int n_dims = 2;
    const int k = 10;
    
    auto data_torch = create_random_data_torch(n_points, n_dims);
    auto data_tensor = torch_to_tensor(data_torch);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 10, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 10, 1e-4f);
    
    // Both should return at most n_points centroids
    EXPECT_LE(centroids_old.size(0), n_points);
    EXPECT_LE(centroids_new.shape()[0], n_points);
    EXPECT_EQ(labels_old.size(0), n_points);
    EXPECT_EQ(labels_new.shape()[0], n_points);
}

// ============================================================================
// Convergence Tests
// ============================================================================

TEST_F(KMeansComparisonTest, ConvergenceWithIterations_Comparison) {
    const int n_points = 500;
    const int n_dims = 2;
    const int k = 4;
    
    auto data_torch = create_clustered_data_torch(n_points, k, n_dims, 8.0f);
    auto data_tensor = torch_to_tensor(data_torch);
    
    // Run with different iteration counts
    auto [centroids_old_5, labels_old_5] = gs::cuda::kmeans(data_torch, k, 5, 1e-4f);
    auto [centroids_old_50, labels_old_50] = gs::cuda::kmeans(data_torch, k, 50, 1e-4f);
    
    auto [centroids_new_5, labels_new_5] = gs::cuda::kmeans_new(data_tensor, k, 5, 1e-4f);
    auto [centroids_new_50, labels_new_50] = gs::cuda::kmeans_new(data_tensor, k, 50, 1e-4f);
    
    float inertia_old_5 = compute_inertia_torch(data_torch, centroids_old_5, labels_old_5);
    float inertia_old_50 = compute_inertia_torch(data_torch, centroids_old_50, labels_old_50);
    
    float inertia_new_5 = compute_inertia_tensor(data_tensor, centroids_new_5, labels_new_5);
    float inertia_new_50 = compute_inertia_tensor(data_tensor, centroids_new_50, labels_new_50);
    
    // More iterations should give equal or better inertia for both
    EXPECT_LE(inertia_old_50, inertia_old_5 + FLOAT_TOLERANCE);
    EXPECT_LE(inertia_new_50, inertia_new_5 + FLOAT_TOLERANCE);
}

// ============================================================================
// 1D Specific Tests
// ============================================================================

TEST_F(KMeansComparisonTest, OneDimensionalOrdering_Comparison) {
    const int n_points = 256;
    const int k = 16;
    
    // Create linearly spaced data
    auto data_torch = torch::linspace(0, 255, n_points, torch::kCUDA).unsqueeze(1);
    auto data_tensor = gs::Tensor::linspace(0, 255, n_points, gs::Device::CUDA).unsqueeze(1);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans_1d(data_torch, k, 20);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_1d_new(data_tensor, k, 20);
    
    auto centroids_old_cpu = centroids_old.squeeze(1).cpu();
    auto centroids_new_cpu = centroids_new.squeeze(1).cpu();
    
    // Verify centroids are strictly increasing for both
    for (int i = 1; i < k; ++i) {
        EXPECT_GT(centroids_old_cpu[i].item<float>(), centroids_old_cpu[i-1].item<float>());
        EXPECT_GT(centroids_new_cpu[i][0], centroids_new_cpu[i-1][0]);
    }
}

TEST_F(KMeansComparisonTest, OneDimensionalWithDuplicates_Comparison) {
    const int n_points = 100;
    const int k = 10;
    
    // Create data with many duplicates
    auto data_torch = torch::floor(torch::arange(0, n_points, torch::kCUDA) / 10.0f).unsqueeze(1);
    auto data_tensor_1d = gs::Tensor::arange(0, n_points).cuda() / 10.0f;
    auto data_tensor = data_tensor_1d.floor().unsqueeze(1);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans_1d(data_torch, k, 20);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_1d_new(data_tensor, k, 20);
    
    // Should handle duplicates gracefully
    EXPECT_TRUE(verify_labels_valid(labels_old, k));
    EXPECT_TRUE(verify_labels_valid(labels_new, k));
}

// ============================================================================
// High Dimensional Tests
// ============================================================================

TEST_F(KMeansComparisonTest, HighDimensionalClustering_Comparison) {
    const int n_points = 200;
    const int n_dims = 50;
    const int k = 5;
    
    auto data_torch = create_random_data_torch(n_points, n_dims);
    auto data_tensor = torch_to_tensor(data_torch);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 30, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 30, 1e-4f);
    
    EXPECT_EQ(centroids_old.size(0), k);
    EXPECT_EQ(centroids_new.shape()[0], k);
    EXPECT_EQ(centroids_old.size(1), n_dims);
    EXPECT_EQ(centroids_new.shape()[1], n_dims);
    
    EXPECT_TRUE(verify_labels_valid(labels_old, k));
    EXPECT_TRUE(verify_labels_valid(labels_new, k));
}

TEST_F(KMeansComparisonTest, ThreeDimensionalClustering_Comparison) {
    const int n_points = 500;
    const int n_dims = 3;
    const int k = 5;
    
    auto data_torch = create_clustered_data_torch(n_points, k, n_dims, 15.0f);
    auto data_tensor = torch_to_tensor(data_torch);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 30, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 30, 1e-4f);
    
    EXPECT_EQ(centroids_old.size(0), k);
    EXPECT_EQ(centroids_new.shape()[0], k);
    EXPECT_TRUE(verify_labels_valid(labels_old, k));
    EXPECT_TRUE(verify_labels_valid(labels_new, k));
    
    int unique_old = count_unique_labels(labels_old);
    int unique_new = count_unique_labels(labels_new);
    
    EXPECT_GE(unique_old, k - 1);
    EXPECT_GE(unique_new, k - 1);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(KMeansComparisonTest, SingleCluster_Comparison) {
    const int n_points = 100;
    const int n_dims = 3;
    const int k = 1;
    
    auto data_torch = create_random_data_torch(n_points, n_dims);
    auto data_tensor = torch_to_tensor(data_torch);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 10, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 10, 1e-4f);
    
    EXPECT_EQ(centroids_old.size(0), 1);
    EXPECT_EQ(centroids_new.shape()[0], 1);
    
    auto labels_old_cpu = labels_old.cpu();
    auto labels_new_cpu = labels_new.cpu();
    
    // All points should be in cluster 0
    for (int i = 0; i < n_points; ++i) {
        EXPECT_EQ(labels_old_cpu[i].item<int>(), 0);
        EXPECT_EQ(static_cast<int>(static_cast<float>(labels_new_cpu[i])), 0);
    }
}

TEST_F(KMeansComparisonTest, VeryTightCluster_Comparison) {
    const int n_points = 100;
    const int n_dims = 2;
    const int k = 3;
    
    // Create a tight cluster
    auto data_torch = torch::randn({n_points, n_dims}, torch::kCUDA) * 0.1f + 42.0f;
    auto data_tensor = torch_to_tensor(data_torch);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 10, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 10, 1e-4f);
    
    EXPECT_GT(centroids_old.size(0), 0);
    EXPECT_GT(centroids_new.shape()[0], 0);
    EXPECT_LE(centroids_old.size(0), k);
    EXPECT_LE(centroids_new.shape()[0], k);
    
    EXPECT_TRUE(verify_labels_valid(labels_old, centroids_old.size(0)));
    EXPECT_TRUE(verify_labels_valid(labels_new, centroids_new.shape()[0]));
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(KMeansComparisonTest, LargeDataset_Comparison) {
    const int n_points = 10000;
    const int n_dims = 3;
    const int k = 8;
    
    auto data_torch = create_random_data_torch(n_points, n_dims);
    auto data_tensor = torch_to_tensor(data_torch);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 20, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 20, 1e-4f);
    
    EXPECT_EQ(centroids_old.size(0), k);
    EXPECT_EQ(centroids_new.shape()[0], k);
    EXPECT_EQ(labels_old.size(0), n_points);
    EXPECT_EQ(labels_new.shape()[0], n_points);
    
    EXPECT_TRUE(verify_labels_valid(labels_old, k));
    EXPECT_TRUE(verify_labels_valid(labels_new, k));
}

TEST_F(KMeansComparisonTest, ManyClusters_Comparison) {
    const int n_points = 5000;
    const int n_dims = 2;
    const int k = 256;
    
    auto data_torch = create_random_data_torch(n_points, n_dims);
    auto data_tensor = torch_to_tensor(data_torch);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 30, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 30, 1e-4f);
    
    EXPECT_EQ(centroids_old.size(0), k);
    EXPECT_EQ(centroids_new.shape()[0], k);
    EXPECT_TRUE(verify_labels_valid(labels_old, k));
    EXPECT_TRUE(verify_labels_valid(labels_new, k));
}

TEST_F(KMeansComparisonTest, LargeOneDimensionalDataset_Comparison) {
    const int n_points = 50000;
    const int k = 256;
    
    auto data_torch = create_random_data_torch(n_points, 1, 0.0f, 10000.0f);
    auto data_tensor = torch_to_tensor(data_torch);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans_1d(data_torch, k, 20);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_1d_new(data_tensor, k, 20);
    
    auto centroids_old_cpu = centroids_old.squeeze(1).cpu();
    auto centroids_new_cpu = centroids_new.squeeze(1).cpu();
    
    // Verify centroids are sorted for both
    for (int i = 1; i < k; ++i) {
        EXPECT_GE(centroids_old_cpu[i].item<float>(), centroids_old_cpu[i-1].item<float>());
        EXPECT_GE(centroids_new_cpu[i][0], centroids_new_cpu[i-1][0]);
    }
    
    EXPECT_TRUE(verify_labels_valid(labels_old, k));
    EXPECT_TRUE(verify_labels_valid(labels_new, k));
}

// ============================================================================
// Quality Tests
// ============================================================================

TEST_F(KMeansComparisonTest, WellSeparatedClusters_Comparison) {
    const int n_points = 600;
    const int n_dims = 2;
    const int k = 3;
    
    auto data_torch = create_clustered_data_torch(n_points, k, n_dims, 50.0f);
    auto data_tensor = torch_to_tensor(data_torch);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 50, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 50, 1e-4f);
    
    int unique_old = count_unique_labels(labels_old);
    int unique_new = count_unique_labels(labels_new);
    
    EXPECT_EQ(unique_old, k);
    EXPECT_EQ(unique_new, k);
}

TEST_F(KMeansComparisonTest, SOGTypical256Clusters_Comparison) {
    // Typical use case for SOG quantization
    const int n_points = 10000;
    const int k = 256;
    
    auto data_torch = create_random_data_torch(n_points, 1, 0.0f, 1.0f);
    auto data_tensor = torch_to_tensor(data_torch);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans_1d(data_torch, k, 20);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_1d_new(data_tensor, k, 20);
    
    EXPECT_EQ(centroids_old.size(0), 256);
    EXPECT_EQ(centroids_new.shape()[0], 256);
    EXPECT_TRUE(verify_labels_valid(labels_old, 256));
    EXPECT_TRUE(verify_labels_valid(labels_new, 256));
    
    // Verify sorted for both
    auto centroids_old_cpu = centroids_old.squeeze(1).cpu();
    auto centroids_new_cpu = centroids_new.squeeze(1).cpu();
    
    for (int i = 1; i < 256; ++i) {
        EXPECT_GE(centroids_old_cpu[i].item<float>(), centroids_old_cpu[i-1].item<float>());
        EXPECT_GE(centroids_new_cpu[i][0], centroids_new_cpu[i-1][0]);
    }
}

TEST_F(KMeansComparisonTest, InertiaComparison_DirectDataComparison) {
    const int n_points = 1000;
    const int n_dims = 2;
    const int k = 5;
    
    // Use same data for both
    auto data_torch = create_clustered_data_torch(n_points, k, n_dims, 10.0f);
    auto data_tensor = torch_to_tensor(data_torch);
    
    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 30, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 30, 1e-4f);
    
    float inertia_old = compute_inertia_torch(data_torch, centroids_old, labels_old);
    float inertia_new = compute_inertia_tensor(data_tensor, centroids_new, labels_new);
    
    // Inertias should be similar (within 20%)
    float rel_diff = std::abs(inertia_old - inertia_new) / std::max(inertia_old, inertia_new);
    EXPECT_LT(rel_diff, 0.2f) 
        << "Inertia mismatch: old=" << inertia_old << " new=" << inertia_new 
        << " rel_diff=" << rel_diff;
}
