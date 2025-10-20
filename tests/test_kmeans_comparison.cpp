/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>

#include "kernels/kmeans.cuh"
#include "kernels/kmeans_new.cuh"
#include "core/tensor.hpp"

namespace {

constexpr float FLOAT_TOLERANCE = 1e-3f;
constexpr float INERTIA_TOLERANCE = 0.15f;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error); \
    } while(0)

// ============================================================================
// Helper Functions
// ============================================================================

torch::Tensor create_random_data_torch(int n, int d, float min_val = 0.0f, float max_val = 1.0f) {
    auto data = torch::rand({n, d}, torch::kCUDA) * (max_val - min_val) + min_val;
    return data;
}

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

bool tensors_are_equal(const torch::Tensor& torch_tensor,
                       const gs::Tensor& gs_tensor,
                       float tolerance = 1e-6f) {
    if (torch_tensor.dim() != static_cast<int64_t>(gs_tensor.ndim())) {
        return false;
    }

    for (int i = 0; i < torch_tensor.dim(); ++i) {
        if (torch_tensor.size(i) != static_cast<int64_t>(gs_tensor.shape()[i])) {
            return false;
        }
    }

    auto torch_cpu = torch_tensor.cpu().contiguous();
    auto gs_cpu = gs_tensor.cpu();

    if (torch_tensor.scalar_type() == torch::kFloat32 &&
        gs_tensor.dtype() == gs::DataType::Float32) {

        auto torch_data = torch_cpu.data_ptr<float>();
        auto gs_data = gs_cpu.to_vector();

        if (torch_cpu.numel() != static_cast<int64_t>(gs_data.size())) {
            return false;
        }

        for (int64_t i = 0; i < torch_cpu.numel(); ++i) {
            float diff = std::abs(torch_data[i] - gs_data[i]);
            if (diff > tolerance) {
                return false;
            }
        }
        return true;
    }

    if (torch_tensor.scalar_type() == torch::kInt32 &&
        gs_tensor.dtype() == gs::DataType::Int32) {

        auto torch_data = torch_cpu.data_ptr<int>();
        auto gs_data = gs_cpu.to_vector_int();

        if (torch_cpu.numel() != static_cast<int64_t>(gs_data.size())) {
            return false;
        }

        for (int64_t i = 0; i < torch_cpu.numel(); ++i) {
            if (torch_data[i] != gs_data[i]) {
                return false;
            }
        }
        return true;
    }

    return false;
}

std::vector<int> find_cluster_permutation(
    const torch::Tensor& torch_centroids,
    const gs::Tensor& gs_centroids) {

    int k = torch_centroids.size(0);
    int d = torch_centroids.size(1);
    std::vector<int> perm(k);
    std::vector<bool> used(k, false);

    auto torch_cpu = torch_centroids.cpu();
    auto gs_cpu = gs_centroids.cpu();

    for (int i = 0; i < k; ++i) {
        float min_dist = std::numeric_limits<float>::max();
        int best_j = -1;

        for (int j = 0; j < k; ++j) {
            if (used[j]) continue;

            float dist = 0.0f;
            for (int dim = 0; dim < d; ++dim) {
                float torch_val = torch_cpu[i][dim].item<float>();
                float gs_val = static_cast<float>(gs_cpu[j][dim]);
                float diff = torch_val - gs_val;
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_j = j;
            }
        }

        perm[i] = best_j;
        used[best_j] = true;
    }

    return perm;
}

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
                             const gs::Tensor& labels,
                             const std::vector<int>& perm = {}) {
    auto n = data.shape()[0];
    auto d = data.shape()[1];

    float inertia = 0.0f;
    auto data_cpu = data.cpu();
    auto centroids_cpu = centroids.cpu();
    auto labels_cpu = labels.cpu();
    auto labels_vec = labels_cpu.to_vector_int();

    for (size_t i = 0; i < n; ++i) {
        int cluster = labels_vec[i];

        if (!perm.empty() && cluster >= 0 && cluster < static_cast<int>(perm.size())) {
            cluster = perm[cluster];
        }

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
    auto label_vec = labels_cpu.to_vector_int();

    std::sort(label_vec.begin(), label_vec.end());
    auto last = std::unique(label_vec.begin(), label_vec.end());
    return std::distance(label_vec.begin(), last);
}

} // anonymous namespace

class KMeansComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        torch::manual_seed(42);
        torch::cuda::manual_seed(42);
        gs::Tensor::manual_seed(42);
    }
};

// ============================================================================
// Conversion Tests
// ============================================================================

TEST_F(KMeansComparisonTest, DataConversionRoundTrip) {
    const int n = 100;
    const int d = 3;

    auto torch_original = torch::arange(0, n * d, torch::kCUDA).reshape({n, d}).to(torch::kFloat32);
    auto gs_converted = torch_to_tensor(torch_original);
    auto torch_roundtrip = tensor_to_torch(gs_converted);

    ASSERT_TRUE(tensors_are_equal(torch_original, gs_converted, 1e-6f));
    ASSERT_TRUE(torch::allclose(torch_original, torch_roundtrip, 1e-6f, 1e-6f));
}

// ============================================================================
// Basic Functionality Comparison Tests
// ============================================================================

TEST_F(KMeansComparisonTest, BasicClustering2D_Comparison) {
    const int n_points = 1000;
    const int n_dims = 2;
    const int k = 3;

    auto data_torch = create_clustered_data_torch(n_points, k, n_dims, 10.0f);
    auto data_tensor = torch_to_tensor(data_torch);

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 50, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 50, 1e-4f);

    EXPECT_EQ(centroids_old.size(0), k);
    EXPECT_EQ(centroids_new.shape()[0], k);
    EXPECT_EQ(labels_old.size(0), n_points);
    EXPECT_EQ(labels_new.shape()[0], n_points);

    EXPECT_TRUE(verify_labels_valid(labels_old, k));
    EXPECT_TRUE(verify_labels_valid(labels_new, k));

    auto perm = find_cluster_permutation(centroids_old, centroids_new);

    float inertia_old = compute_inertia_torch(data_torch, centroids_old, labels_old);
    float inertia_new = compute_inertia_tensor(data_tensor, centroids_new, labels_new);

    float rel_diff = std::abs(inertia_old - inertia_new) / std::max(inertia_old, inertia_new);

    EXPECT_LT(rel_diff, INERTIA_TOLERANCE)
        << "Inertia mismatch: old=" << inertia_old << " new=" << inertia_new
        << " rel_diff=" << (rel_diff * 100.0f) << "%";
}

TEST_F(KMeansComparisonTest, BasicClustering1D_Comparison) {
    const int n_points = 500;
    const int k = 256;

    auto data_torch = create_random_data_torch(n_points, 1, 0.0f, 100.0f);
    auto data_tensor = torch_to_tensor(data_torch);

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

    auto [centroids_old, labels_old] = gs::cuda::kmeans_1d(data_torch, k, 20);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_1d_new(data_tensor, k, 20);

    EXPECT_EQ(centroids_old.size(0), k);
    EXPECT_EQ(centroids_new.shape()[0], k);
    EXPECT_EQ(labels_old.size(0), n_points);
    EXPECT_EQ(labels_new.shape()[0], n_points);

    auto centroids_old_cpu = centroids_old.squeeze(1).cpu();
    auto centroids_new_cpu = centroids_new.squeeze(1).cpu();
    auto centroids_new_vec = centroids_new_cpu.to_vector();

    for (int i = 1; i < k; ++i) {
        EXPECT_GE(centroids_old_cpu[i].item<float>(), centroids_old_cpu[i-1].item<float>());
        EXPECT_GE(centroids_new_vec[i], centroids_new_vec[i-1]);
    }

    EXPECT_TRUE(verify_labels_valid(labels_old, k));
    EXPECT_TRUE(verify_labels_valid(labels_new, k));
}

TEST_F(KMeansComparisonTest, EmptyInput_Comparison) {
    const int k = 3;
    auto empty_data_torch = torch::empty({0, 2}, torch::kCUDA);
    auto empty_data_tensor = gs::Tensor::empty({0, 2}, gs::Device::CUDA, gs::DataType::Float32);

    auto [centroids_old, labels_old] = gs::cuda::kmeans(empty_data_torch, k, 10, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(empty_data_tensor, k, 10, 1e-4f);

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

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 10, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 10, 1e-4f);

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

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

    auto [centroids_old_5, labels_old_5] = gs::cuda::kmeans(data_torch, k, 5, 1e-4f);
    auto [centroids_old_50, labels_old_50] = gs::cuda::kmeans(data_torch, k, 50, 1e-4f);

    auto [centroids_new_5, labels_new_5] = gs::cuda::kmeans_new(data_tensor, k, 5, 1e-4f);
    auto [centroids_new_50, labels_new_50] = gs::cuda::kmeans_new(data_tensor, k, 50, 1e-4f);

    float inertia_old_5 = compute_inertia_torch(data_torch, centroids_old_5, labels_old_5);
    float inertia_old_50 = compute_inertia_torch(data_torch, centroids_old_50, labels_old_50);

    float inertia_new_5 = compute_inertia_tensor(data_tensor, centroids_new_5, labels_new_5, {});
    float inertia_new_50 = compute_inertia_tensor(data_tensor, centroids_new_50, labels_new_50, {});

    EXPECT_LE(inertia_old_50, inertia_old_5 + FLOAT_TOLERANCE);
    EXPECT_LE(inertia_new_50, inertia_new_5 + FLOAT_TOLERANCE);
}

// ============================================================================
// 1D Specific Tests
// ============================================================================

TEST_F(KMeansComparisonTest, OneDimensionalOrdering_Comparison) {
    const int n_points = 256;
    const int k = 16;

    auto data_torch = torch::linspace(0, 255, n_points, torch::kCUDA).unsqueeze(1);
    auto data_tensor = gs::Tensor::linspace(0, 255, n_points, gs::Device::CUDA).unsqueeze(1);

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

    auto [centroids_old, labels_old] = gs::cuda::kmeans_1d(data_torch, k, 20);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_1d_new(data_tensor, k, 20);

    auto centroids_old_vec = centroids_old.squeeze(1).cpu();
    auto centroids_new_squeezed = centroids_new.squeeze(1).cpu();
    auto centroids_new_vec = centroids_new_squeezed.to_vector();

    for (int i = 1; i < k; ++i) {
        float prev = centroids_old_vec[i-1].item<float>();
        float curr = centroids_old_vec[i].item<float>();
        EXPECT_GT(curr, prev);
    }

    for (int i = 1; i < k; ++i) {
        EXPECT_GT(centroids_new_vec[i], centroids_new_vec[i-1]);
    }
}

TEST_F(KMeansComparisonTest, OneDimensionalWithDuplicates_Comparison) {
    const int n_points = 100;
    const int k = 10;

    auto data_torch = torch::floor(torch::arange(0, n_points, torch::kCUDA) / 10.0f).unsqueeze(1);
    auto data_tensor_1d = gs::Tensor::arange(0, n_points).cuda() / 10.0f;
    auto data_tensor = data_tensor_1d.floor().unsqueeze(1);

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

    auto [centroids_old, labels_old] = gs::cuda::kmeans_1d(data_torch, k, 20);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_1d_new(data_tensor, k, 20);

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

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

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

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

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

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 10, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 10, 1e-4f);

    EXPECT_EQ(centroids_old.size(0), 1);
    EXPECT_EQ(centroids_new.shape()[0], 1);

    auto labels_old_cpu = labels_old.cpu();
    auto labels_new_cpu = labels_new.cpu();

    for (int i = 0; i < n_points; ++i) {
        EXPECT_EQ(labels_old_cpu[i].item<int>(), 0);
        EXPECT_EQ(static_cast<int>(static_cast<float>(labels_new_cpu[i])), 0);
    }
}

TEST_F(KMeansComparisonTest, VeryTightCluster_Comparison) {
    const int n_points = 100;
    const int n_dims = 2;
    const int k = 3;

    auto data_torch = torch::randn({n_points, n_dims}, torch::kCUDA) * 0.1f + 42.0f;
    auto data_tensor = torch_to_tensor(data_torch);

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

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

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

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

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

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

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

    auto [centroids_old, labels_old] = gs::cuda::kmeans_1d(data_torch, k, 20);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_1d_new(data_tensor, k, 20);

    auto centroids_old_cpu = centroids_old.squeeze(1).cpu();
    auto centroids_new_cpu = centroids_new.squeeze(1).cpu();
    auto centroids_new_vec = centroids_new_cpu.to_vector();

    for (int i = 1; i < k; ++i) {
        EXPECT_GE(centroids_old_cpu[i].item<float>(), centroids_old_cpu[i-1].item<float>());
        EXPECT_GE(centroids_new_vec[i], centroids_new_vec[i-1]);
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

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

    auto [centroids_old, labels_old] = gs::cuda::kmeans(data_torch, k, 50, 1e-4f);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_new(data_tensor, k, 50, 1e-4f);

    int unique_old = count_unique_labels(labels_old);
    int unique_new = count_unique_labels(labels_new);

    EXPECT_EQ(unique_old, k);
    EXPECT_EQ(unique_new, k);
}

TEST_F(KMeansComparisonTest, SOGTypical256Clusters_Comparison) {
    const int n_points = 10000;
    const int k = 256;

    auto data_torch = create_random_data_torch(n_points, 1, 0.0f, 1.0f);
    auto data_tensor = torch_to_tensor(data_torch);

    ASSERT_TRUE(tensors_are_equal(data_torch, data_tensor, 1e-5f));

    auto [centroids_old, labels_old] = gs::cuda::kmeans_1d(data_torch, k, 20);
    auto [centroids_new, labels_new] = gs::cuda::kmeans_1d_new(data_tensor, k, 20);

    EXPECT_EQ(centroids_old.size(0), 256);
    EXPECT_EQ(centroids_new.shape()[0], 256);
    EXPECT_TRUE(verify_labels_valid(labels_old, 256));
    EXPECT_TRUE(verify_labels_valid(labels_new, 256));

    auto centroids_old_cpu = centroids_old.squeeze(1).cpu();
    auto centroids_new_cpu = centroids_new.squeeze(1).cpu();
    auto centroids_new_vec = centroids_new_cpu.to_vector();

    for (int i = 1; i < 256; ++i) {
        EXPECT_GE(centroids_old_cpu[i].item<float>(), centroids_old_cpu[i-1].item<float>());
        EXPECT_GE(centroids_new_vec[i], centroids_new_vec[i-1]);
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

class KMeansPerformanceTest : public ::testing::Test {
protected:
    struct BenchmarkResult {
        std::string name;
        int n_points;
        int n_dims;
        int k;
        int iterations;
        double time_ms_old;
        double time_ms_new;
        double speedup;
        float inertia_old;
        float inertia_new;
    };

    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        torch::manual_seed(42);
        torch::cuda::manual_seed(42);
        gs::Tensor::manual_seed(42);
        
        // Warm-up GPU
        auto warmup = torch::randn({1000, 10}, torch::kCUDA);
        warmup.sum();
        cudaDeviceSynchronize();
    }

    template<typename Func>
    double benchmark(Func func, int warmup_runs = 3, int timing_runs = 10) {
        // Warm-up
        for (int i = 0; i < warmup_runs; ++i) {
            func();
            cudaDeviceSynchronize();
        }

        // Timing
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timing_runs; ++i) {
            func();
            cudaDeviceSynchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / (1000.0 * timing_runs);
    }

    void print_results(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n" << std::string(120, '=') << "\n";
        std::cout << "K-MEANS PERFORMANCE COMPARISON: kmeans.cu vs kmeans_new.cu\n";
        std::cout << std::string(120, '=') << "\n\n";

        std::cout << std::left << std::setw(30) << "Benchmark"
                  << std::right << std::setw(8) << "N"
                  << std::right << std::setw(6) << "D"
                  << std::right << std::setw(6) << "K"
                  << std::right << std::setw(7) << "Iters"
                  << std::right << std::setw(12) << "Old (ms)"
                  << std::right << std::setw(12) << "New (ms)"
                  << std::right << std::setw(12) << "Speedup"
                  << std::right << std::setw(14) << "Inertia Diff"
                  << "\n";
        std::cout << std::string(120, '-') << "\n";

        for (const auto& r : results) {
            float inertia_diff = std::abs(r.inertia_old - r.inertia_new) / 
                                std::max(r.inertia_old, r.inertia_new) * 100.0f;

            std::cout << std::left << std::setw(30) << r.name
                      << std::right << std::setw(8) << r.n_points
                      << std::right << std::setw(6) << r.n_dims
                      << std::right << std::setw(6) << r.k
                      << std::right << std::setw(7) << r.iterations
                      << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms_old
                      << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms_new
                      << std::right << std::setw(11) << std::fixed << std::setprecision(2) << r.speedup << "x"
                      << std::right << std::setw(13) << std::fixed << std::setprecision(2) << inertia_diff << "%"
                      << "\n";
        }

        std::cout << std::string(120, '=') << "\n";

        // Summary statistics
        double avg_speedup = 0.0;
        double max_speedup = 0.0;
        double min_speedup = std::numeric_limits<double>::max();

        for (const auto& r : results) {
            avg_speedup += r.speedup;
            max_speedup = std::max(max_speedup, r.speedup);
            min_speedup = std::min(min_speedup, r.speedup);
        }
        avg_speedup /= results.size();

        std::cout << "\nSUMMARY:\n";
        std::cout << "  Average Speedup: " << std::fixed << std::setprecision(2) << avg_speedup << "x\n";
        std::cout << "  Maximum Speedup: " << std::fixed << std::setprecision(2) << max_speedup << "x\n";
        std::cout << "  Minimum Speedup: " << std::fixed << std::setprecision(2) << min_speedup << "x\n";
        std::cout << std::string(120, '=') << "\n\n";
    }
};

TEST_F(KMeansPerformanceTest, DISABLED_Comprehensive2DBenchmark) {
    std::vector<BenchmarkResult> results;

    struct TestConfig {
        std::string name;
        int n_points;
        int k;
        int iterations;
    };

    std::vector<TestConfig> configs = {
        {"Small_FewClusters", 1000, 5, 30},
        {"Small_ManyClusters", 1000, 50, 30},
        {"Medium_FewClusters", 10000, 10, 30},
        {"Medium_ManyClusters", 10000, 100, 30},
        {"Large_FewClusters", 50000, 20, 20},
        {"Large_ManyClusters", 50000, 256, 20},
        {"XLarge_Clusters", 100000, 50, 10},
    };

    for (const auto& cfg : configs) {
        auto data_torch = create_random_data_torch(cfg.n_points, 2);
        auto data_tensor = torch_to_tensor(data_torch);

        BenchmarkResult result;
        result.name = cfg.name;
        result.n_points = cfg.n_points;
        result.n_dims = 2;
        result.k = cfg.k;
        result.iterations = cfg.iterations;

        // Benchmark old implementation
        torch::Tensor centroids_old, labels_old;
        result.time_ms_old = benchmark([&]() {
            std::tie(centroids_old, labels_old) = gs::cuda::kmeans(data_torch, cfg.k, cfg.iterations, 1e-4f);
        });
        result.inertia_old = compute_inertia_torch(data_torch, centroids_old, labels_old);

        // Benchmark new implementation
        gs::Tensor centroids_new, labels_new;
        result.time_ms_new = benchmark([&]() {
            std::tie(centroids_new, labels_new) = gs::cuda::kmeans_new(data_tensor, cfg.k, cfg.iterations, 1e-4f);
        });
        result.inertia_new = compute_inertia_tensor(data_tensor, centroids_new, labels_new);

        result.speedup = result.time_ms_old / result.time_ms_new;
        results.push_back(result);
    }

    print_results(results);
}

TEST_F(KMeansPerformanceTest, DISABLED_HighDimensionalBenchmark) {
    std::vector<BenchmarkResult> results;

    struct TestConfig {
        std::string name;
        int n_points;
        int n_dims;
        int k;
        int iterations;
    };

    std::vector<TestConfig> configs = {
        {"HighDim_10D", 5000, 10, 20, 20},
        {"HighDim_50D", 2000, 50, 10, 20},
        {"HighDim_100D", 1000, 100, 8, 15},
        {"HighDim_200D", 500, 200, 5, 15},
    };

    for (const auto& cfg : configs) {
        auto data_torch = create_random_data_torch(cfg.n_points, cfg.n_dims);
        auto data_tensor = torch_to_tensor(data_torch);

        BenchmarkResult result;
        result.name = cfg.name;
        result.n_points = cfg.n_points;
        result.n_dims = cfg.n_dims;
        result.k = cfg.k;
        result.iterations = cfg.iterations;

        torch::Tensor centroids_old, labels_old;
        result.time_ms_old = benchmark([&]() {
            std::tie(centroids_old, labels_old) = gs::cuda::kmeans(data_torch, cfg.k, cfg.iterations, 1e-4f);
        });
        result.inertia_old = compute_inertia_torch(data_torch, centroids_old, labels_old);

        gs::Tensor centroids_new, labels_new;
        result.time_ms_new = benchmark([&]() {
            std::tie(centroids_new, labels_new) = gs::cuda::kmeans_new(data_tensor, cfg.k, cfg.iterations, 1e-4f);
        });
        result.inertia_new = compute_inertia_tensor(data_tensor, centroids_new, labels_new);

        result.speedup = result.time_ms_old / result.time_ms_new;
        results.push_back(result);
    }

    print_results(results);
}

TEST_F(KMeansPerformanceTest, DISABLED_OneDimensionalBenchmark) {
    std::vector<BenchmarkResult> results;

    struct TestConfig {
        std::string name;
        int n_points;
        int k;
        int iterations;
    };

    std::vector<TestConfig> configs = {
        {"1D_Small", 1000, 32, 20},
        {"1D_Medium", 10000, 128, 20},
        {"1D_Large", 50000, 256, 20},
        {"1D_XLarge", 100000, 256, 15},
        {"1D_SOG_Typical", 50000, 256, 20},
    };

    for (const auto& cfg : configs) {
        auto data_torch = create_random_data_torch(cfg.n_points, 1, 0.0f, 1000.0f);
        auto data_tensor = torch_to_tensor(data_torch);

        BenchmarkResult result;
        result.name = cfg.name;
        result.n_points = cfg.n_points;
        result.n_dims = 1;
        result.k = cfg.k;
        result.iterations = cfg.iterations;

        torch::Tensor centroids_old, labels_old;
        result.time_ms_old = benchmark([&]() {
            std::tie(centroids_old, labels_old) = gs::cuda::kmeans_1d(data_torch, cfg.k, cfg.iterations);
        });
        result.inertia_old = compute_inertia_torch(data_torch, centroids_old, labels_old);

        gs::Tensor centroids_new, labels_new;
        result.time_ms_new = benchmark([&]() {
            std::tie(centroids_new, labels_new) = gs::cuda::kmeans_1d_new(data_tensor, cfg.k, cfg.iterations);
        });
        result.inertia_new = compute_inertia_tensor(data_tensor, centroids_new, labels_new);

        result.speedup = result.time_ms_old / result.time_ms_new;
        results.push_back(result);
    }

    print_results(results);
}

TEST_F(KMeansPerformanceTest, DISABLED_IterationScalingBenchmark) {
    std::vector<BenchmarkResult> results;

    const int n_points = 10000;
    const int n_dims = 3;
    const int k = 20;

    auto data_torch = create_random_data_torch(n_points, n_dims);
    auto data_tensor = torch_to_tensor(data_torch);

    std::vector<int> iteration_counts = {5, 10, 20, 50, 100};

    for (int iters : iteration_counts) {
        BenchmarkResult result;
        result.name = "Iterations_" + std::to_string(iters);
        result.n_points = n_points;
        result.n_dims = n_dims;
        result.k = k;
        result.iterations = iters;

        torch::Tensor centroids_old, labels_old;
        result.time_ms_old = benchmark([&]() {
            std::tie(centroids_old, labels_old) = gs::cuda::kmeans(data_torch, k, iters, 1e-4f);
        }, 2, 5);
        result.inertia_old = compute_inertia_torch(data_torch, centroids_old, labels_old);

        gs::Tensor centroids_new, labels_new;
        result.time_ms_new = benchmark([&]() {
            std::tie(centroids_new, labels_new) = gs::cuda::kmeans_new(data_tensor, k, iters, 1e-4f);
        }, 2, 5);
        result.inertia_new = compute_inertia_tensor(data_tensor, centroids_new, labels_new);

        result.speedup = result.time_ms_old / result.time_ms_new;
        results.push_back(result);
    }

    print_results(results);
}

TEST_F(KMeansPerformanceTest, DISABLED_ClusterCountScalingBenchmark) {
    std::vector<BenchmarkResult> results;

    const int n_points = 20000;
    const int n_dims = 2;
    const int iterations = 20;

    auto data_torch = create_random_data_torch(n_points, n_dims);
    auto data_tensor = torch_to_tensor(data_torch);

    std::vector<int> cluster_counts = {5, 10, 25, 50, 100, 200, 256};

    for (int k : cluster_counts) {
        BenchmarkResult result;
        result.name = "K_" + std::to_string(k);
        result.n_points = n_points;
        result.n_dims = n_dims;
        result.k = k;
        result.iterations = iterations;

        torch::Tensor centroids_old, labels_old;
        result.time_ms_old = benchmark([&]() {
            std::tie(centroids_old, labels_old) = gs::cuda::kmeans(data_torch, k, iterations, 1e-4f);
        }, 2, 5);
        result.inertia_old = compute_inertia_torch(data_torch, centroids_old, labels_old);

        gs::Tensor centroids_new, labels_new;
        result.time_ms_new = benchmark([&]() {
            std::tie(centroids_new, labels_new) = gs::cuda::kmeans_new(data_tensor, k, iterations, 1e-4f);
        }, 2, 5);
        result.inertia_new = compute_inertia_tensor(data_tensor, centroids_new, labels_new);

        result.speedup = result.time_ms_old / result.time_ms_new;
        results.push_back(result);
    }

    print_results(results);
}

TEST_F(KMeansPerformanceTest, DISABLED_RealWorldScenarios) {
    std::vector<BenchmarkResult> results;

    struct ScenarioConfig {
        std::string name;
        int n_points;
        int n_dims;
        int k;
        int iterations;
    };

    std::vector<ScenarioConfig> scenarios = {
        {"ImageSegmentation_Small", 10000, 3, 16, 30},
        {"ImageSegmentation_Medium", 50000, 3, 32, 25},
        {"ColorQuantization", 65536, 3, 256, 20},
        {"FeatureVectorClustering", 5000, 128, 50, 20},
        {"DocumentClustering", 10000, 300, 100, 15},
        {"SOG_Quantization", 100000, 1, 256, 20},
    };

    for (const auto& scenario : scenarios) {
        auto data_torch = create_random_data_torch(scenario.n_points, scenario.n_dims);
        auto data_tensor = torch_to_tensor(data_torch);

        BenchmarkResult result;
        result.name = scenario.name;
        result.n_points = scenario.n_points;
        result.n_dims = scenario.n_dims;
        result.k = scenario.k;
        result.iterations = scenario.iterations;

        torch::Tensor centroids_old, labels_old;
        result.time_ms_old = benchmark([&]() {
            if (scenario.n_dims == 1) {
                std::tie(centroids_old, labels_old) = gs::cuda::kmeans_1d(data_torch, scenario.k, scenario.iterations);
            } else {
                std::tie(centroids_old, labels_old) = gs::cuda::kmeans(data_torch, scenario.k, scenario.iterations, 1e-4f);
            }
        }, 2, 5);
        result.inertia_old = compute_inertia_torch(data_torch, centroids_old, labels_old);

        gs::Tensor centroids_new, labels_new;
        result.time_ms_new = benchmark([&]() {
            if (scenario.n_dims == 1) {
                std::tie(centroids_new, labels_new) = gs::cuda::kmeans_1d_new(data_tensor, scenario.k, scenario.iterations);
            } else {
                std::tie(centroids_new, labels_new) = gs::cuda::kmeans_new(data_tensor, scenario.k, scenario.iterations, 1e-4f);
            }
        }, 2, 5);
        result.inertia_new = compute_inertia_tensor(data_tensor, centroids_new, labels_new);

        result.speedup = result.time_ms_old / result.time_ms_new;
        results.push_back(result);
    }

    print_results(results);
}
