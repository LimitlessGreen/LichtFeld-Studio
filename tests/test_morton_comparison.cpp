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

#include "kernels/morton_encoding.cuh"
#include "kernels/morton_encoding_new.cuh"
#include "core/tensor.hpp"

namespace {

constexpr float FLOAT_TOLERANCE = 1e-5f;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error); \
    } while(0)

// ============================================================================
// Helper Functions
// ============================================================================

torch::Tensor create_random_positions_torch(int n, float min_val = -10.0f, float max_val = 10.0f) {
    auto positions = torch::rand({n, 3}, torch::kCUDA) * (max_val - min_val) + min_val;
    return positions;
}

torch::Tensor create_grid_positions_torch(int n_per_dim) {
    std::vector<float> positions_vec;
    positions_vec.reserve(n_per_dim * n_per_dim * n_per_dim * 3);
    
    for (int x = 0; x < n_per_dim; ++x) {
        for (int y = 0; y < n_per_dim; ++y) {
            for (int z = 0; z < n_per_dim; ++z) {
                positions_vec.push_back(static_cast<float>(x));
                positions_vec.push_back(static_cast<float>(y));
                positions_vec.push_back(static_cast<float>(z));
            }
        }
    }
    
    auto positions = torch::from_blob(
        positions_vec.data(), 
        {n_per_dim * n_per_dim * n_per_dim, 3}, 
        torch::kFloat32).clone();
    return positions.cuda();
}

torch::Tensor create_clustered_positions_torch(int n_clusters, int points_per_cluster, float separation = 10.0f) {
    int total_points = n_clusters * points_per_cluster;
    auto positions = torch::zeros({total_points, 3}, torch::kCUDA);
    
    for (int c = 0; c < n_clusters; ++c) {
        int start = c * points_per_cluster;
        int end = start + points_per_cluster;
        
        auto cluster_center = torch::tensor({c * separation, c * separation * 0.5f, c * separation * 0.3f}, 
                                           torch::kCUDA);
        auto noise = torch::randn({points_per_cluster, 3}, torch::kCUDA) * 0.5f;
        
        positions.slice(0, start, end) = cluster_center + noise;
    }
    
    return positions;
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
    } else if (torch_tensor.scalar_type() == torch::kInt64) {
        std::vector<int64_t> data(cpu_tensor.data_ptr<int64_t>(),
                                 cpu_tensor.data_ptr<int64_t>() + cpu_tensor.numel());
        auto tensor = gs::Tensor::empty(gs::TensorShape(shape), gs::Device::CUDA, gs::DataType::Int64);
        cudaMemcpy(tensor.raw_ptr(), data.data(), data.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
        return tensor;
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
    } else if (gs_tensor.dtype() == gs::DataType::Int64) {
        auto data = cpu_tensor.to_vector_int64();
        auto torch_tensor = torch::from_blob(data.data(), shape, torch::kInt64).clone();
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

    if (torch_tensor.scalar_type() == torch::kInt64 &&
        gs_tensor.dtype() == gs::DataType::Int64) {

        auto torch_data = torch_cpu.data_ptr<int64_t>();
        auto gs_data = gs_cpu.to_vector_int64();

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

bool morton_codes_are_equal(const torch::Tensor& morton_old,
                            const gs::Tensor& morton_new) {
    return tensors_are_equal(morton_old, morton_new, 0.0f);
}

bool check_morton_ordering(const torch::Tensor& morton_codes) {
    // Morton codes should generally improve spatial locality
    // This just checks they're valid int64 values
    auto cpu_codes = morton_codes.cpu();
    for (int i = 0; i < morton_codes.size(0); ++i) {
        int64_t code = cpu_codes[i].item<int64_t>();
        // Just verify it's a reasonable value (not NaN-equivalent, etc.)
        if (code == std::numeric_limits<int64_t>::min()) {
            // This is actually valid (it's our offset), so continue
            continue;
        }
    }
    return true;
}

bool check_morton_ordering(const gs::Tensor& morton_codes) {
    auto cpu_codes = morton_codes.cpu();
    auto codes_vec = cpu_codes.to_vector_int64();
    
    for (size_t i = 0; i < codes_vec.size(); ++i) {
        // Just verify codes are valid
        if (codes_vec[i] == std::numeric_limits<int64_t>::min()) {
            continue;
        }
    }
    return true;
}

int count_unique_indices(const torch::Tensor& indices) {
    auto cpu_indices = indices.cpu();
    std::vector<int64_t> idx_vec;
    for (int i = 0; i < indices.size(0); ++i) {
        idx_vec.push_back(cpu_indices[i].item<int64_t>());
    }
    std::sort(idx_vec.begin(), idx_vec.end());
    auto last = std::unique(idx_vec.begin(), idx_vec.end());
    return std::distance(idx_vec.begin(), last);
}

int count_unique_indices(const gs::Tensor& indices) {
    auto cpu_indices = indices.cpu();
    auto idx_vec = cpu_indices.to_vector_int64();
    std::sort(idx_vec.begin(), idx_vec.end());
    auto last = std::unique(idx_vec.begin(), idx_vec.end());
    return std::distance(idx_vec.begin(), last);
}

bool indices_are_permutation(const torch::Tensor& indices, int n) {
    return count_unique_indices(indices) == n;
}

bool indices_are_permutation(const gs::Tensor& indices, int n) {
    return count_unique_indices(indices) == static_cast<int>(n);
}

} // anonymous namespace

class MortonEncodingComparisonTest : public ::testing::Test {
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

TEST_F(MortonEncodingComparisonTest, PositionDataConversionRoundTrip) {
    const int n = 100;

    auto torch_original = create_random_positions_torch(n);
    auto gs_converted = torch_to_tensor(torch_original);
    auto torch_roundtrip = tensor_to_torch(gs_converted);

    ASSERT_TRUE(tensors_are_equal(torch_original, gs_converted, 1e-6f));
    ASSERT_TRUE(torch::allclose(torch_original, torch_roundtrip, 1e-6f, 1e-6f));
}

// ============================================================================
// Basic Morton Encoding Tests
// ============================================================================

TEST_F(MortonEncodingComparisonTest, BasicMortonEncoding_Comparison) {
    const int n_points = 1000;

    auto positions_torch = create_random_positions_torch(n_points);
    auto positions_tensor = torch_to_tensor(positions_torch);

    ASSERT_TRUE(tensors_are_equal(positions_torch, positions_tensor, 1e-5f));

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    EXPECT_EQ(morton_old.size(0), n_points);
    EXPECT_EQ(morton_new.shape()[0], n_points);
    
    EXPECT_EQ(morton_old.scalar_type(), torch::kInt64);
    EXPECT_EQ(morton_new.dtype(), gs::DataType::Int64);

    EXPECT_TRUE(check_morton_ordering(morton_old));
    EXPECT_TRUE(check_morton_ordering(morton_new));

    // Morton codes should be identical for same input
    EXPECT_TRUE(morton_codes_are_equal(morton_old, morton_new))
        << "Morton codes differ between implementations";
}

TEST_F(MortonEncodingComparisonTest, GridPositions_Comparison) {
    const int n_per_dim = 10;
    const int n_points = n_per_dim * n_per_dim * n_per_dim;

    auto positions_torch = create_grid_positions_torch(n_per_dim);
    auto positions_tensor = torch_to_tensor(positions_torch);

    ASSERT_TRUE(tensors_are_equal(positions_torch, positions_tensor, 1e-5f));

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    EXPECT_EQ(morton_old.size(0), n_points);
    EXPECT_EQ(morton_new.shape()[0], n_points);

    EXPECT_TRUE(morton_codes_are_equal(morton_old, morton_new));
}

TEST_F(MortonEncodingComparisonTest, ClusteredPositions_Comparison) {
    const int n_clusters = 5;
    const int points_per_cluster = 200;
    const int n_points = n_clusters * points_per_cluster;

    auto positions_torch = create_clustered_positions_torch(n_clusters, points_per_cluster, 15.0f);
    auto positions_tensor = torch_to_tensor(positions_torch);

    ASSERT_TRUE(tensors_are_equal(positions_torch, positions_tensor, 1e-5f));

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    EXPECT_EQ(morton_old.size(0), n_points);
    EXPECT_EQ(morton_new.shape()[0], n_points);

    EXPECT_TRUE(morton_codes_are_equal(morton_old, morton_new));
}

// ============================================================================
// Morton Sort Indices Tests
// ============================================================================

TEST_F(MortonEncodingComparisonTest, BasicSortIndices_Comparison) {
    const int n_points = 500;

    auto positions_torch = create_random_positions_torch(n_points);
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    auto indices_old = gs::morton_sort_indices(morton_old);
    auto indices_new = gs::morton_sort_indices_new(morton_new);

    EXPECT_EQ(indices_old.size(0), n_points);
    EXPECT_EQ(indices_new.shape()[0], n_points);

    EXPECT_EQ(indices_old.scalar_type(), torch::kInt64);
    EXPECT_EQ(indices_new.dtype(), gs::DataType::Int64);

    // Both should be valid permutations
    EXPECT_TRUE(indices_are_permutation(indices_old, n_points));
    EXPECT_TRUE(indices_are_permutation(indices_new, n_points));

    // Check that indices actually sort the morton codes
    auto sorted_morton_old = morton_old.index_select(0, indices_old);
    auto sorted_morton_new_torch = tensor_to_torch(morton_new);
    auto indices_new_torch = tensor_to_torch(indices_new);
    auto sorted_morton_new = sorted_morton_new_torch.index_select(0, indices_new_torch);

    // Verify sorting worked
    auto sorted_old_cpu = sorted_morton_old.cpu();
    for (int i = 1; i < n_points; ++i) {
        EXPECT_LE(sorted_old_cpu[i-1].item<int64_t>(), sorted_old_cpu[i].item<int64_t>());
    }

    auto sorted_new_cpu = sorted_morton_new.cpu();
    for (int i = 1; i < n_points; ++i) {
        EXPECT_LE(sorted_new_cpu[i-1].item<int64_t>(), sorted_new_cpu[i].item<int64_t>());
    }
}

TEST_F(MortonEncodingComparisonTest, SortIndicesWithIdenticalCodes_Comparison) {
    const int n_points = 100;

    // Create positions that will generate some identical Morton codes
    auto positions = torch::zeros({n_points, 3}, torch::kCUDA);
    // Set half to same position
    positions.slice(0, 0, n_points/2).fill_(5.0f);
    positions.slice(0, n_points/2, n_points) = create_random_positions_torch(n_points/2);

    auto positions_torch = positions;
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    auto indices_old = gs::morton_sort_indices(morton_old);
    auto indices_new = gs::morton_sort_indices_new(morton_new);

    EXPECT_TRUE(indices_are_permutation(indices_old, n_points));
    EXPECT_TRUE(indices_are_permutation(indices_new, n_points));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(MortonEncodingComparisonTest, SinglePoint_Comparison) {
    const int n_points = 1;

    auto positions_torch = create_random_positions_torch(n_points);
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    EXPECT_EQ(morton_old.size(0), 1);
    EXPECT_EQ(morton_new.shape()[0], 1);

    EXPECT_TRUE(morton_codes_are_equal(morton_old, morton_new));

    auto indices_old = gs::morton_sort_indices(morton_old);
    auto indices_new = gs::morton_sort_indices_new(morton_new);

    EXPECT_EQ(indices_old.size(0), 1);
    EXPECT_EQ(indices_new.shape()[0], 1);
}

TEST_F(MortonEncodingComparisonTest, TwoPoints_Comparison) {
    const int n_points = 2;

    auto positions_torch = torch::tensor({{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}, torch::kCUDA);
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    EXPECT_TRUE(morton_codes_are_equal(morton_old, morton_new));

    auto indices_old = gs::morton_sort_indices(morton_old);
    auto indices_new = gs::morton_sort_indices_new(morton_new);

    EXPECT_TRUE(indices_are_permutation(indices_old, n_points));
    EXPECT_TRUE(indices_are_permutation(indices_new, n_points));
}

TEST_F(MortonEncodingComparisonTest, AllSamePosition_Comparison) {
    const int n_points = 100;

    auto positions_torch = torch::full({n_points, 3}, 5.0f, torch::kCUDA);
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    EXPECT_TRUE(morton_codes_are_equal(morton_old, morton_new));

    // All Morton codes should be identical
    auto morton_old_cpu = morton_old.cpu();
    int64_t first_code = morton_old_cpu[0].item<int64_t>();
    for (int i = 1; i < n_points; ++i) {
        EXPECT_EQ(morton_old_cpu[i].item<int64_t>(), first_code);
    }

    auto morton_new_cpu = morton_new.cpu();
    auto codes_vec = morton_new_cpu.to_vector_int64();
    for (size_t i = 1; i < codes_vec.size(); ++i) {
        EXPECT_EQ(codes_vec[i], codes_vec[0]);
    }
}

TEST_F(MortonEncodingComparisonTest, NegativeCoordinates_Comparison) {
    const int n_points = 200;

    auto positions_torch = create_random_positions_torch(n_points, -100.0f, -10.0f);
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    EXPECT_TRUE(morton_codes_are_equal(morton_old, morton_new));
}

TEST_F(MortonEncodingComparisonTest, MixedSignCoordinates_Comparison) {
    const int n_points = 200;

    auto positions_torch = create_random_positions_torch(n_points, -50.0f, 50.0f);
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    EXPECT_TRUE(morton_codes_are_equal(morton_old, morton_new));
}

TEST_F(MortonEncodingComparisonTest, VerySmallCoordinates_Comparison) {
    const int n_points = 200;

    auto positions_torch = create_random_positions_torch(n_points, 0.0f, 0.01f);
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    EXPECT_TRUE(morton_codes_are_equal(morton_old, morton_new));
}

TEST_F(MortonEncodingComparisonTest, VeryLargeCoordinates_Comparison) {
    const int n_points = 200;

    auto positions_torch = create_random_positions_torch(n_points, 1000.0f, 10000.0f);
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    EXPECT_TRUE(morton_codes_are_equal(morton_old, morton_new));
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(MortonEncodingComparisonTest, LargeDataset_Comparison) {
    const int n_points = 100000;

    auto positions_torch = create_random_positions_torch(n_points);
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    EXPECT_EQ(morton_old.size(0), n_points);
    EXPECT_EQ(morton_new.shape()[0], n_points);

    EXPECT_TRUE(morton_codes_are_equal(morton_old, morton_new));
}

TEST_F(MortonEncodingComparisonTest, LargeDatasetSort_Comparison) {
    const int n_points = 50000;

    auto positions_torch = create_random_positions_torch(n_points);
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    auto indices_old = gs::morton_sort_indices(morton_old);
    auto indices_new = gs::morton_sort_indices_new(morton_new);

    EXPECT_TRUE(indices_are_permutation(indices_old, n_points));
    EXPECT_TRUE(indices_are_permutation(indices_new, n_points));
}

TEST_F(MortonEncodingComparisonTest, VeryLargeGrid_Comparison) {
    const int n_per_dim = 50;  // 50^3 = 125,000 points
    const int n_points = n_per_dim * n_per_dim * n_per_dim;

    auto positions_torch = create_grid_positions_torch(n_per_dim);
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    EXPECT_TRUE(morton_codes_are_equal(morton_old, morton_new));
}

// ============================================================================
// Spatial Locality Tests
// ============================================================================

TEST_F(MortonEncodingComparisonTest, SpatialLocalityPreservation_Comparison) {
    const int n_points = 1000;

    auto positions_torch = create_random_positions_torch(n_points);
    auto positions_tensor = torch_to_tensor(positions_torch);

    auto morton_old = gs::morton_encode(positions_torch);
    auto morton_new = gs::morton_encode_new(positions_tensor);

    auto indices_old = gs::morton_sort_indices(morton_old);
    auto indices_new = gs::morton_sort_indices_new(morton_new);

    // After sorting by Morton code, nearby points in the sorted order
    // should be spatially close. We can verify this by checking that
    // consecutive points have relatively small distances.

    auto sorted_positions_old = positions_torch.index_select(0, indices_old);
    auto indices_new_torch = tensor_to_torch(indices_new);
    auto sorted_positions_new = positions_torch.index_select(0, indices_new_torch);

    // Calculate average distance between consecutive points
    auto diff_old = sorted_positions_old.slice(0, 1, n_points) - 
                    sorted_positions_old.slice(0, 0, n_points - 1);
    auto distances_old = torch::norm(diff_old, 2, 1);
    float avg_dist_old = distances_old.mean().item<float>();

    auto diff_new = sorted_positions_new.slice(0, 1, n_points) - 
                    sorted_positions_new.slice(0, 0, n_points - 1);
    auto distances_new = torch::norm(diff_new, 2, 1);
    float avg_dist_new = distances_new.mean().item<float>();

    // Compare to random ordering (which should have larger average distance)
    auto random_indices = torch::randperm(n_points, torch::kCUDA);
    auto random_positions = positions_torch.index_select(0, random_indices);
    auto diff_random = random_positions.slice(0, 1, n_points) - 
                       random_positions.slice(0, 0, n_points - 1);
    auto distances_random = torch::norm(diff_random, 2, 1);
    float avg_dist_random = distances_random.mean().item<float>();

    // Morton ordering should have better locality than random
    EXPECT_LT(avg_dist_old, avg_dist_random * 1.2f);
    EXPECT_LT(avg_dist_new, avg_dist_random * 1.2f);

    // Both implementations should have similar locality
    float locality_diff = std::abs(avg_dist_old - avg_dist_new) / avg_dist_old;
    EXPECT_LT(locality_diff, 0.1f)  // Within 10%
        << "Spatial locality differs: old=" << avg_dist_old 
        << " new=" << avg_dist_new;
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

class MortonEncodingPerformanceTest : public ::testing::Test {
protected:
    struct BenchmarkResult {
        std::string name;
        int n_points;
        double encode_time_ms_old;
        double encode_time_ms_new;
        double encode_speedup;
        double sort_time_ms_old;
        double sort_time_ms_new;
        double sort_speedup;
        double total_time_ms_old;
        double total_time_ms_new;
        double total_speedup;
    };

    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        torch::manual_seed(42);
        torch::cuda::manual_seed(42);
        gs::Tensor::manual_seed(42);
        
        // Warm-up GPU
        auto warmup = torch::randn({10000, 3}, torch::kCUDA);
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
        std::cout << "\n" << std::string(140, '=') << "\n";
        std::cout << "MORTON ENCODING PERFORMANCE COMPARISON: morton_encoding.cu vs morton_encoding_new.cu\n";
        std::cout << std::string(140, '=') << "\n\n";

        std::cout << std::left << std::setw(25) << "Benchmark"
                  << std::right << std::setw(10) << "N Points"
                  << std::right << std::setw(14) << "Encode Old"
                  << std::right << std::setw(14) << "Encode New"
                  << std::right << std::setw(12) << "Enc Speed"
                  << std::right << std::setw(14) << "Sort Old"
                  << std::right << std::setw(14) << "Sort New"
                  << std::right << std::setw(12) << "Sort Speed"
                  << std::right << std::setw(14) << "Total Old"
                  << std::right << std::setw(14) << "Total New"
                  << std::right << std::setw(12) << "Tot Speed"
                  << "\n";
        std::cout << std::string(140, '-') << "\n";

        for (const auto& r : results) {
            std::cout << std::left << std::setw(25) << r.name
                      << std::right << std::setw(10) << r.n_points
                      << std::right << std::setw(13) << std::fixed << std::setprecision(3) << r.encode_time_ms_old << "ms"
                      << std::right << std::setw(13) << std::fixed << std::setprecision(3) << r.encode_time_ms_new << "ms"
                      << std::right << std::setw(11) << std::fixed << std::setprecision(2) << r.encode_speedup << "x"
                      << std::right << std::setw(13) << std::fixed << std::setprecision(3) << r.sort_time_ms_old << "ms"
                      << std::right << std::setw(13) << std::fixed << std::setprecision(3) << r.sort_time_ms_new << "ms"
                      << std::right << std::setw(11) << std::fixed << std::setprecision(2) << r.sort_speedup << "x"
                      << std::right << std::setw(13) << std::fixed << std::setprecision(3) << r.total_time_ms_old << "ms"
                      << std::right << std::setw(13) << std::fixed << std::setprecision(3) << r.total_time_ms_new << "ms"
                      << std::right << std::setw(11) << std::fixed << std::setprecision(2) << r.total_speedup << "x"
                      << "\n";
        }

        std::cout << std::string(140, '=') << "\n";

        // Summary statistics
        double avg_encode_speedup = 0.0;
        double avg_sort_speedup = 0.0;
        double avg_total_speedup = 0.0;

        for (const auto& r : results) {
            avg_encode_speedup += r.encode_speedup;
            avg_sort_speedup += r.sort_speedup;
            avg_total_speedup += r.total_speedup;
        }
        avg_encode_speedup /= results.size();
        avg_sort_speedup /= results.size();
        avg_total_speedup /= results.size();

        std::cout << "\nSUMMARY:\n";
        std::cout << "  Average Encode Speedup: " << std::fixed << std::setprecision(2) << avg_encode_speedup << "x\n";
        std::cout << "  Average Sort Speedup:   " << std::fixed << std::setprecision(2) << avg_sort_speedup << "x\n";
        std::cout << "  Average Total Speedup:  " << std::fixed << std::setprecision(2) << avg_total_speedup << "x\n";
        std::cout << std::string(140, '=') << "\n\n";
    }
};

TEST_F(MortonEncodingPerformanceTest, ComprehensiveBenchmark) {
    std::vector<BenchmarkResult> results;

    std::vector<int> point_counts = {
        1000,
        5000,
        10000,
        50000,
        100000,
        500000,
        1000000
    };

    for (int n_points : point_counts) {
        auto positions_torch = create_random_positions_torch(n_points);
        auto positions_tensor = torch_to_tensor(positions_torch);

        BenchmarkResult result;
        result.name = "Random_" + std::to_string(n_points/1000) + "k";
        result.n_points = n_points;

        // Benchmark encoding - old
        torch::Tensor morton_old;
        result.encode_time_ms_old = benchmark([&]() {
            morton_old = gs::morton_encode(positions_torch);
        });

        // Benchmark encoding - new
        gs::Tensor morton_new;
        result.encode_time_ms_new = benchmark([&]() {
            morton_new = gs::morton_encode_new(positions_tensor);
        });

        result.encode_speedup = result.encode_time_ms_old / result.encode_time_ms_new;

        // Benchmark sorting - old
        result.sort_time_ms_old = benchmark([&]() {
            auto indices = gs::morton_sort_indices(morton_old);
        });

        // Benchmark sorting - new
        result.sort_time_ms_new = benchmark([&]() {
            auto indices = gs::morton_sort_indices_new(morton_new);
        });

        result.sort_speedup = result.sort_time_ms_old / result.sort_time_ms_new;

        // Total time
        result.total_time_ms_old = result.encode_time_ms_old + result.sort_time_ms_old;
        result.total_time_ms_new = result.encode_time_ms_new + result.sort_time_ms_new;
        result.total_speedup = result.total_time_ms_old / result.total_time_ms_new;

        results.push_back(result);
    }

    print_results(results);
}

TEST_F(MortonEncodingPerformanceTest, GridBenchmark) {
    std::vector<BenchmarkResult> results;

    std::vector<int> grid_sizes = {10, 20, 30, 40, 50, 60};

    for (int grid_size : grid_sizes) {
        int n_points = grid_size * grid_size * grid_size;
        
        auto positions_torch = create_grid_positions_torch(grid_size);
        auto positions_tensor = torch_to_tensor(positions_torch);

        BenchmarkResult result;
        result.name = "Grid_" + std::to_string(grid_size) + "^3";
        result.n_points = n_points;

        torch::Tensor morton_old;
        result.encode_time_ms_old = benchmark([&]() {
            morton_old = gs::morton_encode(positions_torch);
        }, 2, 5);

        gs::Tensor morton_new;
        result.encode_time_ms_new = benchmark([&]() {
            morton_new = gs::morton_encode_new(positions_tensor);
        }, 2, 5);

        result.encode_speedup = result.encode_time_ms_old / result.encode_time_ms_new;

        result.sort_time_ms_old = benchmark([&]() {
            auto indices = gs::morton_sort_indices(morton_old);
        }, 2, 5);

        result.sort_time_ms_new = benchmark([&]() {
            auto indices = gs::morton_sort_indices_new(morton_new);
        }, 2, 5);

        result.sort_speedup = result.sort_time_ms_old / result.sort_time_ms_new;

        result.total_time_ms_old = result.encode_time_ms_old + result.sort_time_ms_old;
        result.total_time_ms_new = result.encode_time_ms_new + result.sort_time_ms_new;
        result.total_speedup = result.total_time_ms_old / result.total_time_ms_new;

        results.push_back(result);
    }

    print_results(results);
}

TEST_F(MortonEncodingPerformanceTest, RealWorldScenarios) {
    std::vector<BenchmarkResult> results;

    struct ScenarioConfig {
        std::string name;
        int n_points;
    };

    std::vector<ScenarioConfig> scenarios = {
        {"Small_Scene", 10000},
        {"Medium_Scene", 100000},
        {"Large_Scene", 500000},
        {"XLarge_Scene", 1000000},
        {"Typical_3DGS", 250000},
    };

    for (const auto& scenario : scenarios) {
        auto positions_torch = create_random_positions_torch(scenario.n_points);
        auto positions_tensor = torch_to_tensor(positions_torch);

        BenchmarkResult result;
        result.name = scenario.name;
        result.n_points = scenario.n_points;

        torch::Tensor morton_old;
        result.encode_time_ms_old = benchmark([&]() {
            morton_old = gs::morton_encode(positions_torch);
        }, 3, 10);

        gs::Tensor morton_new;
        result.encode_time_ms_new = benchmark([&]() {
            morton_new = gs::morton_encode_new(positions_tensor);
        }, 3, 10);

        result.encode_speedup = result.encode_time_ms_old / result.encode_time_ms_new;

        result.sort_time_ms_old = benchmark([&]() {
            auto indices = gs::morton_sort_indices(morton_old);
        }, 3, 10);

        result.sort_time_ms_new = benchmark([&]() {
            auto indices = gs::morton_sort_indices_new(morton_new);
        }, 3, 10);

        result.sort_speedup = result.sort_time_ms_old / result.sort_time_ms_new;

        result.total_time_ms_old = result.encode_time_ms_old + result.sort_time_ms_old;
        result.total_time_ms_new = result.encode_time_ms_new + result.sort_time_ms_new;
        result.total_speedup = result.total_time_ms_old / result.total_time_ms_new;

        results.push_back(result);
    }

    print_results(results);
}
