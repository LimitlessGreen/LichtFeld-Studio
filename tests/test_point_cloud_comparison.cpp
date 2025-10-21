/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <print>
#include <random>
#include <torch/torch.h>
#include <vector>

#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "core/point_cloud_new.hpp"
#include "core/tensor.hpp"

namespace {

    constexpr float FLOAT_TOLERANCE = 1e-5f;

    /**
     * @brief Compare two float arrays with tolerance
     */
    bool compare_arrays(const float* a, const float* b, size_t size, float tol = FLOAT_TOLERANCE) {
        for (size_t i = 0; i < size; ++i) {
            float diff = std::abs(a[i] - b[i]);
            float abs_max = std::max(std::abs(a[i]), std::abs(b[i]));
            float relative_diff = (abs_max > 1e-10f) ? (diff / abs_max) : diff;

            if (relative_diff > tol) {
                LOG_ERROR("Array mismatch at index {}: {} vs {} (diff={}, rel={})",
                          i, a[i], b[i], diff, relative_diff);
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Generate random point cloud data
     */
    void generate_random_data(std::vector<float>& positions,
                              std::vector<float>& colors,
                              size_t n_points,
                              unsigned int seed = 42) {
        positions.resize(n_points * 3);
        colors.resize(n_points * 3);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> pos_dist(-10.0f, 10.0f);
        std::uniform_real_distribution<float> col_dist(0.0f, 1.0f);

        for (size_t i = 0; i < n_points * 3; ++i) {
            positions[i] = pos_dist(rng);
            colors[i] = col_dist(rng);
        }
    }

    /**
     * @brief Create old PointCloud using torch tensors (the proper way)
     */
    gs::PointCloud create_old_pointcloud(const std::vector<float>& positions,
                                         const std::vector<float>& colors,
                                         size_t n_points) {
        // Create torch tensors from host data
        auto pos_tensor = torch::from_blob(
                              const_cast<float*>(positions.data()),
                              {static_cast<long>(n_points), 3},
                              torch::TensorOptions().dtype(torch::kFloat32))
                              .cuda()
                              .clone();

        auto col_tensor = torch::from_blob(
                              const_cast<float*>(colors.data()),
                              {static_cast<long>(n_points), 3},
                              torch::TensorOptions().dtype(torch::kFloat32))
                              .cuda()
                              .clone();

        // Allocate raw CUDA memory for PointCloud
        float* means_cuda = nullptr;
        float* colors_cuda = nullptr;

        cudaMalloc(&means_cuda, n_points * 3 * sizeof(float));
        cudaMalloc(&colors_cuda, n_points * 3 * sizeof(float));

        // Copy from torch tensors to raw CUDA
        cudaMemcpy(means_cuda, pos_tensor.data_ptr<float>(),
                   n_points * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(colors_cuda, col_tensor.data_ptr<float>(),
                   n_points * 3 * sizeof(float), cudaMemcpyDeviceToDevice);

        return gs::PointCloud(means_cuda, colors_cuda, n_points);
    }

    /**
     * @brief Copy data from old PointCloud to host
     */
    void copy_old_to_host(const gs::PointCloud& pc, std::vector<float>& means, std::vector<float>& colors) {
        means.resize(pc.num_points * 3);
        colors.resize(pc.num_points * 3);

        cudaMemcpy(means.data(), pc.means_cuda,
                   means.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(colors.data(), pc.colors_cuda,
                   colors.size() * sizeof(float), cudaMemcpyDeviceToHost);
    }

    /**
     * @brief Copy data from new PointCloudNew to host
     */
    void copy_new_to_host(const gs::PointCloudNew& pc, std::vector<float>& means, std::vector<float>& colors) {
        auto means_cpu = pc.means.cpu();
        auto colors_cpu = pc.colors.cpu();

        means = means_cpu.to_vector();
        colors = colors_cpu.to_vector();
    }

} // anonymous namespace

class PointCloudComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";

        // Clear any pending CUDA errors
        cudaGetLastError();

        gs::Tensor::manual_seed(42);
    }

    void TearDown() override {
        // Check for any CUDA errors at test end
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            LOG_WARN("CUDA error at test end: {}", cudaGetErrorString(error));
        }
    }
};

// ============= BASIC FUNCTIONALITY TESTS =============

TEST_F(PointCloudComparisonTest, BasicConstruction_Empty) {
    gs::PointCloud old_pc;
    gs::PointCloudNew new_pc;

    EXPECT_EQ(old_pc.num_points, 0);
    EXPECT_EQ(new_pc.size(), 0);
    EXPECT_FALSE(old_pc.is_gaussian());
    EXPECT_FALSE(new_pc.is_gaussian());
}

TEST_F(PointCloudComparisonTest, BasicConstruction_SmallCloud) {
    const size_t n_points = 100;
    std::vector<float> positions, colors;
    generate_random_data(positions, colors, n_points);

    // Create both implementations from same host data
    gs::PointCloud old_pc = create_old_pointcloud(positions, colors, n_points);
    gs::PointCloudNew new_pc(positions.data(), colors.data(), n_points);

    EXPECT_EQ(old_pc.num_points, n_points);
    EXPECT_EQ(new_pc.size(), n_points);

    // Retrieve and compare results
    std::vector<float> old_means, old_colors;
    std::vector<float> new_means, new_colors;

    copy_old_to_host(old_pc, old_means, old_colors);
    copy_new_to_host(new_pc, new_means, new_colors);

    // Both should match input
    EXPECT_TRUE(compare_arrays(positions.data(), old_means.data(), n_points * 3))
        << "Old PointCloud doesn't preserve input";
    EXPECT_TRUE(compare_arrays(positions.data(), new_means.data(), n_points * 3))
        << "New PointCloudNew doesn't preserve input";

    EXPECT_TRUE(compare_arrays(colors.data(), old_colors.data(), n_points * 3))
        << "Old colors mismatch";
    EXPECT_TRUE(compare_arrays(colors.data(), new_colors.data(), n_points * 3))
        << "New colors mismatch";

    // Both implementations should produce identical results
    EXPECT_TRUE(compare_arrays(old_means.data(), new_means.data(), n_points * 3))
        << "Old and New produce different position results";
    EXPECT_TRUE(compare_arrays(old_colors.data(), new_colors.data(), n_points * 3))
        << "Old and New produce different color results";
}

TEST_F(PointCloudComparisonTest, BasicConstruction_MediumCloud) {
    const size_t n_points = 5000;
    std::vector<float> positions, colors;
    generate_random_data(positions, colors, n_points, 12345);

    gs::PointCloud old_pc = create_old_pointcloud(positions, colors, n_points);
    gs::PointCloudNew new_pc(positions.data(), colors.data(), n_points);

    EXPECT_EQ(old_pc.num_points, n_points);
    EXPECT_EQ(new_pc.size(), n_points);

    std::vector<float> old_means, old_colors;
    std::vector<float> new_means, new_colors;

    copy_old_to_host(old_pc, old_means, old_colors);
    copy_new_to_host(new_pc, new_means, new_colors);

    EXPECT_TRUE(compare_arrays(old_means.data(), new_means.data(), n_points * 3));
    EXPECT_TRUE(compare_arrays(old_colors.data(), new_colors.data(), n_points * 3));
}

TEST_F(PointCloudComparisonTest, BasicConstruction_LargeCloud) {
    const size_t n_points = 50000;
    std::vector<float> positions, colors;
    generate_random_data(positions, colors, n_points);

    gs::PointCloud old_pc = create_old_pointcloud(positions, colors, n_points);
    gs::PointCloudNew new_pc(positions.data(), colors.data(), n_points);

    EXPECT_EQ(old_pc.num_points, n_points);
    EXPECT_EQ(new_pc.size(), n_points);

    std::vector<float> old_means, old_colors;
    std::vector<float> new_means, new_colors;

    copy_old_to_host(old_pc, old_means, old_colors);
    copy_new_to_host(new_pc, new_means, new_colors);

    EXPECT_TRUE(compare_arrays(old_means.data(), new_means.data(), n_points * 3));
    EXPECT_TRUE(compare_arrays(old_colors.data(), new_colors.data(), n_points * 3));
}

// ============= ALLOCATION TESTS =============

TEST_F(PointCloudComparisonTest, AllocateBasic_Small) {
    const size_t n_points = 100;

    gs::PointCloud old_pc;
    old_pc.allocate_basic(n_points);

    gs::PointCloudNew new_pc;
    new_pc.allocate_basic(n_points);

    EXPECT_EQ(old_pc.num_points, n_points);
    EXPECT_EQ(new_pc.size(), n_points);
    EXPECT_FALSE(old_pc.is_gaussian());
    EXPECT_FALSE(new_pc.is_gaussian());

    // Verify tensors are initialized to zero
    auto means_cpu = new_pc.means.cpu();
    auto colors_cpu = new_pc.colors.cpu();

    EXPECT_FLOAT_EQ(means_cpu.sum_scalar(), 0.0f);
    EXPECT_FLOAT_EQ(colors_cpu.sum_scalar(), 0.0f);
}

TEST_F(PointCloudComparisonTest, AllocateBasic_ZeroPoints) {
    gs::PointCloud old_pc;
    old_pc.allocate_basic(0);

    gs::PointCloudNew new_pc;
    new_pc.allocate_basic(0);

    EXPECT_EQ(old_pc.num_points, 0);
    EXPECT_EQ(new_pc.size(), 0);
}

TEST_F(PointCloudComparisonTest, AllocateBasic_Large) {
    const size_t n_points = 10000;

    gs::PointCloud old_pc;
    old_pc.allocate_basic(n_points);

    gs::PointCloudNew new_pc;
    new_pc.allocate_basic(n_points);

    EXPECT_EQ(old_pc.num_points, n_points);
    EXPECT_EQ(new_pc.size(), n_points);
}

TEST_F(PointCloudComparisonTest, AllocateGaussian_Degree0) {
    const size_t n_points = 1000;
    const size_t sh0_channels = 3;
    const size_t sh0_coeffs = 1;
    const size_t shN_channels = 3;
    const size_t shN_coeffs = 0;

    gs::PointCloud old_pc;
    old_pc.allocate_gaussian(n_points, sh0_channels, sh0_coeffs, shN_channels, shN_coeffs);

    gs::PointCloudNew new_pc;
    new_pc.allocate_gaussian(n_points, sh0_channels, sh0_coeffs, shN_channels, shN_coeffs);

    EXPECT_EQ(old_pc.num_points, n_points);
    EXPECT_EQ(new_pc.size(), n_points);
    EXPECT_TRUE(old_pc.is_gaussian());
    EXPECT_TRUE(new_pc.is_gaussian());

    // Verify shapes match - both use [N, channels, coeffs] format
    EXPECT_EQ(old_pc.sh0_dims[0], new_pc.sh0.size(0)); // N == N
    EXPECT_EQ(old_pc.sh0_dims[1], new_pc.sh0.size(1)); // channels == channels
    EXPECT_EQ(old_pc.sh0_dims[2], new_pc.sh0.size(2)); // coeffs == coeffs

    EXPECT_EQ(old_pc.shN_dims[0], new_pc.shN.size(0)); // N == N
    EXPECT_EQ(old_pc.shN_dims[1], new_pc.shN.size(1)); // channels == channels
    EXPECT_EQ(old_pc.shN_dims[2], new_pc.shN.size(2)); // coeffs == coeffs
}

TEST_F(PointCloudComparisonTest, AllocateGaussian_Degree1) {
    const size_t n_points = 2000;
    const size_t shN_coeffs = 3;

    gs::PointCloud old_pc;
    old_pc.allocate_gaussian(n_points, 3, 1, 3, shN_coeffs);

    gs::PointCloudNew new_pc;
    new_pc.allocate_gaussian(n_points, 3, 1, 3, shN_coeffs);

    EXPECT_TRUE(old_pc.is_gaussian());
    EXPECT_TRUE(new_pc.is_gaussian());

    EXPECT_EQ(old_pc.shN_dims[2], shN_coeffs);
    EXPECT_EQ(new_pc.shN.size(2), shN_coeffs);
}

TEST_F(PointCloudComparisonTest, AllocateGaussian_Degree2) {
    const size_t n_points = 3000;
    const size_t shN_coeffs = 8;

    gs::PointCloud old_pc;
    old_pc.allocate_gaussian(n_points, 3, 1, 3, shN_coeffs);

    gs::PointCloudNew new_pc;
    new_pc.allocate_gaussian(n_points, 3, 1, 3, shN_coeffs);

    EXPECT_EQ(old_pc.shN_dims[2], shN_coeffs);
    EXPECT_EQ(new_pc.shN.size(2), shN_coeffs);
}

TEST_F(PointCloudComparisonTest, AllocateGaussian_Degree3) {
    const size_t n_points = 4000;
    const size_t shN_coeffs = 15;

    gs::PointCloud old_pc;
    old_pc.allocate_gaussian(n_points, 3, 1, 3, shN_coeffs);

    gs::PointCloudNew new_pc;
    new_pc.allocate_gaussian(n_points, 3, 1, 3, shN_coeffs);

    EXPECT_EQ(old_pc.shN_dims[2], shN_coeffs);
    EXPECT_EQ(new_pc.shN.size(2), shN_coeffs);
}

// ============= DEVICE TRANSFER TESTS (Old PointCloud doesn't support these) =============

TEST_F(PointCloudComparisonTest, DeviceTransfer_CPUtoGPU) {
    const size_t n_points = 1000;
    std::vector<float> positions, colors;
    generate_random_data(positions, colors, n_points);

    // Create on CPU
    std::vector<size_t> shape = {n_points, 3};
    auto means_cpu = gs::Tensor::from_vector(positions, gs::TensorShape(shape), gs::Device::CPU);
    auto colors_cpu = gs::Tensor::from_vector(colors, gs::TensorShape(shape), gs::Device::CPU);

    gs::PointCloudNew pc_cpu(means_cpu, colors_cpu);

    // Transfer to GPU
    gs::PointCloudNew pc_gpu = pc_cpu.cuda();

    EXPECT_EQ(pc_cpu.size(), n_points);
    EXPECT_EQ(pc_gpu.size(), n_points);
    EXPECT_EQ(pc_cpu.means.device(), gs::Device::CPU);
    EXPECT_EQ(pc_gpu.means.device(), gs::Device::CUDA);

    // Verify data integrity
    auto gpu_means_cpu = pc_gpu.means.cpu();
    auto gpu_colors_cpu = pc_gpu.colors.cpu();

    EXPECT_TRUE(pc_cpu.means.all_close(gpu_means_cpu));
    EXPECT_TRUE(pc_cpu.colors.all_close(gpu_colors_cpu));
}

TEST_F(PointCloudComparisonTest, DeviceTransfer_GPUtoCPU) {
    const size_t n_points = 1000;
    std::vector<float> positions, colors;
    generate_random_data(positions, colors, n_points);

    gs::PointCloudNew pc_gpu(positions.data(), colors.data(), n_points);
    gs::PointCloudNew pc_cpu = pc_gpu.cpu();

    EXPECT_EQ(pc_cpu.means.device(), gs::Device::CPU);
    EXPECT_EQ(pc_gpu.means.device(), gs::Device::CUDA);

    // Verify data integrity
    auto gpu_cpu = pc_gpu.cpu();
    EXPECT_TRUE(pc_cpu.means.all_close(gpu_cpu.means));
    EXPECT_TRUE(pc_cpu.colors.all_close(gpu_cpu.colors));
}

TEST_F(PointCloudComparisonTest, DeviceTransfer_GaussianAttributes) {
    const size_t n_points = 500;

    gs::PointCloudNew pc_gpu;
    pc_gpu.allocate_gaussian(n_points, 3, 1, 3, 3);

    // Fill with random data
    pc_gpu.means.uniform_(-10.0f, 10.0f);
    pc_gpu.colors.uniform_(0.0f, 1.0f);
    pc_gpu.sh0.uniform_(-1.0f, 1.0f);
    pc_gpu.shN.uniform_(-0.5f, 0.5f);
    pc_gpu.opacity.uniform_(0.0f, 1.0f);
    pc_gpu.scaling.uniform_(-2.0f, 2.0f);
    pc_gpu.rotation.uniform_(-1.0f, 1.0f);

    // Transfer to CPU
    gs::PointCloudNew pc_cpu = pc_gpu.cpu();

    // Verify all attributes transferred correctly
    EXPECT_TRUE(pc_cpu.means.all_close(pc_gpu.means.cpu()));
    EXPECT_TRUE(pc_cpu.colors.all_close(pc_gpu.colors.cpu()));
    EXPECT_TRUE(pc_cpu.sh0.all_close(pc_gpu.sh0.cpu()));
    EXPECT_TRUE(pc_cpu.shN.all_close(pc_gpu.shN.cpu()));
    EXPECT_TRUE(pc_cpu.opacity.all_close(pc_gpu.opacity.cpu()));
    EXPECT_TRUE(pc_cpu.scaling.all_close(pc_gpu.scaling.cpu()));
    EXPECT_TRUE(pc_cpu.rotation.all_close(pc_gpu.rotation.cpu()));
}

// ============= COPY OPERATIONS =============

TEST_F(PointCloudComparisonTest, CopyToHost_Basic) {
    const size_t n_points = 100;
    std::vector<float> positions, colors;
    generate_random_data(positions, colors, n_points);

    // Old implementation
    gs::PointCloud old_pc = create_old_pointcloud(positions, colors, n_points);
    std::vector<float> old_means(n_points * 3);
    std::vector<float> old_colors(n_points * 3);
    old_pc.copy_to_host(old_means.data(), old_colors.data());

    // New implementation
    gs::PointCloudNew new_pc(positions.data(), colors.data(), n_points);
    std::vector<float> new_means(n_points * 3);
    std::vector<float> new_colors(n_points * 3);
    new_pc.copy_to_host(new_means.data(), new_colors.data());

    // Both should match input
    EXPECT_TRUE(compare_arrays(positions.data(), old_means.data(), n_points * 3));
    EXPECT_TRUE(compare_arrays(positions.data(), new_means.data(), n_points * 3));
    EXPECT_TRUE(compare_arrays(colors.data(), old_colors.data(), n_points * 3));
    EXPECT_TRUE(compare_arrays(colors.data(), new_colors.data(), n_points * 3));

    // Both implementations should produce same results
    EXPECT_TRUE(compare_arrays(old_means.data(), new_means.data(), n_points * 3));
    EXPECT_TRUE(compare_arrays(old_colors.data(), new_colors.data(), n_points * 3));
}

TEST_F(PointCloudComparisonTest, CopyToHost_VectorAPI) {
    const size_t n_points = 100;
    std::vector<float> positions, colors;
    generate_random_data(positions, colors, n_points);

    gs::PointCloudNew new_pc(positions.data(), colors.data(), n_points);

    auto retrieved_pos = new_pc.get_positions();
    auto retrieved_col = new_pc.get_colors();

    EXPECT_EQ(retrieved_pos.size(), n_points * 3);
    EXPECT_EQ(retrieved_col.size(), n_points * 3);
    EXPECT_TRUE(compare_arrays(positions.data(), retrieved_pos.data(), n_points * 3));
    EXPECT_TRUE(compare_arrays(colors.data(), retrieved_col.data(), n_points * 3));
}

// ============= VALIDATION TESTS =============

TEST_F(PointCloudComparisonTest, Validation_ValidBasic) {
    const size_t n_points = 100;

    gs::PointCloudNew pc;
    pc.allocate_basic(n_points);

    EXPECT_TRUE(pc.is_valid());
}

TEST_F(PointCloudComparisonTest, Validation_ValidGaussian) {
    const size_t n_points = 100;

    gs::PointCloudNew pc;
    pc.allocate_gaussian(n_points, 3, 1, 3, 3);

    EXPECT_TRUE(pc.is_valid());
}

TEST_F(PointCloudComparisonTest, Validation_InvalidEmpty) {
    gs::PointCloudNew pc;
    EXPECT_FALSE(pc.is_valid());
}

TEST_F(PointCloudComparisonTest, Validation_InvalidShapeMismatch) {
    gs::PointCloudNew pc;
    pc.means = gs::Tensor::zeros({100, 3}, gs::Device::CUDA);
    pc.colors = gs::Tensor::zeros({50, 3}, gs::Device::CUDA);

    EXPECT_FALSE(pc.is_valid());
}

// ============= CLONE TESTS =============

TEST_F(PointCloudComparisonTest, Clone_Basic) {
    const size_t n_points = 100;
    std::vector<float> positions, colors;
    generate_random_data(positions, colors, n_points);

    gs::PointCloudNew pc1(positions.data(), colors.data(), n_points);
    gs::PointCloudNew pc2 = pc1.clone();

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error after clone: " << cudaGetErrorString(err);

    EXPECT_EQ(pc1.size(), pc2.size());

    // Compare on CPU
    auto pc1_cpu = pc1.cpu();
    auto pc2_cpu = pc2.cpu();

    EXPECT_TRUE(pc1_cpu.means.all_close(pc2_cpu.means));
    EXPECT_TRUE(pc1_cpu.colors.all_close(pc2_cpu.colors));

    // Verify independence - modifying clone shouldn't affect original
    pc2.means.fill_(999.0f);
    cudaDeviceSynchronize();

    auto pc1_check = pc1.cpu();
    auto pc2_modified = pc2.cpu();

    EXPECT_FALSE(pc1_check.means.all_close(pc2_modified.means));
}

TEST_F(PointCloudComparisonTest, Clone_Gaussian) {
    const size_t n_points = 100;

    gs::PointCloudNew pc1;
    pc1.allocate_gaussian(n_points, 3, 1, 3, 3);
    pc1.means.uniform_(-10.0f, 10.0f);
    pc1.sh0.uniform_(-1.0f, 1.0f);

    gs::PointCloudNew pc2 = pc1.clone();

    EXPECT_TRUE(pc2.is_gaussian());
    EXPECT_TRUE(pc1.means.all_close(pc2.means));
    EXPECT_TRUE(pc1.sh0.all_close(pc2.sh0));

    // Verify independence
    pc2.sh0.mul_(2.0f);
    EXPECT_FALSE(pc1.sh0.all_close(pc2.sh0));
}

// ============= PERFORMANCE COMPARISON =============

TEST_F(PointCloudComparisonTest, Performance_ConstructionSpeed) {
    const size_t n_points = 10000;
    std::vector<float> positions, colors;
    generate_random_data(positions, colors, n_points);

    // Warm up
    gs::PointCloudNew warm_up(positions.data(), colors.data(), 100);
    cudaDeviceSynchronize();

    // Time old implementation (libtorch + raw CUDA)
    auto old_start = std::chrono::high_resolution_clock::now();
    gs::PointCloud old_pc = create_old_pointcloud(positions, colors, n_points);
    cudaDeviceSynchronize();
    auto old_end = std::chrono::high_resolution_clock::now();
    auto old_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            old_end - old_start)
                            .count();

    // Time new implementation (pure Tensor)
    auto new_start = std::chrono::high_resolution_clock::now();
    gs::PointCloudNew new_pc(positions.data(), colors.data(), n_points);
    cudaDeviceSynchronize();
    auto new_end = std::chrono::high_resolution_clock::now();
    auto new_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            new_end - new_start)
                            .count();

    double ratio = static_cast<double>(new_duration) / std::max(old_duration, 1L);
    std::println("Construction time - Old (torch+raw): {} μs, New (tensor): {} μs (ratio: {:.2f}x)",
                 old_duration, new_duration, ratio);

    // New implementation should be reasonably close - allow 10x slower
    EXPECT_LT(ratio, 10.0) << "New implementation significantly slower than old";
}

TEST_F(PointCloudComparisonTest, Performance_AllocationSpeed) {
    const size_t n_points = 10000;

    // Warm up
    gs::PointCloudNew warm_up;
    warm_up.allocate_basic(100);
    cudaDeviceSynchronize();

    // Time old implementation
    auto old_start = std::chrono::high_resolution_clock::now();
    gs::PointCloud old_pc;
    old_pc.allocate_gaussian(n_points, 3, 1, 3, 8);
    cudaDeviceSynchronize();
    auto old_end = std::chrono::high_resolution_clock::now();
    auto old_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            old_end - old_start)
                            .count();

    // Time new implementation
    auto new_start = std::chrono::high_resolution_clock::now();
    gs::PointCloudNew new_pc;
    new_pc.allocate_gaussian(n_points, 3, 1, 3, 8);
    cudaDeviceSynchronize();
    auto new_end = std::chrono::high_resolution_clock::now();
    auto new_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            new_end - new_start)
                            .count();

    double ratio = static_cast<double>(new_duration) / std::max(old_duration, 1L);
    std::println("Allocation time - Old: {} μs, New: {} μs (ratio: {:.2f}x)",
                 old_duration, new_duration, ratio);
}

TEST_F(PointCloudComparisonTest, Performance_CopySpeed) {
    const size_t n_points = 10000;
    std::vector<float> positions, colors;
    generate_random_data(positions, colors, n_points);

    // Old implementation
    gs::PointCloud old_pc = create_old_pointcloud(positions, colors, n_points);
    std::vector<float> old_out(n_points * 3);

    cudaDeviceSynchronize();
    auto old_start = std::chrono::high_resolution_clock::now();
    old_pc.copy_to_host(old_out.data(), nullptr);
    cudaDeviceSynchronize();
    auto old_end = std::chrono::high_resolution_clock::now();
    auto old_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            old_end - old_start)
                            .count();

    // New implementation
    gs::PointCloudNew new_pc(positions.data(), colors.data(), n_points);
    std::vector<float> new_out(n_points * 3);

    cudaDeviceSynchronize();
    auto new_start = std::chrono::high_resolution_clock::now();
    new_pc.copy_to_host(new_out.data(), nullptr);
    cudaDeviceSynchronize();
    auto new_end = std::chrono::high_resolution_clock::now();
    auto new_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            new_end - new_start)
                            .count();

    double ratio = static_cast<double>(new_duration) / std::max(old_duration, 1L);
    std::println("Copy time - Old: {} μs, New: {} μs (ratio: {:.2f}x)",
                 old_duration, new_duration, ratio);
}

// ============= EDGE CASES =============

TEST_F(PointCloudComparisonTest, EdgeCase_SinglePoint) {
    const size_t n_points = 1;
    std::vector<float> positions = {1.0f, 2.0f, 3.0f};
    std::vector<float> colors = {0.5f, 0.6f, 0.7f};

    gs::PointCloud old_pc = create_old_pointcloud(positions, colors, n_points);
    gs::PointCloudNew new_pc(positions.data(), colors.data(), n_points);

    EXPECT_EQ(old_pc.num_points, 1);
    EXPECT_EQ(new_pc.size(), 1);

    std::vector<float> old_means, old_colors;
    std::vector<float> new_means, new_colors;

    copy_old_to_host(old_pc, old_means, old_colors);
    copy_new_to_host(new_pc, new_means, new_colors);

    EXPECT_TRUE(compare_arrays(old_means.data(), new_means.data(), 3));
    EXPECT_TRUE(compare_arrays(old_colors.data(), new_colors.data(), 3));
}

TEST_F(PointCloudComparisonTest, EdgeCase_VeryLargeCloud) {
    const size_t n_points = 100000;
    std::vector<float> positions, colors;
    generate_random_data(positions, colors, n_points);

    gs::PointCloud old_pc = create_old_pointcloud(positions, colors, n_points);
    gs::PointCloudNew new_pc(positions.data(), colors.data(), n_points);

    EXPECT_EQ(old_pc.num_points, n_points);
    EXPECT_EQ(new_pc.size(), n_points);
}

// ============= DIAGNOSTIC STRING =============

TEST_F(PointCloudComparisonTest, DiagnosticString_Basic) {
    gs::PointCloudNew pc;
    pc.allocate_basic(100);

    std::string diag = pc.str();
    EXPECT_TRUE(diag.find("points=100") != std::string::npos);
    EXPECT_TRUE(diag.find("gaussian=false") != std::string::npos);
}

TEST_F(PointCloudComparisonTest, DiagnosticString_Gaussian) {
    gs::PointCloudNew pc;
    pc.allocate_gaussian(500, 3, 1, 3, 8);

    std::string diag = pc.str();
    EXPECT_TRUE(diag.find("points=500") != std::string::npos);
    EXPECT_TRUE(diag.find("gaussian=true") != std::string::npos);
}
