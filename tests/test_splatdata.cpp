/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <print>
#include <torch/torch.h>
#include <vector>

#include "core/parameters.hpp"
#include "core/point_cloud_new.hpp"
#include "core/point_cloud_ref.hpp"
#include "core/splat_data_new.hpp"
#include "core/splat_data_ref.hpp"
#include "core/tensor.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace {

    constexpr float FLOAT_TOLERANCE = 1e-3f;
    constexpr float ACTIVATION_TOLERANCE = 1e-2f; // More lenient for activations

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t error = call;                                                     \
        ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error); \
    } while (0)

    // ============================================================================
    // Helper Functions
    // ============================================================================

    /**
     * @brief Convert torch::Tensor to gs::Tensor
     */
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

    /**
     * @brief Convert gs::Tensor to torch::Tensor
     */
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

    /**
     * @brief Compare two tensors (torch vs gs::Tensor)
     */
    bool tensors_are_close(const torch::Tensor& torch_tensor,
                           const gs::Tensor& gs_tensor,
                           float tolerance = 1e-3f,
                           bool verbose = false) {
        if (torch_tensor.dim() != static_cast<int64_t>(gs_tensor.ndim())) {
            if (verbose) {
                std::print("Dimension mismatch: torch={}, gs={}\n",
                           torch_tensor.dim(), gs_tensor.ndim());
            }
            return false;
        }

        for (int i = 0; i < torch_tensor.dim(); ++i) {
            if (torch_tensor.size(i) != static_cast<int64_t>(gs_tensor.shape()[i])) {
                if (verbose) {
                    std::print("Shape mismatch at dim {}: torch={}, gs={}\n",
                               i, torch_tensor.size(i), gs_tensor.shape()[i]);
                }
                return false;
            }
        }

        auto torch_cpu = torch_tensor.cpu().contiguous();
        auto gs_cpu = gs_tensor.cpu();

        if (torch_tensor.scalar_type() == torch::kFloat32 &&
            gs_tensor.dtype() == gs::DataType::Float32) {

            auto torch_data = torch_cpu.data_ptr<float>();
            auto gs_data = gs_cpu.to_vector();

            size_t mismatch_count = 0;
            float max_diff = 0.0f;

            for (int64_t i = 0; i < torch_cpu.numel(); ++i) {
                float diff = std::abs(torch_data[i] - gs_data[i]);
                max_diff = std::max(max_diff, diff);

                if (diff > tolerance) {
                    mismatch_count++;
                    if (verbose && mismatch_count <= 5) {
                        std::print("Mismatch at index {}: torch={}, gs={}, diff={}\n",
                                   i, torch_data[i], gs_data[i], diff);
                    }
                }
            }

            if (verbose && mismatch_count > 0) {
                std::print("Total mismatches: {}/{}, max_diff={}\n",
                           mismatch_count, torch_cpu.numel(), max_diff);
            }

            return mismatch_count == 0;
        }

        return false;
    }

    /**
     * @brief Create a random point cloud for testing (PointCloudRef version)
     */
    gs::PointCloudRef create_test_point_cloud_ref(int n_points, int sh_degree = 3) {
        // Create tensors directly
        auto means = torch::rand({n_points, 3}, torch::kCUDA) * 20.0f - 10.0f;
        auto colors = torch::randint(0, 256, {n_points, 3}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        auto normals = torch::zeros({n_points, 3}, torch::kCUDA);

        // SH coefficients
        const size_t sh0_coeffs = 1;
        const size_t shN_coeffs = (sh_degree + 1) * (sh_degree + 1) - 1;

        auto sh0 = torch::randn({n_points, 3, static_cast<int64_t>(sh0_coeffs)}, torch::kCUDA) * 0.5f;
        auto shN = torch::randn({n_points, 3, static_cast<int64_t>(shN_coeffs)}, torch::kCUDA) * 0.5f;
        auto opacity = torch::randn({n_points, 1}, torch::kCUDA);
        auto scaling = torch::randn({n_points, 3}, torch::kCUDA) * 0.5f - 1.0f;

        // Rotation (identity + noise)
        auto rotation = torch::zeros({n_points, 4}, torch::kCUDA);
        rotation.index_put_({torch::indexing::Slice(), 0}, 1.0f); // w = 1
        rotation.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, 4)},
                            torch::randn({n_points, 3}, torch::kCUDA) * 0.1f);

        gs::PointCloudRef pc(means, colors);
        pc.normals = normals;
        pc.sh0 = sh0;
        pc.shN = shN;
        pc.opacity = opacity;
        pc.scaling = scaling;
        pc.rotation = rotation;

        return pc;
    }

    /**
     * @brief Convert PointCloudRef to PointCloudNew
     */
    gs::PointCloudNew point_cloud_ref_to_new(const gs::PointCloudRef& pc_ref) {
        gs::PointCloudNew pc_new;

        if (!pc_ref.means.defined() || pc_ref.means.size(0) == 0) {
            return pc_new;
        }

        // Convert torch tensors to gs::Tensor
        pc_new.means = torch_to_tensor(pc_ref.means);

        // Handle colors (may be uint8 or float32)
        if (pc_ref.colors.defined()) {
            if (pc_ref.colors.dtype() == torch::kUInt8) {
                // Convert uint8 [0, 255] to float32 [0, 255]
                auto colors_float = pc_ref.colors.to(torch::kFloat32);
                pc_new.colors = torch_to_tensor(colors_float);
            } else {
                pc_new.colors = torch_to_tensor(pc_ref.colors);
            }
        }

        // Copy Gaussian attributes if present
        if (pc_ref.is_gaussian()) {
            if (pc_ref.normals.defined()) {
                pc_new.normals = torch_to_tensor(pc_ref.normals);
            }

            // CRITICAL: ACTUAL ref format is [N, coeffs, channels], NOT [N, channels, coeffs]!
            // The documentation in splat_data_ref.hpp is WRONG - the actual storage is different
            if (pc_ref.sh0.defined()) {
                // sh0 is [N, 3, 1] in Ref but stored/returned as [N, 1, 3] - direct conversion
                pc_new.sh0 = torch_to_tensor(pc_ref.sh0);
            }

            if (pc_ref.shN.defined()) {
                // shN is [N, 3, coeffs] in Ref but stored/returned as [N, coeffs, 3] - direct conversion
                pc_new.shN = torch_to_tensor(pc_ref.shN);
            }

            if (pc_ref.opacity.defined()) {
                pc_new.opacity = torch_to_tensor(pc_ref.opacity);
            }

            if (pc_ref.scaling.defined()) {
                pc_new.scaling = torch_to_tensor(pc_ref.scaling);
            }

            if (pc_ref.rotation.defined()) {
                pc_new.rotation = torch_to_tensor(pc_ref.rotation);
            }
        }

        pc_new.attribute_names = pc_ref.attribute_names;

        return pc_new;
    }

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class SplatDataComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        torch::manual_seed(42);
        gs::Tensor::manual_seed(42);
    }

    gs::param::TrainingParameters create_test_params(int sh_degree = 3) {
        gs::param::TrainingParameters params;
        params.optimization.sh_degree = sh_degree;
        params.optimization.init_opacity = 0.5f;
        params.optimization.init_scaling = 1.0f;
        params.optimization.random = false;
        return params;
    }
};

// ============================================================================
// Construction and Initialization Tests
// ============================================================================

TEST_F(SplatDataComparisonTest, BasicConstruction_Comparison) {
    const int n_points = 100;
    const int sh_degree = 3;
    const float scene_scale = 5.0f;

    // Create test data with ACTUAL REF layout [N, coeffs, channels]
    auto means_torch = torch::randn({n_points, 3}, torch::kCUDA);
    auto sh0_torch = torch::randn({n_points, 1, 3}, torch::kCUDA);  // [N, 1, 3] coeffs, channels
    auto shN_torch = torch::randn({n_points, 15, 3}, torch::kCUDA); // [N, 15, 3] coeffs, channels
    auto scaling_torch = torch::randn({n_points, 3}, torch::kCUDA);
    auto rotation_torch = torch::randn({n_points, 4}, torch::kCUDA);
    auto opacity_torch = torch::randn({n_points, 1}, torch::kCUDA);

    // Convert to gs::Tensor - same layout for both
    auto means_tensor = torch_to_tensor(means_torch);
    auto sh0_tensor = torch_to_tensor(sh0_torch); // [N, 1, 3]
    auto shN_tensor = torch_to_tensor(shN_torch); // [N, 15, 3]
    auto scaling_tensor = torch_to_tensor(scaling_torch);
    auto rotation_tensor = torch_to_tensor(rotation_torch);
    auto opacity_tensor = torch_to_tensor(opacity_torch);

    // Create SplatData instances - both use [N, coeffs, channels] layout
    gs::SplatDataRef splat_ref(sh_degree, means_torch, sh0_torch, shN_torch,
                               scaling_torch, rotation_torch, opacity_torch, scene_scale);

    gs::SplatDataNew splat_new(sh_degree, means_tensor, sh0_tensor, shN_tensor,
                               scaling_tensor, rotation_tensor, opacity_tensor, scene_scale);

    // Compare basic properties
    EXPECT_EQ(splat_ref.size(), splat_new.size());
    EXPECT_EQ(splat_ref.get_active_sh_degree(), splat_new.get_active_sh_degree());
    EXPECT_FLOAT_EQ(splat_ref.get_scene_scale(), splat_new.get_scene_scale());

    // Compare means (should be exact since we copied the data)
    EXPECT_TRUE(tensors_are_close(splat_ref.get_means(), splat_new.get_means(), FLOAT_TOLERANCE));
}

TEST_F(SplatDataComparisonTest, InitFromPointCloud_Comparison) {
    const int n_points = 200;
    const int sh_degree = 2;

    auto params = create_test_params(sh_degree);
    auto pc_ref = create_test_point_cloud_ref(n_points, sh_degree);
    auto pc_new = point_cloud_ref_to_new(pc_ref);

    // Create scene center
    auto scene_center_torch = torch::tensor({0.0f, 0.0f, 0.0f}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::from_vector({0.0f, 0.0f, 0.0f},
                                                       gs::TensorShape({3}), gs::Device::CUDA);

    // Initialize SplatData from point clouds
    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);

    ASSERT_TRUE(result_ref.has_value()) << "Ref init failed: " << result_ref.error();
    ASSERT_TRUE(result_new.has_value()) << "New init failed: " << result_new.error();

    auto& splat_ref = result_ref.value();
    auto& splat_new = result_new.value();

    // Compare sizes
    EXPECT_EQ(splat_ref.size(), splat_new.size());
    EXPECT_EQ(splat_ref.get_active_sh_degree(), splat_new.get_active_sh_degree());

    // Compare means (should be similar since initialized from same point cloud)
    EXPECT_TRUE(tensors_are_close(splat_ref.get_means(), splat_new.get_means(), FLOAT_TOLERANCE));

    std::print("✓ InitFromPointCloud: ref_size={}, new_size={}\n",
               splat_ref.size(), splat_new.size());
}

TEST_F(SplatDataComparisonTest, RandomInitialization_Comparison) {
    auto params = create_test_params(3);
    params.optimization.random = true;
    params.optimization.init_num_pts = 500;
    params.optimization.init_extent = 10.0f;

    auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

    // Empty point clouds for random init
    gs::PointCloudRef pc_ref;
    gs::PointCloudNew pc_new;

    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);

    ASSERT_TRUE(result_ref.has_value());
    ASSERT_TRUE(result_new.has_value());

    auto& splat_ref = result_ref.value();
    auto& splat_new = result_new.value();

    EXPECT_EQ(splat_ref.size(), params.optimization.init_num_pts);
    EXPECT_EQ(splat_new.size(), params.optimization.init_num_pts);

    std::print("✓ RandomInitialization: both initialized with {} points\n",
               params.optimization.init_num_pts);
}

// ============================================================================
// Getter Methods Comparison Tests
// ============================================================================

TEST_F(SplatDataComparisonTest, GetMeans_Comparison) {
    const int n_points = 150;
    const int sh_degree = 3;

    auto pc_ref = create_test_point_cloud_ref(n_points, sh_degree);
    auto pc_new = point_cloud_ref_to_new(pc_ref);

    auto params = create_test_params(sh_degree);
    auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);

    ASSERT_TRUE(result_ref.has_value());
    ASSERT_TRUE(result_new.has_value());

    auto& splat_ref = result_ref.value();
    auto& splat_new = result_new.value();

    auto means_ref = splat_ref.get_means();
    auto means_new = splat_new.get_means();

    EXPECT_TRUE(tensors_are_close(means_ref, means_new, FLOAT_TOLERANCE));

    std::print("✓ GetMeans: shapes match, values close\n");
}

TEST_F(SplatDataComparisonTest, GetOpacity_Comparison) {
    const int n_points = 100;

    // Create identical raw opacity values
    auto opacity_raw = torch::randn({n_points, 1}, torch::kCUDA);
    auto opacity_tensor = torch_to_tensor(opacity_raw);

    auto means_torch = torch::randn({n_points, 3}, torch::kCUDA);
    auto means_tensor = torch_to_tensor(means_torch);
    auto sh0_torch = torch::randn({n_points, 1, 3}, torch::kCUDA);
    auto sh0_tensor = torch_to_tensor(sh0_torch);
    auto shN_torch = torch::randn({n_points, 15, 3}, torch::kCUDA);
    auto shN_tensor = torch_to_tensor(shN_torch);
    auto scaling_torch = torch::randn({n_points, 3}, torch::kCUDA);
    auto scaling_tensor = torch_to_tensor(scaling_torch);
    auto rotation_torch = torch::randn({n_points, 4}, torch::kCUDA);
    auto rotation_tensor = torch_to_tensor(rotation_torch);

    gs::SplatDataRef splat_ref(3, means_torch, sh0_torch, shN_torch,
                               scaling_torch, rotation_torch, opacity_raw, 1.0f);
    gs::SplatDataNew splat_new(3, means_tensor, sh0_tensor, shN_tensor,
                               scaling_tensor, rotation_tensor, opacity_tensor, 1.0f);

    auto opacity_ref = splat_ref.get_opacity();
    auto opacity_new = splat_new.get_opacity();

    // Both should apply sigmoid activation
    EXPECT_TRUE(tensors_are_close(opacity_ref, opacity_new, ACTIVATION_TOLERANCE, true));

    std::print("✓ GetOpacity: sigmoid activation matches\n");
}

TEST_F(SplatDataComparisonTest, GetScaling_Comparison) {
    const int n_points = 100;

    // Create identical raw scaling values (log scale)
    auto scaling_raw = torch::randn({n_points, 3}, torch::kCUDA) * 0.5f - 1.0f;
    auto scaling_tensor = torch_to_tensor(scaling_raw);

    auto means_torch = torch::randn({n_points, 3}, torch::kCUDA);
    auto means_tensor = torch_to_tensor(means_torch);
    auto sh0_torch = torch::randn({n_points, 1, 3}, torch::kCUDA);
    auto sh0_tensor = torch_to_tensor(sh0_torch);
    auto shN_torch = torch::randn({n_points, 15, 3}, torch::kCUDA);
    auto shN_tensor = torch_to_tensor(shN_torch);
    auto rotation_torch = torch::randn({n_points, 4}, torch::kCUDA);
    auto rotation_tensor = torch_to_tensor(rotation_torch);
    auto opacity_torch = torch::randn({n_points, 1}, torch::kCUDA);
    auto opacity_tensor = torch_to_tensor(opacity_torch);

    gs::SplatDataRef splat_ref(3, means_torch, sh0_torch, shN_torch,
                               scaling_raw, rotation_torch, opacity_torch, 1.0f);
    gs::SplatDataNew splat_new(3, means_tensor, sh0_tensor, shN_tensor,
                               scaling_tensor, rotation_tensor, opacity_tensor, 1.0f);

    auto scaling_ref = splat_ref.get_scaling();
    auto scaling_new = splat_new.get_scaling();

    // Both should apply exp activation
    EXPECT_TRUE(tensors_are_close(scaling_ref, scaling_new, ACTIVATION_TOLERANCE, true));

    std::print("✓ GetScaling: exp activation matches\n");
}

TEST_F(SplatDataComparisonTest, GetRotation_Comparison) {
    const int n_points = 100;

    // Create identical raw rotation values (unnormalized quaternions)
    auto rotation_raw = torch::randn({n_points, 4}, torch::kCUDA);
    auto rotation_tensor = torch_to_tensor(rotation_raw);

    auto means_torch = torch::randn({n_points, 3}, torch::kCUDA);
    auto means_tensor = torch_to_tensor(means_torch);
    auto sh0_torch = torch::randn({n_points, 1, 3}, torch::kCUDA);
    auto sh0_tensor = torch_to_tensor(sh0_torch);
    auto shN_torch = torch::randn({n_points, 15, 3}, torch::kCUDA);
    auto shN_tensor = torch_to_tensor(shN_torch);
    auto scaling_torch = torch::randn({n_points, 3}, torch::kCUDA);
    auto scaling_tensor = torch_to_tensor(scaling_torch);
    auto opacity_torch = torch::randn({n_points, 1}, torch::kCUDA);
    auto opacity_tensor = torch_to_tensor(opacity_torch);

    gs::SplatDataRef splat_ref(3, means_torch, sh0_torch, shN_torch,
                               scaling_torch, rotation_raw, opacity_torch, 1.0f);
    gs::SplatDataNew splat_new(3, means_tensor, sh0_tensor, shN_tensor,
                               scaling_tensor, rotation_tensor, opacity_tensor, 1.0f);

    auto rotation_ref = splat_ref.get_rotation();
    auto rotation_new = splat_new.get_rotation();

    // Both should normalize quaternions
    EXPECT_TRUE(tensors_are_close(rotation_ref, rotation_new, ACTIVATION_TOLERANCE, true));

    std::print("✓ GetRotation: normalization matches\n");
}

TEST_F(SplatDataComparisonTest, GetSHS_Comparison) {
    const int n_points = 100;
    const int sh_degree = 2;

    auto pc_ref = create_test_point_cloud_ref(n_points, sh_degree);
    auto pc_new = point_cloud_ref_to_new(pc_ref);

    auto params = create_test_params(sh_degree);
    auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);

    ASSERT_TRUE(result_ref.has_value());
    ASSERT_TRUE(result_new.has_value());

    auto splat_ref = std::move(result_ref).value();
    auto splat_new = std::move(result_new).value();

    auto shs_ref = splat_ref.get_shs();
    auto shs_new = splat_new.get_shs();

    // Both implementations return [N, total_coeffs, 3] (coeffs, channels) - ACTUAL REF FORMAT
    // They should match directly without any transposition
    EXPECT_TRUE(tensors_are_close(shs_ref, shs_new, FLOAT_TOLERANCE, true));

    std::print("✓ GetSHS: concatenated SH coefficients match\n");
}

// ============================================================================
// Transformation Tests
// ============================================================================

TEST_F(SplatDataComparisonTest, Transform_Translation_Comparison) {
    const int n_points = 100;
    const int sh_degree = 2;

    auto pc_ref = create_test_point_cloud_ref(n_points, sh_degree);
    auto pc_new = point_cloud_ref_to_new(pc_ref);

    auto params = create_test_params(sh_degree);
    auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);

    ASSERT_TRUE(result_ref.has_value());
    ASSERT_TRUE(result_new.has_value());

    auto splat_ref = std::move(result_ref).value();
    auto splat_new = std::move(result_new).value();

    // Create translation matrix
    glm::mat4 transform = glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, 3.0f, -2.0f));

    // CRITICAL: Detach tensors from autograd before in-place operations
    // SplatDataRef::transform() does in-place operations which fail on leaf variables with requires_grad
    {
        torch::NoGradGuard no_grad;
        splat_ref.means().set_requires_grad(false);
        splat_ref.rotation_raw().set_requires_grad(false);
        splat_ref.scaling_raw().set_requires_grad(false);
        splat_ref.transform(transform);
    }

    splat_new.transform(transform);

    // Compare transformed positions
    auto means_ref = splat_ref.get_means();
    auto means_new = splat_new.get_means();

    EXPECT_TRUE(tensors_are_close(means_ref, means_new, FLOAT_TOLERANCE));

    std::print("✓ Transform_Translation: positions match after translation\n");
}

TEST_F(SplatDataComparisonTest, Transform_Rotation_Comparison) {
    // CRITICAL: Fixed seed for reproducible test
    torch::manual_seed(42);
    gs::Tensor::manual_seed(42);

    const int n_points = 100;
    const int sh_degree = 2;

    // Create IDENTICAL data for both implementations
    auto means_torch = torch::randn({n_points, 3}, torch::kCUDA);
    auto sh0_torch = torch::randn({n_points, 1, 3}, torch::kCUDA);
    auto shN_torch = torch::randn({n_points, 8, 3}, torch::kCUDA);
    auto scaling_torch = torch::randn({n_points, 3}, torch::kCUDA);
    auto rotation_torch = torch::randn({n_points, 4}, torch::kCUDA);
    auto opacity_torch = torch::randn({n_points, 1}, torch::kCUDA);

    // Convert to gs::Tensor
    auto means_tensor = torch_to_tensor(means_torch);
    auto sh0_tensor = torch_to_tensor(sh0_torch);
    auto shN_tensor = torch_to_tensor(shN_torch);
    auto scaling_tensor = torch_to_tensor(scaling_torch);
    auto rotation_tensor = torch_to_tensor(rotation_torch);
    auto opacity_tensor = torch_to_tensor(opacity_torch);

    // Create SplatData instances with IDENTICAL data
    auto splat_ref = gs::SplatDataRef(sh_degree, means_torch.clone(), sh0_torch.clone(),
                                      shN_torch.clone(), scaling_torch.clone(),
                                      rotation_torch.clone(), opacity_torch.clone(), 10.0f);
    auto splat_new = gs::SplatDataNew(sh_degree, means_tensor.clone(), sh0_tensor.clone(),
                                      shN_tensor.clone(), scaling_tensor.clone(),
                                      rotation_tensor.clone(), opacity_tensor.clone(), 10.0f);

    // Create rotation matrix (45 degrees around Y axis)
    glm::mat4 transform = glm::rotate(glm::mat4(1.0f), glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    // CRITICAL: Detach tensors from autograd before in-place operations
    {
        torch::NoGradGuard no_grad;
        splat_ref.means().set_requires_grad(false);
        splat_ref.rotation_raw().set_requires_grad(false);
        splat_ref.scaling_raw().set_requires_grad(false);
        splat_ref.transform(transform);
    }

    splat_new.transform(transform);

    // Compare transformed positions
    auto means_ref = splat_ref.get_means();
    auto means_new = splat_new.get_means();

    EXPECT_TRUE(tensors_are_close(means_ref, means_new, FLOAT_TOLERANCE));

    // FIRST: Compare RAW rotations (before normalization)
    auto rotations_ref_raw = splat_ref.rotation_raw();
    auto rotations_new_raw = splat_new.rotation_raw();

    std::cout << "\n=== RAW ROTATION COMPARISON (before normalization) ===" << std::endl;
    std::cout << "REF raw rotations (first 3):" << std::endl;
    std::cout << rotations_ref_raw.slice(0, 0, 3) << std::endl;

    std::cout << "NEW raw rotations (first 3):" << std::endl;
    auto rot_new_raw_cpu = rotations_new_raw.cpu();
    auto rot_new_raw_vec = rot_new_raw_cpu.to_vector();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << rot_new_raw_vec[i * 4 + j] << " ";
        }
        std::cout << std::endl;
    }

    // Compare raw values
    auto raw_diff = torch_to_tensor(rotations_ref_raw).sub(rotations_new_raw).abs();
    auto raw_max_diff = raw_diff.max().item();
    std::cout << "\nRAW Max absolute difference: " << raw_max_diff << std::endl;

    EXPECT_TRUE(tensors_are_close(rotations_ref_raw, rotations_new_raw, 1e-5f))
        << "RAW rotation quaternions should match before normalization";

    // SECOND: Compare normalized rotations
    auto rotations_ref = splat_ref.get_rotation();
    auto rotations_new = splat_new.get_rotation();

    std::cout << "\n=== NORMALIZED ROTATION COMPARISON ===" << std::endl;
    std::cout << "REF normalized rotations (first 3):" << std::endl;
    std::cout << rotations_ref.slice(0, 0, 3) << std::endl;

    std::cout << "NEW normalized rotations (first 3):" << std::endl;
    auto rot_new_cpu = rotations_new.cpu();
    auto rot_new_vec = rot_new_cpu.to_vector();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << rot_new_vec[i * 4 + j] << " ";
        }
        std::cout << std::endl;
    }

    auto norm_diff = torch_to_tensor(rotations_ref).sub(rotations_new).abs();
    auto norm_max_diff = norm_diff.max().item();
    std::cout << "\nNORMALIZED Max absolute difference: " << norm_max_diff << std::endl;
    std::cout << "Tolerance: " << ACTIVATION_TOLERANCE << std::endl;

    EXPECT_TRUE(tensors_are_close(rotations_ref, rotations_new, ACTIVATION_TOLERANCE));

    std::print("✓ Transform_Rotation: positions and rotations match after rotation\n");
}

TEST_F(SplatDataComparisonTest, Transform_Scale_Comparison) {
    const int n_points = 100;
    const int sh_degree = 2;

    // Create IDENTICAL data for both implementations
    auto means_torch = torch::randn({n_points, 3}, torch::kCUDA);
    auto sh0_torch = torch::randn({n_points, 1, 3}, torch::kCUDA);
    auto shN_torch = torch::randn({n_points, 8, 3}, torch::kCUDA);
    auto scaling_torch = torch::randn({n_points, 3}, torch::kCUDA);
    auto rotation_torch = torch::randn({n_points, 4}, torch::kCUDA);
    auto opacity_torch = torch::randn({n_points, 1}, torch::kCUDA);

    // Convert to gs::Tensor
    auto means_tensor = torch_to_tensor(means_torch);
    auto sh0_tensor = torch_to_tensor(sh0_torch);
    auto shN_tensor = torch_to_tensor(shN_torch);
    auto scaling_tensor = torch_to_tensor(scaling_torch);
    auto rotation_tensor = torch_to_tensor(rotation_torch);
    auto opacity_tensor = torch_to_tensor(opacity_torch);

    // Create SplatData instances with IDENTICAL data
    auto splat_ref = gs::SplatDataRef(sh_degree, means_torch.clone(), sh0_torch.clone(),
                                      shN_torch.clone(), scaling_torch.clone(),
                                      rotation_torch.clone(), opacity_torch.clone(), 10.0f);
    auto splat_new = gs::SplatDataNew(sh_degree, means_tensor.clone(), sh0_tensor.clone(),
                                      shN_tensor.clone(), scaling_tensor.clone(),
                                      rotation_tensor.clone(), opacity_tensor.clone(), 10.0f);

    // Create scale matrix
    glm::mat4 transform = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 2.0f, 2.0f));

    // CRITICAL: Detach tensors from autograd before in-place operations
    {
        torch::NoGradGuard no_grad;
        splat_ref.means().set_requires_grad(false);
        splat_ref.rotation_raw().set_requires_grad(false);
        splat_ref.scaling_raw().set_requires_grad(false);
        splat_ref.transform(transform);
    }

    splat_new.transform(transform);

    // Compare transformed positions and scaling
    auto means_ref = splat_ref.get_means();
    auto means_new = splat_new.get_means();
    auto scaling_ref = splat_ref.get_scaling();
    auto scaling_new = splat_new.get_scaling();

    EXPECT_TRUE(tensors_are_close(means_ref, means_new, FLOAT_TOLERANCE));
    EXPECT_TRUE(tensors_are_close(scaling_ref, scaling_new, ACTIVATION_TOLERANCE));

    std::print("✓ Transform_Scale: positions and scaling match after scaling\n");
}

// ============================================================================
// SH Degree Management Tests
// ============================================================================

TEST_F(SplatDataComparisonTest, IncrementSHDegree_Comparison) {
    const int n_points = 100;
    const int max_sh_degree = 3;

    auto pc_ref = create_test_point_cloud_ref(n_points, max_sh_degree);
    auto pc_new = point_cloud_ref_to_new(pc_ref);

    auto params = create_test_params(max_sh_degree);
    auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);

    ASSERT_TRUE(result_ref.has_value());
    ASSERT_TRUE(result_new.has_value());

    auto splat_ref = std::move(result_ref).value();
    auto splat_new = std::move(result_new).value();

    // Both should start at degree 0
    EXPECT_EQ(splat_ref.get_active_sh_degree(), 0);
    EXPECT_EQ(splat_new.get_active_sh_degree(), 0);

    // Increment several times
    for (int i = 0; i < max_sh_degree + 2; ++i) {
        splat_ref.increment_sh_degree();
        splat_new.increment_sh_degree();

        EXPECT_EQ(splat_ref.get_active_sh_degree(), splat_new.get_active_sh_degree());
    }

    // Should cap at max_sh_degree
    EXPECT_EQ(splat_ref.get_active_sh_degree(), max_sh_degree);
    EXPECT_EQ(splat_new.get_active_sh_degree(), max_sh_degree);

    std::print("✓ IncrementSHDegree: both implementations cap correctly at degree {}\n", max_sh_degree);
}

// ============================================================================
// Point Cloud Conversion Tests
// ============================================================================

TEST_F(SplatDataComparisonTest, ToPointCloud_Comparison) {
    const int n_points = 100;
    const int sh_degree = 2;

    auto pc_input_ref = create_test_point_cloud_ref(n_points, sh_degree);
    auto pc_input_new = point_cloud_ref_to_new(pc_input_ref);

    auto params = create_test_params(sh_degree);
    auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_input_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_input_new);

    ASSERT_TRUE(result_ref.has_value());
    ASSERT_TRUE(result_new.has_value());

    auto splat_ref = std::move(result_ref).value();
    auto splat_new = std::move(result_new).value();

    // Convert back to point cloud
    auto pc_output_new = splat_new.to_point_cloud();

    EXPECT_EQ(n_points, pc_output_new.size());
    EXPECT_TRUE(pc_output_new.is_gaussian());

    // Compare means from new implementation
    auto means_splat = splat_new.get_means();
    auto means_pc = pc_output_new.means; // This is already a gs::Tensor

    // Compare the two gs::Tensor objects directly
    EXPECT_TRUE(means_splat.all_close(means_pc, FLOAT_TOLERANCE, FLOAT_TOLERANCE));

    std::print("✓ ToPointCloud: conversion preserves data\n");
}

// ============================================================================
// Attribute Names Test
// ============================================================================

TEST_F(SplatDataComparisonTest, AttributeNames_Comparison) {
    const int n_points = 50;
    const int sh_degree = 3;

    auto pc_ref = create_test_point_cloud_ref(n_points, sh_degree);
    auto pc_new = point_cloud_ref_to_new(pc_ref);

    auto params = create_test_params(sh_degree);
    auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);

    ASSERT_TRUE(result_ref.has_value());
    ASSERT_TRUE(result_new.has_value());

    auto splat_ref = std::move(result_ref).value();
    auto splat_new = std::move(result_new).value();

    auto attrs_ref = splat_ref.get_attribute_names();
    auto attrs_new = splat_new.get_attribute_names();

    EXPECT_EQ(attrs_ref.size(), attrs_new.size());

    for (size_t i = 0; i < attrs_ref.size(); ++i) {
        EXPECT_EQ(attrs_ref[i], attrs_new[i]);
    }

    std::print("✓ AttributeNames: {} attributes match\n", attrs_ref.size());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SplatDataComparisonTest, EmptyPointCloud_Comparison) {
    auto params = create_test_params(3);
    params.optimization.random = false;

    gs::PointCloudRef pc_ref;
    gs::PointCloudNew pc_new;

    auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

    // Both should fail gracefully
    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);

    // Check if both fail or both succeed with 0 points
    if (result_ref.has_value() && result_new.has_value()) {
        EXPECT_EQ(result_ref.value().size(), 0);
        EXPECT_EQ(result_new.value().size(), 0);
    } else {
        EXPECT_FALSE(result_ref.has_value());
        EXPECT_FALSE(result_new.has_value());
    }

    std::print("✓ EmptyPointCloud: both implementations handle empty input consistently\n");
}

TEST_F(SplatDataComparisonTest, SinglePoint_Comparison) {
    const int n_points = 1;
    const int sh_degree = 2;

    auto pc_ref = create_test_point_cloud_ref(n_points, sh_degree);
    auto pc_new = point_cloud_ref_to_new(pc_ref);

    auto params = create_test_params(sh_degree);
    auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);

    ASSERT_TRUE(result_ref.has_value());
    ASSERT_TRUE(result_new.has_value());

    auto& splat_ref = result_ref.value();
    auto& splat_new = result_new.value();

    EXPECT_EQ(splat_ref.size(), 1);
    EXPECT_EQ(splat_new.size(), 1);

    std::print("✓ SinglePoint: both handle single point correctly\n");
}

TEST_F(SplatDataComparisonTest, LargePointCloud_Comparison) {
    const int n_points = 50000;
    const int sh_degree = 3;

    auto pc_ref = create_test_point_cloud_ref(n_points, sh_degree);
    auto pc_new = point_cloud_ref_to_new(pc_ref);

    auto params = create_test_params(sh_degree);
    auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);

    ASSERT_TRUE(result_ref.has_value());
    ASSERT_TRUE(result_new.has_value());

    auto& splat_ref = result_ref.value();
    auto& splat_new = result_new.value();

    EXPECT_EQ(splat_ref.size(), n_points);
    EXPECT_EQ(splat_new.size(), n_points);

    std::print("✓ LargePointCloud: both handle {} points correctly\n", n_points);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_F(SplatDataComparisonTest, ValidationTest) {
    const int n_points = 100;
    const int sh_degree = 2;

    auto pc_ref = create_test_point_cloud_ref(n_points, sh_degree);
    auto pc_new = point_cloud_ref_to_new(pc_ref);

    auto params = create_test_params(sh_degree);
    auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);

    ASSERT_TRUE(result_ref.has_value());
    ASSERT_TRUE(result_new.has_value());

    auto& splat_new = result_new.value();

    EXPECT_TRUE(splat_new.is_valid());

    std::print("✓ Validation: SplatDataNew passes validation checks\n");
}

// ============================================================================
// Performance Tests
// ============================================================================

class SplatDataPerformanceTest : public ::testing::Test {
protected:
    struct BenchmarkResult {
        std::string name;
        int n_points;
        int sh_degree;
        double time_ms_ref;
        double time_ms_new;
        double speedup;
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

    template <typename Func>
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
        std::print("\n{}\n", std::string(100, '='));
        std::print("SPLATDATA PERFORMANCE COMPARISON: SplatDataRef vs SplatDataNew\n");
        std::print("{}\n\n", std::string(100, '='));

        std::print("{:<40} {:>10} {:>8} {:>12} {:>12} {:>12}\n",
                   "Benchmark", "N Points", "SH Deg", "Ref (ms)", "New (ms)", "Speedup");
        std::print("{}\n", std::string(100, '-'));

        for (const auto& r : results) {
            std::print("{:<40} {:>10} {:>8} {:>12.3f} {:>12.3f} {:>11.2f}x\n",
                       r.name, r.n_points, r.sh_degree, r.time_ms_ref, r.time_ms_new, r.speedup);
        }

        std::print("{}\n", std::string(100, '='));

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

        std::print("\nSUMMARY:\n");
        std::print("  Average Speedup: {:.2f}x\n", avg_speedup);
        std::print("  Maximum Speedup: {:.2f}x\n", max_speedup);
        std::print("  Minimum Speedup: {:.2f}x\n", min_speedup);
        std::print("{}\n\n", std::string(100, '='));
    }

    gs::param::TrainingParameters create_test_params(int sh_degree = 3) {
        gs::param::TrainingParameters params;
        params.optimization.sh_degree = sh_degree;
        params.optimization.init_opacity = 0.5f;
        params.optimization.init_scaling = 1.0f;
        params.optimization.random = false;
        return params;
    }
};

TEST_F(SplatDataPerformanceTest, DISABLED_InitializationBenchmark) {
    std::vector<BenchmarkResult> results;

    struct TestConfig {
        std::string name;
        int n_points;
        int sh_degree;
    };

    std::vector<TestConfig> configs = {
        {"Small_Degree0", 1000, 0},
        {"Small_Degree2", 1000, 2},
        {"Medium_Degree2", 10000, 2},
        {"Medium_Degree3", 10000, 3},
        {"Large_Degree2", 50000, 2},
        {"Large_Degree3", 50000, 3},
        {"XLarge_Degree3", 100000, 3},
    };

    for (const auto& cfg : configs) {
        auto pc_ref = create_test_point_cloud_ref(cfg.n_points, cfg.sh_degree);
        auto pc_new = point_cloud_ref_to_new(pc_ref);

        auto params = create_test_params(cfg.sh_degree);
        auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
        auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

        BenchmarkResult result;
        result.name = cfg.name;
        result.n_points = cfg.n_points;
        result.sh_degree = cfg.sh_degree;

        // Benchmark ref implementation
        result.time_ms_ref = benchmark([&]() {
            auto res = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
        });

        // Benchmark new implementation
        result.time_ms_new = benchmark([&]() {
            auto res = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);
        });

        result.speedup = result.time_ms_ref / result.time_ms_new;
        results.push_back(result);
    }

    print_results(results);
}

TEST_F(SplatDataPerformanceTest, DISABLED_TransformationBenchmark) {
    std::vector<BenchmarkResult> results;

    struct TestConfig {
        std::string name;
        int n_points;
        int sh_degree;
    };

    std::vector<TestConfig> configs = {
        {"Transform_Small", 1000, 2},
        {"Transform_Medium", 10000, 3},
        {"Transform_Large", 50000, 3},
        {"Transform_XLarge", 100000, 3},
    };

    glm::mat4 transform = glm::rotate(glm::mat4(1.0f), glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    transform = glm::translate(transform, glm::vec3(5.0f, 0.0f, 0.0f));

    for (const auto& cfg : configs) {
        auto params = create_test_params(cfg.sh_degree);
        auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
        auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

        BenchmarkResult result;
        result.name = cfg.name;
        result.n_points = cfg.n_points;
        result.sh_degree = cfg.sh_degree;

        // Benchmark ref implementation - create fresh copy each time
        result.time_ms_ref = benchmark([&]() {
            auto pc_temp = create_test_point_cloud_ref(cfg.n_points, cfg.sh_degree);
            auto res = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_temp);
            if (res.has_value()) {
                res.value().transform(transform);
            }
        },
                                       2, 5);

        // Benchmark new implementation - create fresh copy each time
        result.time_ms_new = benchmark([&]() {
            auto pc_temp = point_cloud_ref_to_new(create_test_point_cloud_ref(cfg.n_points, cfg.sh_degree));
            auto res = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_temp);
            if (res.has_value()) {
                res.value().transform(transform);
            }
        },
                                       2, 5);

        result.speedup = result.time_ms_ref / result.time_ms_new;
        results.push_back(result);
    }

    print_results(results);
}

TEST_F(SplatDataPerformanceTest, DISABLED_GettersBenchmark) {
    std::vector<BenchmarkResult> results;

    const int n_points = 50000;
    const int sh_degree = 3;

    auto pc_ref = create_test_point_cloud_ref(n_points, sh_degree);
    auto pc_new = point_cloud_ref_to_new(pc_ref);

    auto params = create_test_params(sh_degree);
    auto scene_center_torch = torch::zeros({3}, torch::kCUDA);
    auto scene_center_tensor = gs::Tensor::zeros({3}, gs::Device::CUDA);

    auto result_ref = gs::SplatDataRef::init_model_from_pointcloud(params, scene_center_torch, pc_ref);
    auto result_new = gs::SplatDataNew::init_model_from_pointcloud(params, scene_center_tensor, pc_new);

    ASSERT_TRUE(result_ref.has_value());
    ASSERT_TRUE(result_new.has_value());

    auto splat_ref = std::move(result_ref).value();
    auto splat_new = std::move(result_new).value();

    // Benchmark individual getters
    std::vector<std::pair<std::string, std::function<void()>>> tests_ref = {
        {"GetMeans", [&]() { auto m = splat_ref.get_means(); }},
        {"GetOpacity", [&]() { auto o = splat_ref.get_opacity(); }},
        {"GetRotation", [&]() { auto r = splat_ref.get_rotation(); }},
        {"GetScaling", [&]() { auto s = splat_ref.get_scaling(); }},
        {"GetSHS", [&]() { auto sh = splat_ref.get_shs(); }},
    };

    std::vector<std::pair<std::string, std::function<void()>>> tests_new = {
        {"GetMeans", [&]() { auto m = splat_new.get_means(); }},
        {"GetOpacity", [&]() { auto o = splat_new.get_opacity(); }},
        {"GetRotation", [&]() { auto r = splat_new.get_rotation(); }},
        {"GetScaling", [&]() { auto s = splat_new.get_scaling(); }},
        {"GetSHS", [&]() { auto sh = splat_new.get_shs(); }},
    };

    for (size_t i = 0; i < tests_ref.size(); ++i) {
        BenchmarkResult result;
        result.name = tests_ref[i].first;
        result.n_points = n_points;
        result.sh_degree = sh_degree;

        result.time_ms_ref = benchmark(tests_ref[i].second, 5, 20);
        result.time_ms_new = benchmark(tests_new[i].second, 5, 20);

        result.speedup = result.time_ms_ref / result.time_ms_new;
        results.push_back(result);
    }

    print_results(results);
}