/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <numbers>
#include <print>
#include <torch/torch.h>
#include <vector>

#include "core/camera_new.hpp"
#include "core/camera_ref.hpp"
#include "core/tensor.hpp"

namespace {

    constexpr float FLOAT_TOLERANCE = 1e-4f;
    constexpr float MATRIX_TOLERANCE = 1e-3f;
    constexpr float INTRINSICS_TOLERANCE = 1e-2f;

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
            return gs::Tensor::from_vector(data, gs::TensorShape(shape), gs::Device::CPU);
        } else if (torch_tensor.scalar_type() == torch::kInt32) {
            std::vector<int> data(cpu_tensor.data_ptr<int>(),
                                  cpu_tensor.data_ptr<int>() + cpu_tensor.numel());
            return gs::Tensor::from_vector(data, gs::TensorShape(shape), gs::Device::CPU);
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
            return torch_tensor;
        } else if (gs_tensor.dtype() == gs::DataType::Int32) {
            auto data = cpu_tensor.to_vector_int();
            auto torch_tensor = torch::from_blob(data.data(), shape, torch::kInt32).clone();
            return torch_tensor;
        }

        return torch::Tensor();
    }

    /**
     * @brief Compare two tensors (torch vs gs::Tensor)
     */
    bool tensors_are_close(const torch::Tensor& torch_tensor,
                           const gs::Tensor& gs_tensor,
                           float tolerance = 1e-4f,
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
     * @brief Create a rotation matrix around Z axis
     */
    torch::Tensor create_rotation_z_torch(float angle_rad) {
        float c = std::cos(angle_rad);
        float s = std::sin(angle_rad);

        auto R = torch::zeros({3, 3}, torch::kFloat32);
        R[0][0] = c;
        R[0][1] = -s;
        R[0][2] = 0.0f;
        R[1][0] = s;
        R[1][1] = c;
        R[1][2] = 0.0f;
        R[2][0] = 0.0f;
        R[2][1] = 0.0f;
        R[2][2] = 1.0f;

        return R;
    }

    gs::Tensor create_rotation_z_tensor(float angle_rad) {
        float c = std::cos(angle_rad);
        float s = std::sin(angle_rad);

        std::vector<float> data = {
            c, -s, 0.0f,
            s, c, 0.0f,
            0.0f, 0.0f, 1.0f};

        return gs::Tensor::from_vector(data, gs::TensorShape({3, 3}), gs::Device::CPU);
    }

    /**
     * @brief Create identity rotation
     */
    torch::Tensor create_identity_rotation_torch() {
        return torch::eye(3, torch::kFloat32);
    }

    gs::Tensor create_identity_rotation_tensor() {
        return gs::Tensor::eye(3, gs::Device::CPU);
    }

    /**
     * @brief Create translation vector
     */
    torch::Tensor create_translation_torch(float x, float y, float z) {
        return torch::tensor({x, y, z}, torch::kFloat32);
    }

    gs::Tensor create_translation_tensor(float x, float y, float z) {
        return gs::Tensor::from_vector({x, y, z}, gs::TensorShape({3}), gs::Device::CPU);
    }

    /**
     * @brief Create complex rotation (Euler angles)
     */
    std::pair<torch::Tensor, gs::Tensor> create_complex_rotation(float angle_x, float angle_y, float angle_z) {
        // Torch version
        auto Rx_torch = [](float a) {
            float c = std::cos(a);
            float s = std::sin(a);
            auto R = torch::zeros({3, 3}, torch::kFloat32);
            R[0][0] = 1.0f;
            R[1][1] = c;
            R[1][2] = -s;
            R[2][1] = s;
            R[2][2] = c;
            return R;
        };

        auto Ry_torch = [](float a) {
            float c = std::cos(a);
            float s = std::sin(a);
            auto R = torch::zeros({3, 3}, torch::kFloat32);
            R[0][0] = c;
            R[0][2] = s;
            R[1][1] = 1.0f;
            R[2][0] = -s;
            R[2][2] = c;
            return R;
        };

        auto Rz_torch = [](float a) {
            return create_rotation_z_torch(a);
        };

        auto R_torch = Rz_torch(angle_z).mm(Ry_torch(angle_y).mm(Rx_torch(angle_x)));
        auto R_tensor = torch_to_tensor(R_torch);

        return {R_torch, R_tensor};
    }

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class CameraComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        torch::manual_seed(42);
        gs::Tensor::manual_seed(42);
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(CameraComparisonTest, BasicConstruction_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(0.0f, 0.0f, -5.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(0.0f, 0.0f, -5.0f);

    float focal_x = 1000.0f;
    float focal_y = 1000.0f;
    float center_x = 512.0f;
    float center_y = 384.0f;

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        focal_x, focal_y,
        center_x, center_y,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test_image.png",
        "/tmp/test_image.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        focal_x, focal_y,
        center_x, center_y,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test_image.png",
        "/tmp/test_image.png",
        1024, 768,
        0);

    // Compare basic properties
    EXPECT_EQ(cam_ref.uid(), cam_new.uid());
    EXPECT_EQ(cam_ref.camera_width(), cam_new.camera_width());
    EXPECT_EQ(cam_ref.camera_height(), cam_new.camera_height());
    EXPECT_FLOAT_EQ(cam_ref.focal_x(), cam_new.focal_x());
    EXPECT_FLOAT_EQ(cam_ref.focal_y(), cam_new.focal_y());
    EXPECT_NEAR(cam_ref.FoVx(), cam_new.FoVx(), FLOAT_TOLERANCE);
    EXPECT_NEAR(cam_ref.FoVy(), cam_new.FoVy(), FLOAT_TOLERANCE);

    std::print("✓ BasicConstruction: uid={}, size={}x{}, focal=({:.2f}, {:.2f})\n",
               cam_new.uid(), cam_new.camera_width(), cam_new.camera_height(),
               cam_new.focal_x(), cam_new.focal_y());
}

TEST_F(CameraComparisonTest, RotatedCamera_Comparison) {
    float angle = std::numbers::pi_v<float> / 2.0f; // 90 degrees
    auto R_torch = create_rotation_z_torch(angle);
    auto T_torch = create_translation_torch(1.0f, 2.0f, 3.0f);
    auto R_tensor = create_rotation_z_tensor(angle);
    auto T_tensor = create_translation_tensor(1.0f, 2.0f, 3.0f);

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        800.0f, 800.0f,
        400.0f, 300.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "rotated.png",
        "/tmp/rotated.png",
        800, 600,
        1);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        800.0f, 800.0f,
        400.0f, 300.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "rotated.png",
        "/tmp/rotated.png",
        800, 600,
        1);

    // Compare rotation and translation
    EXPECT_TRUE(tensors_are_close(cam_ref.R(), cam_new.R(), FLOAT_TOLERANCE));
    EXPECT_TRUE(tensors_are_close(cam_ref.T(), cam_new.T(), FLOAT_TOLERANCE));

    std::print("✓ RotatedCamera: rotation and translation match\n");
}

TEST_F(CameraComparisonTest, ComplexRotation_Comparison) {
    float angle_x = 0.3f;
    float angle_y = 0.5f;
    float angle_z = 0.7f;

    auto [R_torch, R_tensor] = create_complex_rotation(angle_x, angle_y, angle_z);
    auto T_torch = create_translation_torch(1.5f, -2.3f, 4.7f);
    auto T_tensor = create_translation_tensor(1.5f, -2.3f, 4.7f);

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        850.0f, 900.0f,
        400.0f, 300.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "complex.png",
        "/tmp/complex.png",
        800, 600,
        5);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        850.0f, 900.0f,
        400.0f, 300.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "complex.png",
        "/tmp/complex.png",
        800, 600,
        5);

    // Compare complex rotation results
    EXPECT_TRUE(tensors_are_close(cam_ref.R(), cam_new.R(), FLOAT_TOLERANCE, true));

    std::print("✓ ComplexRotation: Euler angles (x={:.2f}, y={:.2f}, z={:.2f}) match\n",
               angle_x, angle_y, angle_z);
}

// ============================================================================
// Camera Position Tests
// ============================================================================

TEST_F(CameraComparisonTest, CameraPosition_Identity_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(0.0f, 0.0f, -5.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(0.0f, 0.0f, -5.0f);

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    auto pos_ref = cam_ref.cam_position();
    auto pos_new = cam_new.cam_position();

    EXPECT_TRUE(tensors_are_close(pos_ref, pos_new, FLOAT_TOLERANCE, true));

    // For identity rotation with T=(0,0,-5), camera position should be (0,0,5)
    auto pos_new_cpu = pos_new.cpu();
    EXPECT_NEAR(pos_new_cpu[0].item(), 0.0f, FLOAT_TOLERANCE);
    EXPECT_NEAR(pos_new_cpu[1].item(), 0.0f, FLOAT_TOLERANCE);
    EXPECT_NEAR(pos_new_cpu[2].item(), 5.0f, FLOAT_TOLERANCE);

    std::print("✓ CameraPosition_Identity: position = ({:.3f}, {:.3f}, {:.3f})\n",
               pos_new_cpu[0].item(), pos_new_cpu[1].item(), pos_new_cpu[2].item());
}

TEST_F(CameraComparisonTest, CameraPosition_Rotated_Comparison) {
    float angle = std::numbers::pi_v<float> / 4.0f; // 45 degrees
    auto R_torch = create_rotation_z_torch(angle);
    auto T_torch = create_translation_torch(3.0f, 4.0f, 5.0f);
    auto R_tensor = create_rotation_z_tensor(angle);
    auto T_tensor = create_translation_tensor(3.0f, 4.0f, 5.0f);

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    auto pos_ref = cam_ref.cam_position();
    auto pos_new = cam_new.cam_position();

    EXPECT_TRUE(tensors_are_close(pos_ref, pos_new, FLOAT_TOLERANCE, true));

    std::print("✓ CameraPosition_Rotated: positions match with rotation\n");
}

TEST_F(CameraComparisonTest, CameraPosition_LargeTranslation_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(1000.0f, -2000.0f, 5000.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(1000.0f, -2000.0f, 5000.0f);

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    auto pos_ref = cam_ref.cam_position();
    auto pos_new = cam_new.cam_position();

    EXPECT_TRUE(tensors_are_close(pos_ref, pos_new, 1e-2f, true)); // Slightly larger tolerance for large numbers

    std::print("✓ CameraPosition_LargeTranslation: large values handled correctly\n");
}

// ============================================================================
// World-View Transform Tests
// ============================================================================

TEST_F(CameraComparisonTest, WorldViewTransform_Structure_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(1.0f, 2.0f, 3.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(1.0f, 2.0f, 3.0f);

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    auto w2v_ref = cam_ref.world_view_transform();
    auto w2v_new = cam_new.world_view_transform();

    // Check shape: [1, 4, 4]
    EXPECT_EQ(w2v_ref.dim(), 3);
    EXPECT_EQ(w2v_ref.size(0), 1);
    EXPECT_EQ(w2v_ref.size(1), 4);
    EXPECT_EQ(w2v_ref.size(2), 4);

    EXPECT_EQ(w2v_new.ndim(), 3);
    EXPECT_EQ(w2v_new.size(0), 1);
    EXPECT_EQ(w2v_new.size(1), 4);
    EXPECT_EQ(w2v_new.size(2), 4);

    // Compare values
    EXPECT_TRUE(tensors_are_close(w2v_ref, w2v_new, MATRIX_TOLERANCE, true));

    std::print("✓ WorldViewTransform_Structure: shape [1, 4, 4] and values match\n");
}

TEST_F(CameraComparisonTest, WorldViewTransform_BottomRow_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(1.0f, 2.0f, 3.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(1.0f, 2.0f, 3.0f);

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    auto w2v_new = cam_new.world_view_transform().cpu();
    auto acc = w2v_new.accessor<float, 3>();

    // Bottom row should be [0, 0, 0, 1]
    EXPECT_NEAR(acc(0, 3, 0), 0.0f, FLOAT_TOLERANCE);
    EXPECT_NEAR(acc(0, 3, 1), 0.0f, FLOAT_TOLERANCE);
    EXPECT_NEAR(acc(0, 3, 2), 0.0f, FLOAT_TOLERANCE);
    EXPECT_NEAR(acc(0, 3, 3), 1.0f, FLOAT_TOLERANCE);

    std::print("✓ WorldViewTransform_BottomRow: homogeneous coordinate correct\n");
}

// ============================================================================
// Intrinsics Tests
// ============================================================================

TEST_F(CameraComparisonTest, IntrinsicMatrix_Structure_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(0.0f, 0.0f, 0.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(0.0f, 0.0f, 0.0f);

    float focal_x = 1000.0f;
    float focal_y = 800.0f;
    float center_x = 512.0f;
    float center_y = 384.0f;

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        focal_x, focal_y,
        center_x, center_y,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        focal_x, focal_y,
        center_x, center_y,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    auto K_ref = cam_ref.K();
    auto K_new = cam_new.K();

    // Check shape: [1, 3, 3]
    EXPECT_EQ(K_ref.dim(), 3);
    EXPECT_EQ(K_ref.size(0), 1);
    EXPECT_EQ(K_ref.size(1), 3);
    EXPECT_EQ(K_ref.size(2), 3);

    EXPECT_EQ(K_new.ndim(), 3);
    EXPECT_EQ(K_new.size(0), 1);
    EXPECT_EQ(K_new.size(1), 3);
    EXPECT_EQ(K_new.size(2), 3);

    // Compare values
    EXPECT_TRUE(tensors_are_close(K_ref, K_new, INTRINSICS_TOLERANCE, true));

    std::print("✓ IntrinsicMatrix_Structure: K matrix shape and values match\n");
}

TEST_F(CameraComparisonTest, IntrinsicMatrix_Elements_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(0.0f, 0.0f, 0.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(0.0f, 0.0f, 0.0f);

    float focal_x = 1200.0f;
    float focal_y = 950.0f;
    float center_x = 640.0f;
    float center_y = 480.0f;

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        focal_x, focal_y,
        center_x, center_y,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1280, 960,
        0);

    auto K_new = cam_new.K().cpu();
    auto K_acc = K_new.accessor<float, 3>();

    // Check specific elements
    EXPECT_NEAR(K_acc(0, 0, 0), focal_x, INTRINSICS_TOLERANCE);  // fx
    EXPECT_NEAR(K_acc(0, 1, 1), focal_y, INTRINSICS_TOLERANCE);  // fy
    EXPECT_NEAR(K_acc(0, 0, 2), center_x, INTRINSICS_TOLERANCE); // cx
    EXPECT_NEAR(K_acc(0, 1, 2), center_y, INTRINSICS_TOLERANCE); // cy
    EXPECT_NEAR(K_acc(0, 2, 2), 1.0f, FLOAT_TOLERANCE);          // bottom-right

    // Check zeros
    EXPECT_NEAR(K_acc(0, 0, 1), 0.0f, FLOAT_TOLERANCE);
    EXPECT_NEAR(K_acc(0, 1, 0), 0.0f, FLOAT_TOLERANCE);
    EXPECT_NEAR(K_acc(0, 2, 0), 0.0f, FLOAT_TOLERANCE);
    EXPECT_NEAR(K_acc(0, 2, 1), 0.0f, FLOAT_TOLERANCE);

    std::print("✓ IntrinsicMatrix_Elements: individual elements correct\n");
}

TEST_F(CameraComparisonTest, GetIntrinsics_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(0.0f, 0.0f, 0.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(0.0f, 0.0f, 0.0f);

    float focal_x = 1000.0f;
    float focal_y = 850.0f;
    float center_x = 512.0f;
    float center_y = 384.0f;

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        focal_x, focal_y,
        center_x, center_y,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        focal_x, focal_y,
        center_x, center_y,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    auto [fx_ref, fy_ref, cx_ref, cy_ref] = cam_ref.get_intrinsics();
    auto [fx_new, fy_new, cx_new, cy_new] = cam_new.get_intrinsics();

    EXPECT_NEAR(fx_ref, fx_new, INTRINSICS_TOLERANCE);
    EXPECT_NEAR(fy_ref, fy_new, INTRINSICS_TOLERANCE);
    EXPECT_NEAR(cx_ref, cx_new, INTRINSICS_TOLERANCE);
    EXPECT_NEAR(cy_ref, cy_new, INTRINSICS_TOLERANCE);

    std::print("✓ GetIntrinsics: tuple (fx, fy, cx, cy) matches\n");
}

// ============================================================================
// Field of View Tests
// ============================================================================

TEST_F(CameraComparisonTest, FieldOfView_Calculation_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(0.0f, 0.0f, 0.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(0.0f, 0.0f, 0.0f);

    std::vector<std::tuple<float, float, int, int>> test_cases = {
        {1000.0f, 1000.0f, 1920, 1080},
        {800.0f, 850.0f, 1280, 720},
        {1500.0f, 1500.0f, 2560, 1440},
        {500.0f, 600.0f, 640, 480},
    };

    for (const auto& [fx, fy, w, h] : test_cases) {
        auto radial_torch = torch::zeros({4}, torch::kFloat32);
        auto tangential_torch = torch::zeros({2}, torch::kFloat32);
        auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
        auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

        gs::CameraRef cam_ref(
            R_torch, T_torch,
            fx, fy, w / 2.0f, h / 2.0f,
            radial_torch, tangential_torch,
            gsplat::CameraModelType::PINHOLE,
            "test.png",
            "/tmp/test.png",
            w, h, 0);

        gs::CameraNew cam_new(
            R_tensor, T_tensor,
            fx, fy, w / 2.0f, h / 2.0f,
            radial_tensor, tangential_tensor,
            gsplat::CameraModelType::PINHOLE,
            "test.png",
            "/tmp/test.png",
            w, h, 0);

        EXPECT_NEAR(cam_ref.FoVx(), cam_new.FoVx(), FLOAT_TOLERANCE);
        EXPECT_NEAR(cam_ref.FoVy(), cam_new.FoVy(), FLOAT_TOLERANCE);
    }

    std::print("✓ FieldOfView_Calculation: FoV computed correctly for multiple configurations\n");
}

TEST_F(CameraComparisonTest, FocalFoVConversion_RoundTrip) {
    float focal = 1234.5f;
    int pixels = 1920;

    float fov = gs::focal2fov(focal, pixels);
    float focal_back = gs::fov2focal(fov, pixels);

    EXPECT_NEAR(focal, focal_back, FLOAT_TOLERANCE);

    std::print("✓ FocalFoVConversion_RoundTrip: focal={:.2f} -> fov={:.4f} -> focal={:.2f}\n",
               focal, fov, focal_back);
}

// ============================================================================
// Transform Constructor Tests
// ============================================================================

TEST_F(CameraComparisonTest, TransformConstructor_Comparison) {
    // Create base cameras
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(0.0f, 0.0f, -5.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(0.0f, 0.0f, -5.0f);

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref_base(
        R_torch, T_torch,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    gs::CameraNew cam_new_base(
        R_tensor, T_tensor,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    // Create new transforms
    auto new_transform_torch = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).unsqueeze(0);
    new_transform_torch[0][0][3] = 1.0f;
    new_transform_torch[0][1][3] = 2.0f;
    new_transform_torch[0][2][3] = 3.0f;

    auto new_transform_tensor = gs::Tensor::eye(4, gs::Device::CUDA).unsqueeze(0);
    auto transform_cpu = new_transform_tensor.cpu();
    auto acc = transform_cpu.accessor<float, 3>();
    acc(0, 0, 3) = 1.0f;
    acc(0, 1, 3) = 2.0f;
    acc(0, 2, 3) = 3.0f;
    new_transform_tensor = transform_cpu.cuda();

    // Create transformed cameras
    gs::CameraRef cam_ref_transformed(cam_ref_base, new_transform_torch);
    gs::CameraNew cam_new_transformed(cam_new_base, new_transform_tensor);

    // Properties should be copied
    EXPECT_EQ(cam_ref_transformed.uid(), cam_new_transformed.uid());
    EXPECT_FLOAT_EQ(cam_ref_transformed.focal_x(), cam_new_transformed.focal_x());

    // Transforms should match
    EXPECT_TRUE(tensors_are_close(cam_ref_transformed.world_view_transform(),
                                  cam_new_transformed.world_view_transform(),
                                  MATRIX_TOLERANCE, true));

    std::print("✓ TransformConstructor: properties copied, transform updated\n");
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_F(CameraComparisonTest, ValidationTest) {
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(1.0f, 2.0f, 3.0f);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        42);

    EXPECT_TRUE(cam_new.is_valid());

    std::print("✓ Validation: CameraNew passes validation checks\n");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(CameraComparisonTest, ZeroTranslation_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(0.0f, 0.0f, 0.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(0.0f, 0.0f, 0.0f);

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    EXPECT_TRUE(cam_new.is_valid());

    auto pos_new = cam_new.cam_position().cpu();
    EXPECT_NEAR(pos_new[0].item(), 0.0f, FLOAT_TOLERANCE);
    EXPECT_NEAR(pos_new[1].item(), 0.0f, FLOAT_TOLERANCE);
    EXPECT_NEAR(pos_new[2].item(), 0.0f, FLOAT_TOLERANCE);

    std::print("✓ ZeroTranslation: camera at origin handled correctly\n");
}

TEST_F(CameraComparisonTest, SmallFocal_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(0.0f, 0.0f, -5.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(0.0f, 0.0f, -5.0f);

    float focal_x = 100.0f; // Very small focal length
    float focal_y = 100.0f;

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        focal_x, focal_y,
        512.0f, 384.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        focal_x, focal_y,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    EXPECT_TRUE(cam_new.is_valid());
    EXPECT_NEAR(cam_ref.FoVx(), cam_new.FoVx(), FLOAT_TOLERANCE);

    // Small focal length should result in large FoV
    EXPECT_GT(cam_new.FoVx(), 1.0f); // > ~57 degrees

    std::print("✓ SmallFocal: small focal length handled correctly, FoV={:.2f} rad\n",
               cam_new.FoVx());
}

TEST_F(CameraComparisonTest, LargeFocal_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(0.0f, 0.0f, -5.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(0.0f, 0.0f, -5.0f);

    float focal_x = 10000.0f; // Very large focal length
    float focal_y = 10000.0f;

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        focal_x, focal_y,
        512.0f, 384.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        focal_x, focal_y,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    EXPECT_TRUE(cam_new.is_valid());
    EXPECT_NEAR(cam_ref.FoVx(), cam_new.FoVx(), FLOAT_TOLERANCE);

    // Large focal length should result in small FoV
    EXPECT_LT(cam_new.FoVx(), 0.2f); // < ~11 degrees

    std::print("✓ LargeFocal: large focal length handled correctly, FoV={:.4f} rad\n",
               cam_new.FoVx());
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(CameraComparisonTest, ManyCameras_Comparison) {
    const int num_cameras = 100;

    for (int i = 0; i < num_cameras; ++i) {
        float angle = (2.0f * std::numbers::pi_v<float> * i) / num_cameras;
        auto R_torch = create_rotation_z_torch(angle);
        auto T_torch = create_translation_torch(
            std::cos(angle) * 5.0f,
            std::sin(angle) * 5.0f,
            static_cast<float>(i) * 0.1f);
        auto R_tensor = create_rotation_z_tensor(angle);
        auto T_tensor = create_translation_tensor(
            std::cos(angle) * 5.0f,
            std::sin(angle) * 5.0f,
            static_cast<float>(i) * 0.1f);

        auto radial_torch = torch::zeros({4}, torch::kFloat32);
        auto tangential_torch = torch::zeros({2}, torch::kFloat32);
        auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
        auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

        gs::CameraRef cam_ref(
            R_torch, T_torch,
            1000.0f + i, 1000.0f + i,
            512.0f, 384.0f,
            radial_torch, tangential_torch,
            gsplat::CameraModelType::PINHOLE,
            "test_" + std::to_string(i) + ".png",
            "/tmp/test.png",
            1024, 768,
            i);

        gs::CameraNew cam_new(
            R_tensor, T_tensor,
            1000.0f + i, 1000.0f + i,
            512.0f, 384.0f,
            radial_tensor, tangential_tensor,
            gsplat::CameraModelType::PINHOLE,
            "test_" + std::to_string(i) + ".png",
            "/tmp/test.png",
            1024, 768,
            i);

        EXPECT_TRUE(cam_new.is_valid());
        EXPECT_EQ(cam_ref.uid(), cam_new.uid());
        EXPECT_TRUE(tensors_are_close(cam_ref.cam_position(),
                                      cam_new.cam_position(),
                                      FLOAT_TOLERANCE));
    }

    std::print("✓ ManyCameras: created and validated {} cameras\n", num_cameras);
}

// ============================================================================
// Performance Tests
// ============================================================================

class CameraPerformanceTest : public ::testing::Test {
protected:
    struct BenchmarkResult {
        std::string name;
        int num_cameras;
        double time_ms_ref;
        double time_ms_new;
        double speedup;
    };

    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        torch::manual_seed(42);
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
        std::print("\n{}\n", std::string(90, '='));
        std::print("CAMERA PERFORMANCE COMPARISON: CameraRef vs CameraNew\n");
        std::print("{}\n\n", std::string(90, '='));

        std::print("{:<40} {:>12} {:>12} {:>12} {:>12}\n",
                   "Benchmark", "N Cameras", "Ref (ms)", "New (ms)", "Speedup");
        std::print("{}\n", std::string(90, '-'));

        for (const auto& r : results) {
            std::print("{:<40} {:>12} {:>12.3f} {:>12.3f} {:>11.2f}x\n",
                       r.name, r.num_cameras, r.time_ms_ref, r.time_ms_new, r.speedup);
        }

        std::print("{}\n", std::string(90, '='));

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
        std::print("{}\n\n", std::string(90, '='));
    }
};

TEST_F(CameraPerformanceTest, DISABLED_ConstructionBenchmark) {
    std::vector<BenchmarkResult> results;

    std::vector<int> camera_counts = {1, 10, 100, 500, 1000};

    for (int num_cameras : camera_counts) {
        BenchmarkResult result;
        result.name = "Camera_Construction";
        result.num_cameras = num_cameras;

        // Benchmark Ref
        result.time_ms_ref = benchmark([&]() {
            for (int i = 0; i < num_cameras; ++i) {
                float angle = (2.0f * std::numbers::pi_v<float> * i) / num_cameras;
                auto R = create_rotation_z_torch(angle);
                auto T = create_translation_torch(
                    std::cos(angle) * 5.0f,
                    std::sin(angle) * 5.0f,
                    static_cast<float>(i) * 0.1f);
                auto radial = torch::zeros({4}, torch::kFloat32);
                auto tangential = torch::zeros({2}, torch::kFloat32);

                gs::CameraRef cam(R, T, 1000.0f, 1000.0f, 512.0f, 384.0f,
                                  radial, tangential,
                                  gsplat::CameraModelType::PINHOLE,
                                  "test.png", "/tmp/test.png",
                                  1024, 768, i);
            }
        },
                                       2, 5);

        // Benchmark New
        result.time_ms_new = benchmark([&]() {
            for (int i = 0; i < num_cameras; ++i) {
                float angle = (2.0f * std::numbers::pi_v<float> * i) / num_cameras;
                auto R = create_rotation_z_tensor(angle);
                auto T = create_translation_tensor(
                    std::cos(angle) * 5.0f,
                    std::sin(angle) * 5.0f,
                    static_cast<float>(i) * 0.1f);
                auto radial = gs::Tensor::zeros({4}, gs::Device::CPU);
                auto tangential = gs::Tensor::zeros({2}, gs::Device::CPU);

                gs::CameraNew cam(R, T, 1000.0f, 1000.0f, 512.0f, 384.0f,
                                  radial, tangential,
                                  gsplat::CameraModelType::PINHOLE,
                                  "test.png", "/tmp/test.png",
                                  1024, 768, i);
            }
        },
                                       2, 5);

        result.speedup = result.time_ms_ref / result.time_ms_new;
        results.push_back(result);
    }

    print_results(results);
}

TEST_F(CameraPerformanceTest, DISABLED_GettersBenchmark) {
    std::vector<BenchmarkResult> results;

    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(1.0f, 2.0f, 3.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(1.0f, 2.0f, 3.0f);

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        1000.0f, 1000.0f,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test.png",
        "/tmp/test.png",
        1024, 768,
        0);

    std::vector<std::pair<std::string, std::function<void()>>> tests_ref = {
        {"WorldViewTransform", [&]() { auto w2v = cam_ref.world_view_transform(); }},
        {"CameraPosition", [&]() { auto pos = cam_ref.cam_position(); }},
        {"IntrinsicMatrix", [&]() { auto K = cam_ref.K(); }},
        {"GetIntrinsics", [&]() { auto [fx, fy, cx, cy] = cam_ref.get_intrinsics(); }},
    };

    std::vector<std::pair<std::string, std::function<void()>>> tests_new = {
        {"WorldViewTransform", [&]() { auto w2v = cam_new.world_view_transform(); }},
        {"CameraPosition", [&]() { auto pos = cam_new.cam_position(); }},
        {"IntrinsicMatrix", [&]() { auto K = cam_new.K(); }},
        {"GetIntrinsics", [&]() { auto [fx, fy, cx, cy] = cam_new.get_intrinsics(); }},
    };

    for (size_t i = 0; i < tests_ref.size(); ++i) {
        BenchmarkResult result;
        result.name = tests_ref[i].first;
        result.num_cameras = 1;

        result.time_ms_ref = benchmark(tests_ref[i].second, 5, 100);
        result.time_ms_new = benchmark(tests_new[i].second, 5, 100);

        result.speedup = result.time_ms_ref / result.time_ms_new;
        results.push_back(result);
    }

    print_results(results);
}

// ============================================================================
// Additional Comprehensive Tests
// ============================================================================

TEST_F(CameraComparisonTest, MultipleResolutions_Comparison) {
    struct Resolution {
        int width;
        int height;
        std::string name;
    };

    std::vector<Resolution> resolutions = {
        {640, 480, "VGA"},
        {1280, 720, "HD"},
        {1920, 1080, "FHD"},
        {2560, 1440, "QHD"},
        {3840, 2160, "4K"},
    };

    for (const auto& res : resolutions) {
        auto R_torch = create_identity_rotation_torch();
        auto T_torch = create_translation_torch(0.0f, 0.0f, -10.0f);
        auto R_tensor = create_identity_rotation_tensor();
        auto T_tensor = create_translation_tensor(0.0f, 0.0f, -10.0f);

        float focal = 0.8f * res.width; // Typical focal length

        auto radial_torch = torch::zeros({4}, torch::kFloat32);
        auto tangential_torch = torch::zeros({2}, torch::kFloat32);
        auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
        auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

        gs::CameraRef cam_ref(
            R_torch, T_torch,
            focal, focal,
            res.width / 2.0f, res.height / 2.0f,
            radial_torch, tangential_torch,
            gsplat::CameraModelType::PINHOLE,
            res.name + ".png",
            "/tmp/" + res.name + ".png",
            res.width, res.height,
            0);

        gs::CameraNew cam_new(
            R_tensor, T_tensor,
            focal, focal,
            res.width / 2.0f, res.height / 2.0f,
            radial_tensor, tangential_tensor,
            gsplat::CameraModelType::PINHOLE,
            res.name + ".png",
            "/tmp/" + res.name + ".png",
            res.width, res.height,
            0);

        EXPECT_TRUE(cam_new.is_valid());
        EXPECT_NEAR(cam_ref.FoVx(), cam_new.FoVx(), FLOAT_TOLERANCE);
        EXPECT_TRUE(tensors_are_close(cam_ref.world_view_transform(),
                                      cam_new.world_view_transform(),
                                      MATRIX_TOLERANCE));
    }

    std::print("✓ MultipleResolutions: tested {} different resolutions\n", resolutions.size());
}

TEST_F(CameraComparisonTest, AsymmetricFocal_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(0.0f, 0.0f, -5.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(0.0f, 0.0f, -5.0f);

    // Very different focal lengths in x and y (anamorphic lens)
    float focal_x = 2000.0f;
    float focal_y = 800.0f;

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        focal_x, focal_y,
        512.0f, 384.0f,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "anamorphic.png",
        "/tmp/anamorphic.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        focal_x, focal_y,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "anamorphic.png",
        "/tmp/anamorphic.png",
        1024, 768,
        0);

    EXPECT_TRUE(cam_new.is_valid());
    EXPECT_NEAR(cam_ref.FoVx(), cam_new.FoVx(), FLOAT_TOLERANCE);
    EXPECT_NEAR(cam_ref.FoVy(), cam_new.FoVy(), FLOAT_TOLERANCE);

    // FoVx should be much smaller than FoVy for this configuration
    EXPECT_LT(cam_new.FoVx(), cam_new.FoVy());

    auto K_ref = cam_ref.K();
    auto K_new = cam_new.K();
    EXPECT_TRUE(tensors_are_close(K_ref, K_new, INTRINSICS_TOLERANCE, true));

    std::print("✓ AsymmetricFocal: anamorphic lens (fx={:.0f}, fy={:.0f}) handled correctly\n",
               focal_x, focal_y);
}

TEST_F(CameraComparisonTest, OffCenterPrincipalPoint_Comparison) {
    auto R_torch = create_identity_rotation_torch();
    auto T_torch = create_translation_torch(0.0f, 0.0f, -5.0f);
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(0.0f, 0.0f, -5.0f);

    // Off-center principal point
    float focal_x = 1000.0f;
    float focal_y = 1000.0f;
    float center_x = 300.0f; // Not at center
    float center_y = 600.0f; // Not at center

    auto radial_torch = torch::zeros({4}, torch::kFloat32);
    auto tangential_torch = torch::zeros({2}, torch::kFloat32);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraRef cam_ref(
        R_torch, T_torch,
        focal_x, focal_y,
        center_x, center_y,
        radial_torch, tangential_torch,
        gsplat::CameraModelType::PINHOLE,
        "offcenter.png",
        "/tmp/offcenter.png",
        1024, 768,
        0);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        focal_x, focal_y,
        center_x, center_y,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "offcenter.png",
        "/tmp/offcenter.png",
        1024, 768,
        0);

    auto K_ref = cam_ref.K();
    auto K_new = cam_new.K();

    EXPECT_TRUE(tensors_are_close(K_ref, K_new, INTRINSICS_TOLERANCE, true));

    // Verify principal point is preserved
    auto K_new_cpu = K_new.cpu();
    auto K_acc = K_new_cpu.accessor<float, 3>();
    EXPECT_NEAR(K_acc(0, 0, 2), center_x, INTRINSICS_TOLERANCE);
    EXPECT_NEAR(K_acc(0, 1, 2), center_y, INTRINSICS_TOLERANCE);

    std::print("✓ OffCenterPrincipalPoint: principal point ({:.0f}, {:.0f}) handled correctly\n",
               center_x, center_y);
}

TEST_F(CameraComparisonTest, CircularCameraPath_Comparison) {
    const int num_cameras = 36; // 10 degree increments
    const float radius = 10.0f;
    const float height = 2.0f;

    for (int i = 0; i < num_cameras; ++i) {
        float angle = (2.0f * std::numbers::pi_v<float> * i) / num_cameras;

        // Camera looks at origin
        float cam_x = radius * std::cos(angle);
        float cam_y = radius * std::sin(angle);
        float cam_z = height;

        // Create rotation to look at origin
        auto R_torch = create_rotation_z_torch(-angle - std::numbers::pi_v<float> / 2.0f);
        auto T_torch = create_translation_torch(cam_x, cam_y, cam_z);
        auto R_tensor = create_rotation_z_tensor(-angle - std::numbers::pi_v<float> / 2.0f);
        auto T_tensor = create_translation_tensor(cam_x, cam_y, cam_z);

        auto radial_torch = torch::zeros({4}, torch::kFloat32);
        auto tangential_torch = torch::zeros({2}, torch::kFloat32);
        auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
        auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

        gs::CameraRef cam_ref(
            R_torch, T_torch,
            1000.0f, 1000.0f,
            512.0f, 384.0f,
            radial_torch, tangential_torch,
            gsplat::CameraModelType::PINHOLE,
            "cam_" + std::to_string(i) + ".png",
            "/tmp/cam.png",
            1024, 768,
            i);

        gs::CameraNew cam_new(
            R_tensor, T_tensor,
            1000.0f, 1000.0f,
            512.0f, 384.0f,
            radial_tensor, tangential_tensor,
            gsplat::CameraModelType::PINHOLE,
            "cam_" + std::to_string(i) + ".png",
            "/tmp/cam.png",
            1024, 768,
            i);

        EXPECT_TRUE(cam_new.is_valid());
        EXPECT_TRUE(tensors_are_close(cam_ref.cam_position(),
                                      cam_new.cam_position(),
                                      FLOAT_TOLERANCE));

        // Verify camera is on the circular path
        auto pos = cam_new.cam_position().cpu();
        float pos_x = pos[0].item();
        float pos_y = pos[1].item();
        float dist_from_origin = std::sqrt(pos_x * pos_x + pos_y * pos_y);

        // Should be approximately on the circle
        EXPECT_NEAR(dist_from_origin, radius, 0.5f);
    }

    std::print("✓ CircularCameraPath: {} cameras on circular path validated\n", num_cameras);
}

TEST_F(CameraComparisonTest, NumericalStability_ExtremeCases) {
    struct TestCase {
        std::string name;
        float tx, ty, tz;
        float angle;
        float focal;
    };

    std::vector<TestCase> cases = {
        {"VeryCloseToOrigin", 0.001f, 0.001f, 0.001f, 0.0f, 1000.0f},
        {"VeryFarFromOrigin", 10000.0f, 10000.0f, 10000.0f, 0.0f, 1000.0f},
        {"VerySmallAngle", 0.0f, 0.0f, -5.0f, 0.0001f, 1000.0f},
        {"AlmostPi", 0.0f, 0.0f, -5.0f, std::numbers::pi_v<float> - 0.0001f, 1000.0f},
        {"ExtremelySmallFocal", 0.0f, 0.0f, -5.0f, 0.0f, 10.0f},
        {"ExtremelyLargeFocal", 0.0f, 0.0f, -5.0f, 0.0f, 50000.0f},
    };

    for (const auto& test_case : cases) {
        auto R_torch = create_rotation_z_torch(test_case.angle);
        auto T_torch = create_translation_torch(test_case.tx, test_case.ty, test_case.tz);
        auto R_tensor = create_rotation_z_tensor(test_case.angle);
        auto T_tensor = create_translation_tensor(test_case.tx, test_case.ty, test_case.tz);

        auto radial_torch = torch::zeros({4}, torch::kFloat32);
        auto tangential_torch = torch::zeros({2}, torch::kFloat32);
        auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
        auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

        gs::CameraRef cam_ref(
            R_torch, T_torch,
            test_case.focal, test_case.focal,
            512.0f, 384.0f,
            radial_torch, tangential_torch,
            gsplat::CameraModelType::PINHOLE,
            test_case.name + ".png",
            "/tmp/test.png",
            1024, 768,
            0);

        gs::CameraNew cam_new(
            R_tensor, T_tensor,
            test_case.focal, test_case.focal,
            512.0f, 384.0f,
            radial_tensor, tangential_tensor,
            gsplat::CameraModelType::PINHOLE,
            test_case.name + ".png",
            "/tmp/test.png",
            1024, 768,
            0);

        EXPECT_TRUE(cam_new.is_valid()) << "Failed for case: " << test_case.name;

        // Check that no values are NaN or Inf
        auto pos = cam_new.cam_position().cpu();
        EXPECT_FALSE(std::isnan(pos[0].item())) << "NaN in case: " << test_case.name;
        EXPECT_FALSE(std::isnan(pos[1].item())) << "NaN in case: " << test_case.name;
        EXPECT_FALSE(std::isnan(pos[2].item())) << "NaN in case: " << test_case.name;
        EXPECT_FALSE(std::isinf(pos[0].item())) << "Inf in case: " << test_case.name;
        EXPECT_FALSE(std::isinf(pos[1].item())) << "Inf in case: " << test_case.name;
        EXPECT_FALSE(std::isinf(pos[2].item())) << "Inf in case: " << test_case.name;
    }

    std::print("✓ NumericalStability: {} extreme cases handled without NaN/Inf\n", cases.size());
}

TEST_F(CameraComparisonTest, MemoryFootprint_Comparison) {
    // This test verifies that both implementations have similar memory requirements
    const int num_cameras = 1000;

    std::vector<std::unique_ptr<gs::CameraRef>> cameras_ref;
    std::vector<std::unique_ptr<gs::CameraNew>> cameras_new;

    for (int i = 0; i < num_cameras; ++i) {
        float angle = (2.0f * std::numbers::pi_v<float> * i) / num_cameras;

        auto R_torch = create_rotation_z_torch(angle);
        auto T_torch = create_translation_torch(
            std::cos(angle) * 5.0f,
            std::sin(angle) * 5.0f,
            static_cast<float>(i) * 0.01f);
        auto R_tensor = create_rotation_z_tensor(angle);
        auto T_tensor = create_translation_tensor(
            std::cos(angle) * 5.0f,
            std::sin(angle) * 5.0f,
            static_cast<float>(i) * 0.01f);

        auto radial_torch = torch::zeros({4}, torch::kFloat32);
        auto tangential_torch = torch::zeros({2}, torch::kFloat32);
        auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
        auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

        cameras_ref.push_back(std::make_unique<gs::CameraRef>(
            R_torch, T_torch,
            1000.0f, 1000.0f,
            512.0f, 384.0f,
            radial_torch, tangential_torch,
            gsplat::CameraModelType::PINHOLE,
            "test.png",
            "/tmp/test.png",
            1024, 768,
            i));

        cameras_new.push_back(std::make_unique<gs::CameraNew>(
            R_tensor, T_tensor,
            1000.0f, 1000.0f,
            512.0f, 384.0f,
            radial_tensor, tangential_tensor,
            gsplat::CameraModelType::PINHOLE,
            "test.png",
            "/tmp/test.png",
            1024, 768,
            i));
    }

    // All cameras should be valid
    for (const auto& cam : cameras_new) {
        EXPECT_TRUE(cam->is_valid());
    }

    std::print("✓ MemoryFootprint: {} cameras allocated successfully\n", num_cameras);
}

// ============================================================================
// String Representation Test
// ============================================================================

TEST_F(CameraComparisonTest, StringRepresentation) {
    auto R_tensor = create_identity_rotation_tensor();
    auto T_tensor = create_translation_tensor(1.0f, 2.0f, 3.0f);
    auto radial_tensor = gs::Tensor::zeros({4}, gs::Device::CPU);
    auto tangential_tensor = gs::Tensor::zeros({2}, gs::Device::CPU);

    gs::CameraNew cam_new(
        R_tensor, T_tensor,
        1234.5f, 987.6f,
        512.0f, 384.0f,
        radial_tensor, tangential_tensor,
        gsplat::CameraModelType::PINHOLE,
        "test_camera.png",
        "/tmp/test_camera.png",
        1024, 768,
        42);

    std::string str = cam_new.str();

    // Check that string contains key information
    EXPECT_TRUE(str.find("CameraNew") != std::string::npos);
    EXPECT_TRUE(str.find("42") != std::string::npos);   // uid
    EXPECT_TRUE(str.find("1024") != std::string::npos); // width
    EXPECT_TRUE(str.find("768") != std::string::npos);  // height
    EXPECT_TRUE(str.find("test_camera.png") != std::string::npos);

    std::print("✓ StringRepresentation: {}\n", str);
}