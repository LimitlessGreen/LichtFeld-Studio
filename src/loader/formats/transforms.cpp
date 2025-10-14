/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "transforms.hpp"
#include "core/image_io_new.hpp"
#include "core/logger.hpp"
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <random>

namespace gs::loader {

    // Constants for random point cloud generation
    constexpr int DEFAULT_NUM_INIT_GAUSSIAN = 10000;
    constexpr uint64_t DEFAULT_RANDOM_SEED = 8128;

    float fov_deg_to_focal_length(int resolution, float fov_deg) {
        return 0.5f * static_cast<float>(resolution) / std::tan(0.5f * fov_deg * std::numbers::pi_v<float> / 180.0f);
    }

    float fov_rad_to_focal_length(int resolution, float fov_rad) {
        return 0.5f * static_cast<float>(resolution) / std::tan(0.5f * fov_rad);
    }

    // Create a 4x4 rotation matrix around Y-axis
    Tensor createYRotationMatrix(float angle_radians) {
        float cos_angle = std::cos(angle_radians);
        float sin_angle = std::sin(angle_radians);

        // Rotation matrix around Y-axis:
        // [cos(θ)   0   sin(θ)  0]
        // [  0      1     0     0]
        // [-sin(θ)  0   cos(θ)  0]
        // [  0      0     0     1]

        std::vector<float> data = {
            cos_angle, 0.0f, sin_angle, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            -sin_angle, 0.0f, cos_angle, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f};

        return Tensor::from_vector(data, TensorShape{4, 4}, Device::CPU);
    }

    // Matrix inversion for 4x4 matrices (manual implementation)
    Tensor inverse_4x4(const Tensor& mat) {
        if (mat.shape() != TensorShape{4, 4}) {
            LOG_ERROR("inverse_4x4 requires 4x4 matrix");
            throw std::runtime_error("Invalid matrix dimensions for inversion");
        }

        auto mat_cpu = mat.cpu();
        const float* m = mat_cpu.ptr<float>();

        // Compute inverse using adjugate method
        std::vector<float> inv(16);

        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

        float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

        if (std::abs(det) < 1e-8f) {
            LOG_ERROR("Matrix is singular, cannot invert");
            throw std::runtime_error("Singular matrix");
        }

        float invDet = 1.0f / det;
        for (int i = 0; i < 16; i++) {
            inv[i] *= invDet;
        }

        return Tensor::from_vector(inv, TensorShape{4, 4}, Device::CPU);
    }

    // Matrix multiplication for 4x4 matrices
    Tensor matmul_4x4(const Tensor& a, const Tensor& b) {
        if (a.shape() != TensorShape{4, 4} || b.shape() != TensorShape{4, 4}) {
            LOG_ERROR("matmul_4x4 requires 4x4 matrices");
            throw std::runtime_error("Invalid matrix dimensions for multiplication");
        }

        auto a_cpu = a.cpu();
        auto b_cpu = b.cpu();

        const float* am = a_cpu.ptr<float>();
        const float* bm = b_cpu.ptr<float>();

        std::vector<float> result(16, 0.0f);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 4; ++k) {
                    result[i * 4 + j] += am[i * 4 + k] * bm[k * 4 + j];
                }
            }
        }

        return Tensor::from_vector(result, TensorShape{4, 4}, Device::CPU);
    }

    std::filesystem::path GetTransformImagePath(const std::filesystem::path& dir_path, const nlohmann::json& frame) {
        auto image_path = dir_path / frame["file_path"].get<std::string>();
        auto image_path_png = std::filesystem::path(image_path.string() + ".png");
        if (std::filesystem::exists(image_path_png)) {
            image_path = image_path_png;
            LOG_TRACE("Using PNG extension for image: {}", image_path.string());
        }
        return image_path;
    }

    std::tuple<std::vector<std::shared_ptr<CameraNew>>, Tensor>
    read_transforms_cameras_and_images(const std::filesystem::path& transPath) {

        LOG_TIMER_TRACE("Read transforms file");

        std::filesystem::path transformsFile = transPath;
        if (std::filesystem::is_directory(transPath)) {
            if (std::filesystem::is_regular_file(transPath / "transforms_train.json")) {
                transformsFile = transPath / "transforms_train.json";
            } else if (std::filesystem::is_regular_file(transPath / "transforms.json")) {
                transformsFile = transPath / "transforms.json";
            } else {
                LOG_ERROR("Could not find transforms file in: {}", transPath.string());
                throw std::runtime_error("Could not find transforms_train.json or transforms.json");
            }
        }

        if (!std::filesystem::is_regular_file(transformsFile)) {
            LOG_ERROR("Not a valid file: {}", transformsFile.string());
            throw std::runtime_error("Not a valid file");
        }

        LOG_DEBUG("Reading transforms from: {}", transformsFile.string());
        std::ifstream trans_file{transformsFile.string()};

        std::filesystem::path dir_path = transformsFile.parent_path();

        nlohmann::json transforms = nlohmann::json::parse(trans_file, nullptr, true, true);

        int w = -1, h = -1;
        if (!transforms.contains("w") || !transforms.contains("h")) {
            try {
                LOG_DEBUG("Width/height not in transforms.json, reading from first image");
                auto first_frame_img_path = GetTransformImagePath(dir_path, transforms["frames"][0]);
                auto result = image_io::get_image_info(first_frame_img_path);

                w = std::get<0>(result);
                h = std::get<1>(result);

                LOG_DEBUG("Got image dimensions: {}x{}", w, h);
            } catch (const std::exception& e) {
                LOG_ERROR("Error reading image dimensions: {}", e.what());
                throw;
            }
        } else {
            w = transforms["w"].get<int>();
            h = transforms["h"].get<int>();
        }

        float fl_x = -1, fl_y = -1;
        if (transforms.contains("fl_x")) {
            fl_x = transforms["fl_x"].get<float>();
        } else if (transforms.contains("camera_angle_x")) {
            fl_x = fov_rad_to_focal_length(w, transforms["camera_angle_x"].get<float>());
        }

        if (transforms.contains("fl_y")) {
            fl_y = transforms["fl_y"].get<float>();
        } else if (transforms.contains("camera_angle_y")) {
            fl_y = fov_rad_to_focal_length(h, transforms["camera_angle_y"].get<float>());
        } else {
            if (w != h) {
                LOG_ERROR("No camera_angle_y but w!=h: {}!={}", w, h);
                throw std::runtime_error("No camera_angle_y with non-square images");
            }
            fl_y = fl_x;
        }

        float cx = transforms.contains("cx") ? transforms["cx"].get<float>() : 0.5f * w;
        float cy = transforms.contains("cy") ? transforms["cy"].get<float>() : 0.5f * h;

        // Check for distortion (not supported)
        if (transforms.contains("k1") || transforms.contains("k2") || transforms.contains("p1") || transforms.contains("p2")) {
            float k1 = transforms.value("k1", 0.0f);
            float k2 = transforms.value("k2", 0.0f);
            float p1 = transforms.value("p1", 0.0f);
            float p2 = transforms.value("p2", 0.0f);

            if (k1 != 0.0f || k2 != 0.0f || p1 != 0.0f || p2 != 0.0f) {
                LOG_ERROR("Distortion parameters not supported: k1={}, k2={}, p1={}, p2={}", k1, k2, p1, p2);
                throw std::runtime_error("Distortion parameters not supported");
            }
        }

        std::vector<std::shared_ptr<CameraNew>> cameras;
        if (transforms.contains("frames") && transforms["frames"].is_array()) {
            LOG_DEBUG("Processing {} frames", transforms["frames"].size());

            for (size_t frameInd = 0; frameInd < transforms["frames"].size(); ++frameInd) {
                auto& frame = transforms["frames"][frameInd];

                if (!frame.contains("transform_matrix")) {
                    LOG_ERROR("Frame {} missing transform_matrix", frameInd);
                    throw std::runtime_error("Frame missing transform_matrix");
                }

                if (!frame["transform_matrix"].is_array() || frame["transform_matrix"].size() != 4) {
                    LOG_ERROR("Frame {} has invalid transform_matrix dimensions", frameInd);
                    throw std::runtime_error("Invalid transform_matrix dimensions");
                }

                // Create camera-to-world transform matrix
                std::vector<float> c2w_data(16);
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        c2w_data[i * 4 + j] = frame["transform_matrix"][i][j].get<float>();
                    }
                }

                Tensor c2w = Tensor::from_vector(c2w_data, TensorShape{4, 4}, Device::CPU);

                // Change from OpenGL/Blender axes (Y up, Z back) to COLMAP (Y down, Z forward)
                // Flip Y and Z columns
                auto c2w_cpu = c2w.cpu();
                float* c2w_ptr = c2w_cpu.ptr<float>();
                for (int i = 0; i < 3; ++i) {
                    c2w_ptr[i * 4 + 1] *= -1.0f; // Flip Y column
                    c2w_ptr[i * 4 + 2] *= -1.0f; // Flip Z column
                }
                c2w = c2w_cpu;

                // Get world-to-camera transform
                Tensor w2c = inverse_4x4(c2w);

                // Apply Y-axis rotation fix
                Tensor fixMat = createYRotationMatrix(std::numbers::pi_v<float>);
                w2c = matmul_4x4(w2c, fixMat);

                // Extract rotation matrix R [3, 3]
                auto w2c_cpu = w2c.cpu();
                const float* w2c_ptr = w2c_cpu.ptr<float>();

                std::vector<float> R_data(9);
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        R_data[i * 3 + j] = w2c_ptr[i * 4 + j];
                    }
                }
                Tensor R = Tensor::from_vector(R_data, TensorShape{3, 3}, Device::CPU);

                // Extract translation vector T [3]
                std::vector<float> T_data = {w2c_ptr[0 * 4 + 3], w2c_ptr[1 * 4 + 3], w2c_ptr[2 * 4 + 3]};
                Tensor T = Tensor::from_vector(T_data, TensorShape{3}, Device::CPU);

                auto image_path = GetTransformImagePath(dir_path, frame);
                auto image_name = image_path.filename().string();

                // Empty distortion tensors
                Tensor radial_dist = Tensor::empty({0}, Device::CPU);
                Tensor tangential_dist = Tensor::empty({0}, Device::CPU);

                auto camera = std::make_shared<CameraNew>(
                    R,
                    T,
                    fl_x, fl_y,
                    cx, cy,
                    radial_dist,
                    tangential_dist,
                    gsplat::CameraModelType::PINHOLE,
                    image_name,
                    image_path,
                    w,
                    h,
                    static_cast<int>(frameInd));

                cameras.push_back(std::move(camera));
                LOG_TRACE("Processed frame {}: {}", frameInd, image_name);
            }
        }

        // Scene center is at origin for synthetic datasets
        Tensor center = Tensor::zeros({3}, Device::CPU);

        LOG_INFO("Loaded {} cameras from transforms file", cameras.size());

        return {std::move(cameras), center};
    }

    PointCloudNew generate_random_point_cloud() {
        LOG_DEBUG("Generating random point cloud with {} points", DEFAULT_NUM_INIT_GAUSSIAN);

        const int num_points = DEFAULT_NUM_INIT_GAUSSIAN;

        // Create random number generator
        std::mt19937 gen(DEFAULT_RANDOM_SEED);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        // Generate random positions in [-1, 1]^3
        std::vector<float> positions(num_points * 3);
        for (int i = 0; i < num_points * 3; ++i) {
            positions[i] = dist(gen) * 2.0f - 1.0f;
        }

        // Generate random colors in [0, 1]
        std::vector<float> colors(num_points * 3);
        for (int i = 0; i < num_points * 3; ++i) {
            colors[i] = dist(gen);
        }

        Tensor means = Tensor::from_vector(positions, TensorShape{num_points, 3}, Device::CUDA);
        Tensor colors_tensor = Tensor::from_vector(colors, TensorShape{num_points, 3}, Device::CUDA);

        return PointCloudNew(std::move(means), std::move(colors_tensor));
    }

} // namespace gs::loader