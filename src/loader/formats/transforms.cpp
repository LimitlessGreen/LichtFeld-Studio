/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "transforms.hpp"
#include "core/image_io.hpp"
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
        return 0.5f * (float)resolution / tanf(0.5f * fov_deg * std::numbers::pi / 180.0f);
    }

    float fov_rad_to_focal_length(int resolution, float fov_rad) {
        return 0.5f * (float)resolution / tanf(0.5f * fov_rad);
    }

    // Function to create a 3x3 rotation matrix around Y-axis embedded in 4x4 matrix
    void createYRotationMatrix(float angle_radians, float rotMat[16]) {
        float cos_angle = std::cos(angle_radians);
        float sin_angle = std::sin(angle_radians);

        // Initialize as identity matrix
        for (int i = 0; i < 16; ++i) {
            rotMat[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        }

        // Rotation matrix around Y-axis by angle θ:
        rotMat[0] = cos_angle;  // cos(θ)
        rotMat[2] = sin_angle;  // sin(θ)
        rotMat[8] = -sin_angle; // -sin(θ)
        rotMat[10] = cos_angle; // cos(θ)
    }

    std::filesystem::path GetTransformImagePath(const std::filesystem::path& dir_path, const nlohmann::json& frame) {
        auto image_path = dir_path / frame["file_path"];
        auto image_path_png = std::filesystem::path(image_path.string() + ".png");
        if (std::filesystem::exists(image_path_png)) {
            image_path = image_path_png;
            LOG_TRACE("Using PNG extension for image: {}", image_path.string());
        }
        return image_path;
    }

    TransformsCameraResult read_transforms_cameras_and_images_cuda(
        const std::filesystem::path& transPath) {

        LOG_TIMER_TRACE("Read transforms file");

        std::filesystem::path transformsFile = transPath;
        if (std::filesystem::is_directory(transPath)) {
            if (std::filesystem::is_regular_file(transPath / "transforms_train.json")) {
                transformsFile = transPath / "transforms_train.json";
            } else if (std::filesystem::is_regular_file(transPath / "transforms.json")) {
                transformsFile = transPath / "transforms.json";
            } else {
                LOG_ERROR("Could not find transforms file in: {}", transPath.string());
                throw std::runtime_error("could not find transforms_train.json nor transforms.json in " + transPath.string());
            }
        }

        if (!std::filesystem::is_regular_file(transformsFile)) {
            LOG_ERROR("Not a valid file: {}", transformsFile.string());
            throw std::runtime_error(transformsFile.string() + " is not a valid file");
        }

        LOG_DEBUG("Reading transforms from: {}", transformsFile.string());
        std::ifstream trans_file{transformsFile.string()};

        std::filesystem::path dir_path = transformsFile.parent_path();

        // should throw if parse fails
        nlohmann::json transforms = nlohmann::json::parse(trans_file, nullptr, true, true);
        int w = -1, h = -1;
        if (!transforms.contains("w") or !transforms.contains("h")) {
            try {
                LOG_DEBUG("Width/height not in transforms.json, reading from first image");
                auto first_frame_img_path = GetTransformImagePath(dir_path, transforms["frames"][0]);
                auto result = load_image(first_frame_img_path);
                w = std::get<1>(result);
                h = std::get<2>(result);
                LOG_DEBUG("Got image dimensions: {}x{}", w, h);
            } catch (const std::exception& e) {
                std::string error_msg = "Error while trying to read image dimensions: " + std::string(e.what());
                LOG_ERROR("{}", error_msg);
                throw std::runtime_error(error_msg);
            } catch (...) {
                std::string error_msg = "Unknown error while trying to read image dimensions";
                LOG_ERROR("{}", error_msg);
                throw std::runtime_error(error_msg);
            }
        } else {
            w = int(transforms["w"]);
            h = int(transforms["h"]);
        }

        float fl_x = -1, fl_y = -1;
        if (transforms.contains("fl_x")) {
            fl_x = float(transforms["fl_x"]);
        } else if (transforms.contains("camera_angle_x")) {
            fl_x = fov_rad_to_focal_length(w, float(transforms["camera_angle_x"]));
        }

        if (transforms.contains("fl_y")) {
            fl_y = float(transforms["fl_y"]);
        } else if (transforms.contains("camera_angle_y")) {
            fl_y = fov_rad_to_focal_length(h, float(transforms["camera_angle_y"]));
        } else {
            if (w != h) {
                LOG_ERROR("No camera_angle_y but w!=h: {}!={}", w, h);
                throw std::runtime_error("no camera_angle_y expected w!=h");
            }
            fl_y = fl_x;
        }

        float cx = -1, cy = -1;
        if (transforms.contains("cx")) {
            cx = float(transforms["cx"]);
        } else {
            cx = 0.5 * w;
        }

        if (transforms.contains("cy")) {
            cy = float(transforms["cy"]);
        } else {
            cy = 0.5 * h;
        }

        float k1 = 0, k2 = 0, p1 = 0, p2 = 0;
        if (transforms.contains("k1")) {
            k1 = float(transforms["k1"]);
        }
        if (transforms.contains("k2")) {
            k2 = float(transforms["k2"]);
        }
        if (transforms.contains("p1")) {
            p1 = float(transforms["p1"]);
        }
        if (transforms.contains("p2")) {
            p2 = float(transforms["p2"]);
        }
        if (k1 > 0 || k2 > 0 || p1 > 0 || p2 > 0) {
            LOG_ERROR("Distortion parameters not supported: k1={}, k2={}, p1={}, p2={}", k1, k2, p1, p2);
            throw std::runtime_error(std::format("GS don't support distortion for now: k1={}, k2={}, p1={}, p2={}", k1, k2, p1, p2));
        }

        std::vector<internal::CudaCameraData> camerasdata;
        if (transforms.contains("frames") && transforms["frames"].is_array()) {
            uint64_t counter = 0;
            LOG_DEBUG("Processing {} frames", transforms["frames"].size());

            for (size_t frameInd = 0; frameInd < transforms["frames"].size(); ++frameInd) {
                internal::CudaCameraData camdata;
                auto& frame = transforms["frames"][frameInd];
                if (!frame.contains("transform_matrix")) {
                    LOG_ERROR("Frame {} missing transform_matrix", frameInd);
                    throw std::runtime_error("expected all frames to contain transform_matrix");
                }
                if (!(frame["transform_matrix"].is_array() and frame["transform_matrix"].size() == 4)) {
                    LOG_ERROR("Frame {} has invalid transform_matrix dimensions", frameInd);
                    throw std::runtime_error("transform_matrix has the wrong dimensions");
                }

                // Create camera-to-world transform matrix
                float c2w[16];
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        c2w[i * 4 + j] = float(frame["transform_matrix"][i][j]);
                    }
                }

                // Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                // c2w[:3, 1:3] *= -1
                for (int i = 0; i < 3; ++i) {
                    c2w[i * 4 + 1] *= -1;
                    c2w[i * 4 + 2] *= -1;
                }

                // Get the world-to-camera transform by computing inverse of c2w
                float w2c[16];
                // Simple 4x4 matrix inverse (assuming affine transform)
                {
                    // Extract rotation and translation
                    float R[9], t[3];
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            R[i * 3 + j] = c2w[i * 4 + j];
                        }
                        t[i] = c2w[i * 4 + 3];
                    }

                    // Compute R^T
                    float RT[9];
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            RT[i * 3 + j] = R[j * 3 + i];
                        }
                    }

                    // Compute -R^T * t
                    float RTt[3];
                    for (int i = 0; i < 3; ++i) {
                        RTt[i] = 0;
                        for (int j = 0; j < 3; ++j) {
                            RTt[i] -= RT[i * 3 + j] * t[j];
                        }
                    }

                    // Build w2c matrix
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            w2c[i * 4 + j] = RT[i * 3 + j];
                        }
                        w2c[i * 4 + 3] = RTt[i];
                    }
                    w2c[12] = 0;
                    w2c[13] = 0;
                    w2c[14] = 0;
                    w2c[15] = 1;
                }

                // Fix so that the z direction will be the same (currently it is facing downward)
                float fixMat[16];
                createYRotationMatrix(M_PI, fixMat);

                // Matrix multiply w2c = w2c * fixMat
                float w2c_fixed[16];
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        w2c_fixed[i * 4 + j] = 0;
                        for (int k = 0; k < 4; ++k) {
                            w2c_fixed[i * 4 + j] += w2c[i * 4 + k] * fixMat[k * 4 + j];
                        }
                    }
                }

                // Extract rotation matrix R and translation vector T
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        camdata.R[i * 3 + j] = w2c_fixed[i * 4 + j];
                    }
                    camdata.T[i] = w2c_fixed[i * 4 + 3];
                }

                camdata.image_path = GetTransformImagePath(dir_path, frame);
                camdata.image_name = std::filesystem::path(camdata.image_path).filename().string();

                camdata.width = w;
                camdata.height = h;

                camdata.focal_x = fl_x;
                camdata.focal_y = fl_y;
                camdata.center_x = cx;
                camdata.center_y = cy;

                camdata.camera_model_type = 0; // PINHOLE
                camdata.camera_id = counter++;

                camerasdata.push_back(camdata);
                LOG_TRACE("Processed frame {}: {}", frameInd, camdata.image_name);
            }
        }

        TransformsCameraResult result;
        result.cameras = std::move(camerasdata);
        result.scene_center[0] = 0.0f;
        result.scene_center[1] = 0.0f;
        result.scene_center[2] = 0.0f;

        LOG_INFO("Loaded {} cameras from transforms file", result.cameras.size());
        return result;
    }

    internal::CudaPointCloud generate_random_point_cloud_cuda() {
        LOG_DEBUG("Generating random point cloud with {} points", DEFAULT_NUM_INIT_GAUSSIAN);

        int numInitGaussian = DEFAULT_NUM_INIT_GAUSSIAN;
        uint64_t seed = DEFAULT_RANDOM_SEED;

        // Use standard C++ random number generation
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> pos_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<uint8_t> color_dist(0, 255);

        // Generate host data
        std::vector<float> host_positions(numInitGaussian * 3);
        std::vector<uint8_t> host_colors(numInitGaussian * 3);

        for (int i = 0; i < numInitGaussian; ++i) {
            host_positions[i * 3 + 0] = pos_dist(gen);
            host_positions[i * 3 + 1] = pos_dist(gen);
            host_positions[i * 3 + 2] = pos_dist(gen);

            host_colors[i * 3 + 0] = color_dist(gen);
            host_colors[i * 3 + 1] = color_dist(gen);
            host_colors[i * 3 + 2] = color_dist(gen);
        }

        // Upload to CUDA
        internal::CudaPointCloud result(numInitGaussian);
        result.positions.upload(host_positions.data(), numInitGaussian * 3);
        result.colors.upload(host_colors.data(), numInitGaussian * 3);

        return result;
    }

} // namespace gs::loader