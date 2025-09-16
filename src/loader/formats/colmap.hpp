/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "loader/cuda_data.hpp"
#include <filesystem>
#include <vector>

namespace gs::loader {

    enum class CAMERA_MODEL {
        SIMPLE_PINHOLE = 0,
        PINHOLE = 1,
        SIMPLE_RADIAL = 2,
        RADIAL = 3,
        OPENCV = 4,
        OPENCV_FISHEYE = 5,
        FULL_OPENCV = 6,
        FOV = 7,
        SIMPLE_RADIAL_FISHEYE = 8,
        RADIAL_FISHEYE = 9,
        THIN_PRISM_FISHEYE = 10,
        UNDEFINED = 11
    };

    // Structure to return camera data with scene center
    struct ColmapCameraResult {
        std::vector<internal::CudaCameraData> cameras;
        float scene_center[3];
    };

    // CUDA-native versions
    ColmapCameraResult read_colmap_cameras_and_images_cuda(
        const std::filesystem::path& base,
        const std::string& images_folder = "images");

    internal::CudaPointCloud read_colmap_point_cloud_cuda(const std::filesystem::path& filepath);

    ColmapCameraResult read_colmap_cameras_and_images_text_cuda(
        const std::filesystem::path& base,
        const std::string& images_folder = "images");

    internal::CudaPointCloud read_colmap_point_cloud_text_cuda(const std::filesystem::path& filepath);

} // namespace gs::loader