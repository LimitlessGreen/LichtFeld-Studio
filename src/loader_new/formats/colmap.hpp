/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "Common.h"
#include "core_new/camera.hpp"
#include "core_new/point_cloud.hpp"
#include "core_new/tensor.hpp"
#include <filesystem>
#include <memory>
#include <vector>

namespace lfs::loader {

    // Import types from lfs::core for convenience
    using lfs::core::Camera;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::PointCloud;
    using lfs::core::Tensor;

    // Camera data structure used for intermediate loading before Camera creation
    struct CameraData {
        // Static data loaded from COLMAP/transforms
        uint32_t _camera_ID = 0;
        Tensor _R = Tensor::eye(3);
        Tensor _T = Tensor::zeros({3}, Device::CPU, DataType::Float32);
        float _focal_x = 0.f;
        float _focal_y = 0.f;
        float _center_x = 0.f;
        float _center_y = 0.f;
        std::string _image_name;
        std::filesystem::path _image_path;
        gsplat::CameraModelType _camera_model_type = gsplat::CameraModelType::PINHOLE;
        int _width = 0;
        int _height = 0;
    };

    /**
     * @brief Read COLMAP cameras and images
     * @param base Base directory containing COLMAP data
     * @param images_folder Folder containing images (default: "images")
     * @return Tuple of (vector of Camera, scene_center tensor [3])
     */
    std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>
    read_colmap_cameras_and_images(
        const std::filesystem::path& base,
        const std::string& images_folder = "images");

    /**
     * @brief Read COLMAP point cloud (binary format)
     * @param filepath Base directory containing points3D.bin
     * @return PointCloud
     */
    PointCloud read_colmap_point_cloud(const std::filesystem::path& filepath);

    /**
     * @brief Read COLMAP cameras and images from text files
     * @param base Base directory containing COLMAP data
     * @param images_folder Folder containing images (default: "images")
     * @return Tuple of (vector of Camera, scene_center tensor [3])
     */
    std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>
    read_colmap_cameras_and_images_text(
        const std::filesystem::path& base,
        const std::string& images_folder = "images");

    /**
     * @brief Read COLMAP point cloud from text file
     * @param filepath Base directory containing points3D.txt
     * @return PointCloud
     */
    PointCloud read_colmap_point_cloud_text(const std::filesystem::path& filepath);

} // namespace lfs::loader