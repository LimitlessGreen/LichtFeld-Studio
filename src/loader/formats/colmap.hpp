/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "Common.h"
#include "core/camera_new.hpp"
#include "core/point_cloud_new.hpp"
#include "core/tensor.hpp"
#include <filesystem>
#include <memory>
#include <vector>

namespace gs::loader {

    /**
     * @brief Read COLMAP cameras and images
     * @param base Base directory containing COLMAP data
     * @param images_folder Folder containing images (default: "images")
     * @return Tuple of (vector of CameraNew, scene_center tensor [3])
     */
    std::tuple<std::vector<std::shared_ptr<CameraNew>>, Tensor>
    read_colmap_cameras_and_images(
        const std::filesystem::path& base,
        const std::string& images_folder = "images");

    /**
     * @brief Read COLMAP point cloud (binary format)
     * @param filepath Base directory containing points3D.bin
     * @return PointCloudNew
     */
    PointCloudNew read_colmap_point_cloud(const std::filesystem::path& filepath);

    /**
     * @brief Read COLMAP cameras and images from text files
     * @param base Base directory containing COLMAP data
     * @param images_folder Folder containing images (default: "images")
     * @return Tuple of (vector of CameraNew, scene_center tensor [3])
     */
    std::tuple<std::vector<std::shared_ptr<CameraNew>>, Tensor>
    read_colmap_cameras_and_images_text(
        const std::filesystem::path& base,
        const std::string& images_folder = "images");

    /**
     * @brief Read COLMAP point cloud from text file
     * @param filepath Base directory containing points3D.txt
     * @return PointCloudNew
     */
    PointCloudNew read_colmap_point_cloud_text(const std::filesystem::path& filepath);

} // namespace gs::loader