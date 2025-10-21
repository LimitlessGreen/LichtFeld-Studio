/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_new.hpp"
#include "core/point_cloud_new.hpp"
#include "core/tensor.hpp"
#include <filesystem>
#include <memory>
#include <vector>

namespace gs::loader {

    /**
     * @brief Read cameras from Blender/NeRF transforms.json file
     * @param transPath Path to transforms.json file or directory containing it
     * @return Tuple of (vector of CameraNew, scene_center tensor [3])
     */
    std::tuple<std::vector<std::shared_ptr<CameraNew>>, Tensor>
    read_transforms_cameras_and_images(const std::filesystem::path& transPath);

    /**
     * @brief Generate random point cloud for initialization
     * @return PointCloudNew with random points in [-1, 1]^3
     */
    PointCloudNew generate_random_point_cloud();

} // namespace gs::loader