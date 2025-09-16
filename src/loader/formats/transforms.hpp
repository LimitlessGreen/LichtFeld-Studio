/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "loader/cuda_data.hpp"
#include <filesystem>
#include <vector>

namespace gs::loader {

    struct TransformsCameraResult {
        std::vector<internal::CudaCameraData> cameras;
        float scene_center[3];
    };

    TransformsCameraResult read_transforms_cameras_and_images_cuda(
        const std::filesystem::path& transPath);

    internal::CudaPointCloud generate_random_point_cloud_cuda();

} // namespace gs::loader