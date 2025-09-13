/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "loader/cuda_data.hpp"
#include "core/splat_data.hpp"
#include "core/point_cloud.hpp"
#include "core/camera.hpp"
#include "Common.h"
#include <torch/torch.h>
#include <memory>

namespace gs::loader::internal {

// Convert internal CUDA representation to SplatData (which uses torch)
inline SplatData cuda_to_splat_data(CudaSplatData&& cuda_data) {
    // Create torch tensors that wrap the existing CUDA memory
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Create tensors from raw CUDA pointers - must clone to ensure ownership
    torch::Tensor means = torch::from_blob(
        cuda_data.means.data,
        {static_cast<int64_t>(cuda_data.num_points), 3},
        options
    ).clone();

    torch::Tensor sh0 = torch::from_blob(
        cuda_data.sh0.data,
        {static_cast<int64_t>(cuda_data.num_points),
         static_cast<int64_t>(cuda_data.sh0_dim1),
         static_cast<int64_t>(cuda_data.sh0_dim2)},
        options
    ).clone();

    torch::Tensor shN;
    if (cuda_data.shN_dim1 > 0 && cuda_data.shN_dim2 > 0) {
        shN = torch::from_blob(
            cuda_data.shN.data,
            {static_cast<int64_t>(cuda_data.num_points),
             static_cast<int64_t>(cuda_data.shN_dim1),
             static_cast<int64_t>(cuda_data.shN_dim2)},
            options
        ).clone();
    } else {
        shN = torch::zeros({static_cast<int64_t>(cuda_data.num_points), 0, 3}, options);
    }

    torch::Tensor scales = torch::from_blob(
        cuda_data.scales.data,
        {static_cast<int64_t>(cuda_data.num_points), 3},
        options
    ).clone();

    torch::Tensor rotations = torch::from_blob(
        cuda_data.rotations.data,
        {static_cast<int64_t>(cuda_data.num_points), 4},
        options
    ).clone();

    torch::Tensor opacity = torch::from_blob(
        cuda_data.opacity.data,
        {static_cast<int64_t>(cuda_data.num_points), 1},
        options
    ).clone();

    return SplatData(
        cuda_data.sh_degree,
        means,
        sh0,
        shN,
        scales,
        rotations,
        opacity,
        cuda_data.scene_scale
    );
}

// Convert CudaPointCloud to PointCloud
inline PointCloud cuda_to_point_cloud(const CudaPointCloud& cuda_pc) {
    if (cuda_pc.num_points == 0) {
        return PointCloud();
    }

    // Download from CUDA to host
    std::vector<float> host_positions(cuda_pc.num_points * 3);
    std::vector<uint8_t> host_colors(cuda_pc.num_points * 3);

    cuda_pc.positions.download(host_positions.data(), cuda_pc.num_points * 3);
    cuda_pc.colors.download(host_colors.data(), cuda_pc.num_points * 3);

    // Create torch tensors
    torch::Tensor positions = torch::from_blob(
        host_positions.data(),
        {static_cast<int64_t>(cuda_pc.num_points), 3},
        torch::kFloat32
    ).clone();

    torch::Tensor colors = torch::from_blob(
        host_colors.data(),
        {static_cast<int64_t>(cuda_pc.num_points), 3},
        torch::kUInt8
    ).clone();

    return PointCloud(positions, colors);
}

// Convert CudaCameraData to Camera
inline std::shared_ptr<Camera> cuda_to_camera(const CudaCameraData& cuda_cam, int uid) {
    // Convert rotation matrix
    torch::Tensor R = torch::from_blob(
        const_cast<float*>(cuda_cam.R),
        {3, 3},
        torch::kFloat32
    ).clone();

    // Convert translation vector
    torch::Tensor T = torch::from_blob(
        const_cast<float*>(cuda_cam.T),
        {3},
        torch::kFloat32
    ).clone();

    // Convert distortion parameters
    torch::Tensor radial_distortion;
    if (!cuda_cam.radial_distortion.empty()) {
        radial_distortion = torch::from_blob(
            const_cast<float*>(cuda_cam.radial_distortion.data()),
            {static_cast<int64_t>(cuda_cam.radial_distortion.size())},
            torch::kFloat32
        ).clone();
    } else {
        radial_distortion = torch::empty({0}, torch::kFloat32);
    }

    torch::Tensor tangential_distortion;
    if (!cuda_cam.tangential_distortion.empty()) {
        tangential_distortion = torch::from_blob(
            const_cast<float*>(cuda_cam.tangential_distortion.data()),
            {static_cast<int64_t>(cuda_cam.tangential_distortion.size())},
            torch::kFloat32
        ).clone();
    } else {
        tangential_distortion = torch::empty({0}, torch::kFloat32);
    }

    // Convert camera model type
    gsplat::CameraModelType camera_model_type = static_cast<gsplat::CameraModelType>(cuda_cam.camera_model_type);

    return std::make_shared<Camera>(
        R,
        T,
        cuda_cam.focal_x,
        cuda_cam.focal_y,
        cuda_cam.center_x,
        cuda_cam.center_y,
        radial_distortion,
        tangential_distortion,
        camera_model_type,
        cuda_cam.image_name,
        cuda_cam.image_path,
        cuda_cam.width,
        cuda_cam.height,
        uid
    );
}

// Convert scene center array to torch tensor
inline torch::Tensor array_to_tensor(const float center[3]) {
    return torch::tensor({center[0], center[1], center[2]}, torch::kFloat32);
}

} // namespace gs::loader::internal