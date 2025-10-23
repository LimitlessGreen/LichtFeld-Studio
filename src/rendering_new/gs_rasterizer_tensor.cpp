/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gs_rasterizer_tensor.hpp"
#include "core_new/logger.hpp"
#include "rasterization_api_tensor.h"
#include <glm/glm.hpp>

namespace lfs::rendering {

    Tensor rasterize_tensor(
        const lfs::core::Camera& viewpoint_camera,
        const lfs::core::SplatData& gaussian_model,
        const Tensor& bg_color) {

        // Get camera parameters
        float fx = viewpoint_camera.focal_x();
        float fy = viewpoint_camera.focal_y();
        float cx = viewpoint_camera.center_x();
        float cy = viewpoint_camera.center_y();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        // Build world-to-camera transform matrix [4, 4]
        // w2c = [R | t]
        //       [0 | 1]
        const auto& R = viewpoint_camera.R(); // [3, 3] rotation
        const auto& T = viewpoint_camera.T(); // [3] translation

        // Create w2c matrix [4, 4] on CPU
        std::vector<float> w2c_data(16, 0.0f);

        // Copy rotation (first 3x3)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                // Access tensor data through CPU copy
                auto R_cpu = R.cpu();
                w2c_data[i * 4 + j] = R_cpu.ptr<float>()[i * 3 + j];
            }
        }

        // Copy translation (last column of first 3 rows)
        auto T_cpu = T.cpu();
        const float* T_ptr = T_cpu.ptr<float>();
        w2c_data[0 * 4 + 3] = T_ptr[0];
        w2c_data[1 * 4 + 3] = T_ptr[1];
        w2c_data[2 * 4 + 3] = T_ptr[2];

        // Set last row [0, 0, 0, 1]
        w2c_data[3 * 4 + 3] = 1.0f;

        Tensor w2c = Tensor::from_vector(w2c_data, {4, 4}, lfs::core::Device::CPU).cuda();

        // Camera position is -R^T @ t
        // We can compute this from the existing R and T
        auto R_t = R.transpose(0, 1);                 // R^T
        auto T_expanded = T.unsqueeze(1);             // [3, 1]
        auto cam_pos = -R_t.mm(T_expanded).squeeze(); // [3]

        // Get model data
        const auto& means = gaussian_model.means_raw();
        const auto& scales_raw = gaussian_model.scaling_raw();
        const auto& rotations_raw = gaussian_model.rotation_raw();
        const auto& opacities_raw = gaussian_model.opacity_raw();
        const auto& sh0 = gaussian_model.sh0_raw();
        const auto& shN = gaussian_model.shN_raw();

        // Call the tensor-based forward wrapper
        auto [image, alpha] = forward_wrapper_tensor(
            means,
            scales_raw,
            rotations_raw,
            opacities_raw,
            sh0,
            shN,
            w2c,
            cam_pos,
            active_sh_bases,
            viewpoint_camera.camera_width(),
            viewpoint_camera.camera_height(),
            fx,
            fy,
            cx,
            cy,
            near_plane,
            far_plane);

        // Manually blend the background since the forward pass does not support it
        // bg_color is [3], need to make it [3, 1, 1]
        Tensor bg = bg_color.unsqueeze(1).unsqueeze(2); // [3, 1, 1]

        // blended_image = image + (1.0 - alpha) * bg
        // Note: Tensor - Tensor works, but float - Tensor doesn't
        Tensor one_tensor = Tensor::ones_like(alpha);
        Tensor one_minus_alpha = one_tensor - alpha; // 1.0 - alpha
        Tensor blended_image = image + one_minus_alpha * bg;

        // Clamp to [0, 1] range
        blended_image = blended_image.clamp(0.0f, 1.0f);

        LOG_TRACE("Tensor rasterization completed: {}x{}",
                  viewpoint_camera.camera_width(),
                  viewpoint_camera.camera_height());

        return blended_image;
    }

} // namespace lfs::rendering
