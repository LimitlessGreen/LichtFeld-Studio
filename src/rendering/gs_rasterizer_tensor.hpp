/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_new.hpp"
#include "core/splat_data_new.hpp"
#include "core/tensor.hpp"

namespace gs::rendering {

    /**
     * @brief Rasterize Gaussians using tensor-based backend (libtorch-free)
     *
     * @param viewpoint_camera Camera parameters
     * @param gaussian_model Gaussian splat data
     * @param bg_color Background color [3]
     * @return Rendered image [3, H, W]
     */
    Tensor rasterize_tensor(
        const CameraNew& viewpoint_camera,
        const SplatDataNew& gaussian_model,
        const Tensor& bg_color);

} // namespace gs::rendering
