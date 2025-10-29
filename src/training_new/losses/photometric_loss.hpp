/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/tensor.hpp"
#include <expected>
#include <string>

namespace lfs::training::losses {

/**
 * @brief Photometric loss combining L1 and SSIM with manual gradient computation
 *
 * Loss = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
 *
 * This is a libtorch-free implementation that wraps the existing CUDA SSIM kernels.
 */
struct PhotometricLoss {
    struct Params {
        float lambda_dssim; ///< Weight for D-SSIM term (0.0 = pure L1, 1.0 = pure SSIM)
    };

    struct Context {
        lfs::core::Tensor grad_image; ///< [H, W, C] gradient w.r.t. rendered image
    };

    /**
     * @brief Compute photometric loss and gradient
     * @param rendered [H, W, C] rendered image
     * @param gt_image [H, W, C] ground truth image
     * @param params Loss parameters
     * @return (loss_value, context) or error
     */
    static std::expected<std::pair<float, Context>, std::string> forward(
        const lfs::core::Tensor& rendered,
        const lfs::core::Tensor& gt_image,
        const Params& params);
};

} // namespace lfs::training::losses
