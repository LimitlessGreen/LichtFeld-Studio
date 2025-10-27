/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <string>
#include <torch/torch.h>

namespace lfs::training::losses {

/**
 * @brief Photometric loss combining L1 and SSIM with manual gradient computation
 *
 * Loss = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
 */
struct PhotometricLoss {
    struct Params {
        float lambda_dssim; ///< Weight for D-SSIM term (0.0 = pure L1, 1.0 = pure SSIM)
    };

    struct Context {
        torch::Tensor grad_image; ///< [H, W, C] gradient w.r.t. rendered image
    };

    /**
     * @brief Compute photometric loss and gradient
     * @param rendered [H, W, C] rendered image
     * @param gt_image [H, W, C] ground truth image
     * @param params Loss parameters
     * @return (loss_value, context) or error
     */
    static std::expected<std::pair<float, Context>, std::string> forward(
        const torch::Tensor& rendered,
        const torch::Tensor& gt_image,
        const Params& params);
};

} // namespace lfs::training::losses
