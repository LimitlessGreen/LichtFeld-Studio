/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core_new/tensor.hpp"
#include <tuple>

namespace lfs::training::kernels {

    // Context for manual SSIM forward/backward (like RasterizeContext)
    struct SSIMContext {
        lfs::core::Tensor img1;
        lfs::core::Tensor img2;
        lfs::core::Tensor dm_dmu1;
        lfs::core::Tensor dm_dsigma1_sq;
        lfs::core::Tensor dm_dsigma12;
        int original_h;
        int original_w;
        bool apply_valid_padding;
    };

    // Manual SSIM forward (no autograd) - returns (loss_value, context)
    std::pair<float, SSIMContext> ssim_forward(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        bool apply_valid_padding = true);

    // Manual SSIM backward (no autograd) - computes gradient w.r.t. img1
    lfs::core::Tensor ssim_backward(
        const SSIMContext& ctx,
        float grad_loss); // Gradient of loss w.r.t. SSIM value (scalar)

} // namespace lfs::training::kernels
