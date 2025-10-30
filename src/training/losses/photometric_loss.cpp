/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "photometric_loss.hpp"
#include "kernels/ssim.cuh"
#include <format>

namespace lfs::training::losses {

    std::expected<std::pair<float, PhotometricLoss::Context>, std::string>
    PhotometricLoss::forward(
        const torch::Tensor& rendered,
        const torch::Tensor& gt_image,
        const Params& params) {
        try {
            torch::Tensor rendered_4d = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
            torch::Tensor gt_4d = gt_image.dim() == 3 ? gt_image.unsqueeze(0) : gt_image;

            TORCH_CHECK(rendered_4d.sizes() == gt_4d.sizes(),
                        "ERROR: size mismatch – rendered ", rendered_4d.sizes(),
                        " vs. ground truth ", gt_4d.sizes());

            // Compute L1 loss and gradient manually
            // L1 loss: mean(|rendered - gt|)
            // Gradient: sign(rendered - gt) / N
            auto diff = rendered_4d - gt_4d;
            auto l1_loss_tensor = diff.abs().mean(); // Keep as tensor (no sync!)
            auto grad_l1 = diff.sign() / static_cast<float>(diff.numel());

            // Compute SSIM loss and gradient manually (no autograd)
            auto [ssim_value, ssim_ctx] = ssim_forward(rendered_4d, gt_4d, /*apply_valid_padding=*/true);
            float ssim_loss = 1.0f - ssim_value;

            // Backward: d(loss)/d(ssim) = -1 (since loss = 1 - ssim)
            auto grad_ssim = ssim_backward(ssim_ctx, -1.0f);

            // Handle dimension: if input was 3D, squeeze the output
            if (rendered.dim() == 3 && grad_ssim.dim() == 4) {
                grad_ssim = grad_ssim.squeeze(0);
            }

            // Combine gradients: grad = (1 - lambda) * grad_l1 + lambda * grad_ssim
            auto grad_combined = (1.0f - params.lambda_dssim) * grad_l1 +
                                 params.lambda_dssim * grad_ssim;

            // Remove batch dimension if input was 3D
            if (rendered.dim() == 3) {
                grad_combined = grad_combined.squeeze(0);
            }

            // Compute total loss value (both are already floats, no CPU sync needed!)
            float l1_loss = l1_loss_tensor.item<float>(); // Extract scalar
            float total_loss = (1.0f - params.lambda_dssim) * l1_loss +
                               params.lambda_dssim * ssim_loss;

            Context ctx{.grad_image = grad_combined};
            return std::make_pair(total_loss, ctx);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing photometric loss with gradient: {}", e.what()));
        }
    }

} // namespace lfs::training::losses
