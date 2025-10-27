/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "photometric_loss.hpp"
#include "lfs/kernels/ssim.cuh"
#include "lfs/kernels/l1_loss.cuh"
#include <format>

namespace lfs::training::losses {

std::expected<std::pair<float, PhotometricLoss::Context>, std::string>
PhotometricLoss::forward(
    const lfs::core::Tensor& rendered,
    const lfs::core::Tensor& gt_image,
    const Params& params) {
    try {
        // Ensure 4D shape [N, C, H, W] by adding batch dimension if needed
        auto rendered_4d = rendered.ndim() == 3 ? rendered.unsqueeze(0) : rendered;
        auto gt_4d = gt_image.ndim() == 3 ? gt_image.unsqueeze(0) : gt_image;

        // Validate shapes
        if (rendered_4d.shape() != gt_4d.shape()) {
            return std::unexpected("Shape mismatch: rendered and gt_image must have same shape");
        }

        lfs::core::Tensor grad_combined;
        float total_loss;

        // Optimize: only compute what's needed based on lambda_dssim
        if (params.lambda_dssim == 0.0f) {
            // Pure L1 loss - use optimized fused kernel
            size_t N = rendered_4d.numel();
            size_t num_blocks = std::min((N + 255) / 256, size_t(1024));

            grad_combined = lfs::core::Tensor::empty(rendered_4d.shape(), lfs::core::Device::CUDA);
            auto loss_tensor = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::CUDA);

            lfs::training::kernels::launch_fused_l1_loss(
                rendered_4d.ptr<float>(),
                gt_4d.ptr<float>(),
                grad_combined.ptr<float>(),
                loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                N,
                nullptr);

            total_loss = loss_tensor.item<float>();

        } else if (params.lambda_dssim == 1.0f) {
            // Pure SSIM loss - skip L1 computation entirely
            auto [ssim_value, ssim_ctx] = lfs::training::kernels::ssim_forward(
                rendered_4d, gt_4d, /*apply_valid_padding=*/true);
            float ssim_loss = 1.0f - ssim_value;

            // Backward: d(loss)/d(ssim) = -1 (since loss = 1 - ssim)
            grad_combined = lfs::training::kernels::ssim_backward(ssim_ctx, -1.0f);
            total_loss = ssim_loss;

        } else {
            // Combined loss - compute both
            size_t N = rendered_4d.numel();
            size_t num_blocks = std::min((N + 255) / 256, size_t(1024));

            // L1 component - use fused kernel
            auto grad_l1 = lfs::core::Tensor::empty(rendered_4d.shape(), lfs::core::Device::CUDA);
            auto l1_loss_tensor = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::CUDA);

            lfs::training::kernels::launch_fused_l1_loss(
                rendered_4d.ptr<float>(),
                gt_4d.ptr<float>(),
                grad_l1.ptr<float>(),
                l1_loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                N,
                nullptr);

            float l1_loss = l1_loss_tensor.item<float>();

            // SSIM component
            auto [ssim_value, ssim_ctx] = lfs::training::kernels::ssim_forward(
                rendered_4d, gt_4d, /*apply_valid_padding=*/true);
            float ssim_loss = 1.0f - ssim_value;

            // Backward: d(loss)/d(ssim) = -1 (since loss = 1 - ssim)
            auto grad_ssim = lfs::training::kernels::ssim_backward(ssim_ctx, -1.0f);

            // Combine gradients: grad = (1 - lambda) * grad_l1 + lambda * grad_ssim
            grad_combined = grad_l1 * (1.0f - params.lambda_dssim) +
                           grad_ssim * params.lambda_dssim;

            // Compute total loss
            total_loss = (1.0f - params.lambda_dssim) * l1_loss +
                        params.lambda_dssim * ssim_loss;
        }

        // Remove batch dimension if input was 3D
        if (rendered.ndim() == 3) {
            grad_combined = grad_combined.squeeze(0);
        }

        Context ctx{.grad_image = grad_combined};
        return std::make_pair(total_loss, ctx);
    } catch (const std::exception& e) {
        return std::unexpected(std::format("Error computing photometric loss with gradient: {}", e.what()));
    }
}

} // namespace lfs::training::losses
