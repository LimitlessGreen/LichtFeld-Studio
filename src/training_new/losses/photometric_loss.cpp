/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "photometric_loss.hpp"
#include "kernels/ssim.cuh"  // CUDA SSIM kernels
#include <format>
#include <torch/torch.h>

namespace lfs::training::losses {

// Helper to convert lfs::core::Tensor to torch::Tensor (zero-copy view)
static torch::Tensor to_torch_view(const lfs::core::Tensor& tensor) {
    std::vector<int64_t> torch_shape;
    for (size_t i = 0; i < tensor.ndim(); i++) {
        torch_shape.push_back(static_cast<int64_t>(tensor.shape()[i]));
    }

    torch::Device device = tensor.device() == lfs::core::Device::CUDA
                              ? torch::kCUDA
                              : torch::kCPU;

    return torch::from_blob(
        const_cast<float*>(tensor.ptr<float>()),
        torch_shape,
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

// OPTIMIZED: Helper to create torch::Tensor VIEW of lfs::core::Tensor gradient buffer (zero-copy)
static torch::Tensor create_grad_view(lfs::core::Tensor& grad_tensor) {
    std::vector<int64_t> torch_shape;
    for (size_t i = 0; i < grad_tensor.ndim(); i++) {
        torch_shape.push_back(static_cast<int64_t>(grad_tensor.shape()[i]));
    }

    torch::Device device = grad_tensor.device() == lfs::core::Device::CUDA
                              ? torch::kCUDA
                              : torch::kCPU;

    // Create a writable view - torch will write directly into our gradient buffer!
    return torch::from_blob(
        grad_tensor.ptr<float>(),
        torch_shape,
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

std::expected<std::pair<float, PhotometricLoss::Context>, std::string>
PhotometricLoss::forward(
    const lfs::core::Tensor& rendered,
    const lfs::core::Tensor& gt_image,
    const Params& params) {
    try {
        // Validate inputs
        if (rendered.device() != lfs::core::Device::CUDA || gt_image.device() != lfs::core::Device::CUDA) {
            return std::unexpected("rendered and gt_image must be on CUDA device");
        }

        if (rendered.shape() != gt_image.shape()) {
            return std::unexpected("Shape mismatch: rendered and gt_image must have same shape");
        }

        // Convert to torch views (zero-copy)
        auto torch_rendered = to_torch_view(rendered);
        auto torch_gt = to_torch_view(gt_image);

        // Add batch dimension if needed (SSIM expects 4D)
        torch::Tensor rendered_4d = rendered.ndim() == 3 ? torch_rendered.unsqueeze(0) : torch_rendered;
        torch::Tensor gt_4d = gt_image.ndim() == 3 ? torch_gt.unsqueeze(0) : torch_gt;

        // Compute L1 loss and gradient manually
        auto diff = rendered_4d - gt_4d;
        auto l1_loss_tensor = diff.abs().mean();
        auto grad_l1 = diff.sign() / static_cast<float>(diff.numel());

        // Compute SSIM loss and gradient (calls existing CUDA kernels)
        auto [ssim_value, ssim_ctx] = ssim_forward(rendered_4d, gt_4d, /*apply_valid_padding=*/true);
        float ssim_loss = 1.0f - ssim_value;

        // Backward: d(loss)/d(ssim) = -1 (since loss = 1 - ssim)
        auto grad_ssim = ssim_backward(ssim_ctx, -1.0f);

        // Remove batch dimension if input was 3D
        if (rendered.ndim() == 3 && grad_ssim.dim() == 4) {
            grad_ssim = grad_ssim.squeeze(0);
        }

        // OPTIMIZED: Pre-allocate gradient buffer (same shape as input)
        auto grad_image = lfs::core::Tensor::zeros(rendered.shape(), lfs::core::Device::CUDA);

        // Create torch view of our gradient buffer (zero-copy!)
        auto grad_view = create_grad_view(grad_image);

        // Handle batch dimension for gradient view
        torch::Tensor grad_view_4d = rendered.ndim() == 3 ? grad_view.unsqueeze(0) : grad_view;

        // Combine gradients: grad = (1 - lambda) * grad_l1 + lambda * grad_ssim
        // Write DIRECTLY into our pre-allocated buffer!
        auto grad_combined = (1.0f - params.lambda_dssim) * grad_l1 +
                            params.lambda_dssim * grad_ssim;

        // Copy the result into our buffer (single GPU→GPU copy, much faster!)
        grad_view_4d.copy_(grad_combined);

        // Compute total loss
        float l1_loss = l1_loss_tensor.item<float>();
        float total_loss = (1.0f - params.lambda_dssim) * l1_loss +
                          params.lambda_dssim * ssim_loss;

        // Return pre-allocated gradient (zero-copy from here on!)
        Context ctx{.grad_image = std::move(grad_image)};

        return std::make_pair(total_loss, ctx);

    } catch (const std::exception& e) {
        return std::unexpected(std::format("Error in PhotometricLoss::forward: {}", e.what()));
    }
}

} // namespace lfs::training::losses
