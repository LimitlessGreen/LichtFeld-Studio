/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scale_regularization.hpp"
#include "kernels/regularization.cuh"
#include <format>

namespace lfs::training::losses {

std::expected<float, std::string> ScaleRegularization::forward(
    torch::Tensor& scaling_raw,
    const Params& params) {
    try {
        if (params.weight > 0.0f) {
            // Efficient fused CUDA kernel for exp regularization with chain rule
            // Forward:  scaling = exp(_scaling)
            // Loss:     L = weight * mean(scaling)
            // Gradient: ∂L/∂_scaling = (weight / N) * exp(_scaling)
            float loss = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
                scaling_raw,
                params.weight);
            return loss;
        }
        return 0.0f;
    } catch (const std::exception& e) {
        return std::unexpected(std::format("Error computing scale regularization loss: {}", e.what()));
    }
}

} // namespace lfs::training::losses
