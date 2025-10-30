/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "opacity_regularization.hpp"
#include "kernels/regularization.cuh"
#include <format>

namespace lfs::training::losses {

    std::expected<float, std::string> OpacityRegularization::forward(
        torch::Tensor& opacity_raw,
        const Params& params) {
        try {
            if (params.weight > 0.0f) {
                // Use efficient fused CUDA kernel that computes:
                // 1. sigmoid(opacity_raw)
                // 2. Accumulates gradient with chain rule: ∂L/∂opacity_raw = (weight/N) * σ(x) * (1 - σ(x))
                // 3. Returns loss: weight * mean(sigmoid(opacity_raw))
                float loss_value = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
                    opacity_raw,
                    params.weight);
                return loss_value;
            }
            return 0.0f;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing opacity regularization loss: {}", e.what()));
        }
    }

} // namespace lfs::training::losses
