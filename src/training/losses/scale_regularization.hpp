/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <string>
#include <torch/torch.h>

namespace lfs::training::losses {

/**
 * @brief L1 regularization on exp(scaling_raw) with fused CUDA kernel
 *
 * Forward:  scaling = exp(scaling_raw)
 * Loss:     L = weight * mean(scaling)
 * Gradient: ∂L/∂scaling_raw = (weight / N) * exp(scaling_raw)
 *
 * NOTE: This loss writes gradients directly to scaling_raw.grad() in-place
 */
struct ScaleRegularization {
    struct Params {
        float weight; ///< Regularization weight
    };

    /**
     * @brief Compute scale regularization loss and accumulate gradients
     * @param scaling_raw [N, 3] raw scaling parameters (requires_grad=true)
     * @param params Loss parameters
     * @return loss_value or error
     * @note Accumulates gradients directly to scaling_raw.grad()
     */
    static std::expected<float, std::string> forward(
        torch::Tensor& scaling_raw,
        const Params& params);
};

} // namespace lfs::training::losses
