/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <string>
#include <torch/torch.h>

namespace lfs::training::losses {

    /**
     * @brief L1 regularization on sigmoid(opacity_raw) with fused CUDA kernel
     *
     * Forward:  opacity = sigmoid(opacity_raw)
     * Loss:     L = weight * mean(opacity)
     * Gradient: ∂L/∂opacity_raw = (weight / N) * sigmoid(x) * (1 - sigmoid(x))
     *
     * NOTE: This loss writes gradients directly to opacity_raw.grad() in-place
     */
    struct OpacityRegularization {
        struct Params {
            float weight; ///< Regularization weight
        };

        /**
         * @brief Compute opacity regularization loss and accumulate gradients
         * @param opacity_raw [N, 1] raw opacity parameters (requires_grad=true)
         * @param params Loss parameters
         * @return loss_value or error
         * @note Accumulates gradients directly to opacity_raw.grad()
         */
        static std::expected<float, std::string> forward(
            torch::Tensor& opacity_raw,
            const Params& params);
    };

} // namespace lfs::training::losses
