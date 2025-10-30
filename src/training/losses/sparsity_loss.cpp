/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sparsity_loss.hpp"
#include <format>

namespace lfs::training::losses {

    std::expected<std::pair<float, SparsityLoss::Context>, std::string>
    SparsityLoss::forward(
        const gs::SplatData& splatData,
        const Params& params) {
        try {
            if (params.optimizer_ptr && params.optimizer_ptr->should_apply_loss(params.current_iteration)) {
                // Initialize on first use (lazy initialization)
                if (!params.optimizer_ptr->is_initialized()) {
                    // NOTE: We need to initialize, but this requires mutable access
                    // In practice, the trainer will handle initialization before calling this
                    return std::unexpected("Sparsity optimizer not initialized - should be initialized by trainer");
                }

                auto loss_result = params.optimizer_ptr->compute_loss_forward(splatData.opacity_raw());
                if (!loss_result) {
                    return std::unexpected(loss_result.error());
                }
                return *loss_result;
            }
            // Return zero loss with empty context
            gs::training::SparsityLossContext empty_ctx;
            return std::make_pair(0.0f, empty_ctx);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing sparsity loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> SparsityLoss::backward(
        const Context& ctx,
        float grad_loss,
        const Params& params) {
        try {
            if (params.optimizer_ptr) {
                auto grad_result = params.optimizer_ptr->compute_loss_backward(ctx, grad_loss);
                if (!grad_result) {
                    return std::unexpected(grad_result.error());
                }
                return *grad_result;
            }
            // Return zero gradient if no optimizer
            return torch::zeros_like(ctx.opacities);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing sparsity loss backward: {}", e.what()));
        }
    }

} // namespace lfs::training::losses
