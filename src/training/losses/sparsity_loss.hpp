/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../components/sparsity_optimizer.hpp"
#include "core/splat_data.hpp"
#include <expected>
#include <memory>
#include <string>
#include <torch/torch.h>

namespace lfs::training::losses {

    /**
     * @brief Sparsity loss using ADMM optimizer (wrapper for ISparsityOptimizer)
     *
     * This is a wrapper around the ISparsityOptimizer interface that provides
     * a consistent API with other losses while delegating to the sparsity optimizer.
     *
     * The context is the same SparsityLossContext from the optimizer.
     */
    struct SparsityLoss {
        // Re-export the context type for consistency
        using Context = gs::training::SparsityLossContext;

        struct Params {
            int current_iteration;                                 ///< Current training iteration
            const gs::training::ISparsityOptimizer* optimizer_ptr; ///< Pointer to sparsity optimizer (nullable)
        };

        /**
         * @brief Compute sparsity loss and context
         * @param opacity_raw [N, 1] raw opacity parameters
         * @param splatData Full splat data (for accessing opacity_raw)
         * @param params Loss parameters including optimizer and iteration
         * @return (loss_value, context) or error
         */
        static std::expected<std::pair<float, Context>, std::string> forward(
            const gs::SplatData& splatData,
            const Params& params);

        /**
         * @brief Compute sparsity loss gradient manually
         * @param ctx Context from forward pass
         * @param grad_loss Gradient of total loss w.r.t. sparsity loss (usually 1.0)
         * @param params Loss parameters including optimizer
         * @return Gradient w.r.t. opacity_raw or error
         */
        static std::expected<torch::Tensor, std::string> backward(
            const Context& ctx,
            float grad_loss,
            const Params& params);
    };

} // namespace lfs::training::losses
