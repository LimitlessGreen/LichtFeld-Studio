/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/splat_data_new.hpp"
#include "core/tensor.hpp"

namespace gs::training {
    struct RenderOutput;

    /**
     * LibTorch-free strategy interface for Gaussian Splatting training
     *
     * This is the equivalent of IStrategy but using the new tensor library
     * (gs::Tensor instead of torch::Tensor, SplatDataNew instead of SplatData).
     *
     * Strategies implement different densification/pruning algorithms:
     * - Default: Traditional densification with gradient-based splitting
     * - MCMC: Monte Carlo-based relocation with hard cap on Gaussian count
     */
    class IStrategyNew {
    public:
        virtual ~IStrategyNew() = default;

        /**
         * Initialize the strategy with optimization parameters.
         * This sets up optimizers, schedulers, and internal buffers.
         */
        virtual void initialize(const gs::param::OptimizationParameters& optimParams) = 0;

        /**
         * Called after backward pass to perform strategy-specific updates.
         * This may include densification, pruning, relocation, etc.
         *
         * @param iter Current training iteration
         * @param render_output Output from the rendering pass (contains gradients, etc.)
         */
        virtual void post_backward(int iter, RenderOutput& render_output) = 0;

        /**
         * Perform optimizer step and update learning rates.
         *
         * @param iter Current training iteration
         */
        virtual void step(int iter) = 0;

        /**
         * Check if we should perform refinement (densification/pruning) at this iteration.
         *
         * @param iter Current training iteration
         * @return true if refinement should occur
         */
        virtual bool is_refining(int iter) const = 0;

        /**
         * Get the underlying Gaussian model for rendering.
         * @return Reference to mutable SplatDataNew
         */
        virtual gs::SplatDataNew& get_model() = 0;

        /**
         * Get the underlying Gaussian model for rendering (const version).
         * @return Reference to const SplatDataNew
         */
        virtual const gs::SplatDataNew& get_model() const = 0;

        /**
         * Remove Gaussians based on a boolean mask.
         *
         * @param mask Boolean tensor of shape [N] where true indicates Gaussians to remove
         */
        virtual void remove_gaussians(const gs::Tensor& mask) = 0;
    };
} // namespace gs::training
