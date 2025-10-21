/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data_new.hpp"
#include "core/tensor.hpp"
#include "strategies/istrategy_new.hpp"

namespace gs::training {
    // Forward declarations
    class FusedAdam;
    class ExponentialLR;

    /**
     * MCMC-based optimization strategy (LibTorch-free version)
     *
     * This strategy uses Monte Carlo Markov Chain methods for Gaussian optimization,
     * with a hard cap on the maximum number of Gaussians (controlled by max_cap parameter).
     */
    class MCMCNew : public IStrategyNew {
    public:
        explicit MCMCNew(gs::SplatDataNew&& splat_data);
        ~MCMCNew() override; // Must be defined in .cpp due to unique_ptr of incomplete types

        void initialize(const gs::param::OptimizationParameters& params) override;
        void post_backward(int iter, RenderOutput& render_output) override;
        void step(int iter) override;
        void remove_gaussians(const gs::Tensor& mask) override;
        bool is_refining(int iter) const override;

        gs::SplatDataNew& get_model() override { return _splat_data; }
        const gs::SplatDataNew& get_model() const override { return _splat_data; }

        // Legacy accessor for backward compatibility
        gs::SplatDataNew& splat_data() { return _splat_data; }
        const gs::SplatDataNew& splat_data() const { return _splat_data; }

        /**
         * Sample indices from a categorical distribution using multinomial sampling.
         *
         * This implementation handles both small and large weight arrays:
         * - For small arrays: Uses the built-in Tensor::multinomial if available
         * - For large arrays: Implements manual sampling using cumulative distribution
         *
         * @param weights 1D tensor of non-negative weights (will be normalized internally)
         * @param n Number of samples to draw
         * @param replacement Whether to sample with replacement
         * @return Tensor of shape [n] containing sampled indices (Int64)
         */
        static gs::Tensor multinomial_sample(const gs::Tensor& weights, int n, bool replacement);

    private:
        /**
         * Relocate dead Gaussians by sampling from alive ones.
         * Dead Gaussians are determined by low opacity or invalid rotations.
         *
         * @return Number of Gaussians relocated
         */
        int relocate_gs();

        /**
         * Add new Gaussians to grow towards max_cap.
         * New Gaussians are sampled from existing ones proportional to their opacity.
         *
         * @return Number of new Gaussians added
         */
        int add_new_gs();

        /**
         * Inject noise to Gaussian parameters to encourage exploration.
         * Noise is scaled by the current learning rate.
         */
        void inject_noise();

        /**
         * Update optimizer state when relocating Gaussians.
         * Currently a placeholder - optimizer will adapt naturally.
         */
        void update_optimizer_for_relocate(FusedAdam* optimizer,
                                           const gs::Tensor& sampled_indices,
                                           const gs::Tensor& dead_indices,
                                           int param_position);

    private:
        gs::SplatDataNew _splat_data;
        std::unique_ptr<const gs::param::OptimizationParameters> _params;
        std::unique_ptr<FusedAdam> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;

        // Binomial coefficients for relocation
        gs::Tensor _binoms;

        // Noise buffer for inject_noise (using Tensor instead of raw CUDA)
        gs::Tensor noise_buffer_;

        // Noise learning rate multiplier
        float _noise_lr = 5e5f;
    };

} // namespace gs::training
