/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/parameters.hpp"
#include "core_new/splat_data.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace lfs::training {

    // Initialize Gaussians (move to GPU, etc.)
    void initialize_gaussians(lfs::core::SplatData& splat_data);

    // Create optimizer for splat data
    std::unique_ptr<AdamOptimizer> create_optimizer(
        lfs::core::SplatData& splat_data,
        const lfs::core::param::OptimizationParameters& params);

    // Create exponential LR scheduler
    std::unique_ptr<ExponentialLR> create_scheduler(
        const lfs::core::param::OptimizationParameters& params,
        AdamOptimizer& optimizer);

    // Function types for parameter and optimizer state updates
    using ParamUpdateFn = std::function<lfs::core::Tensor(const int, const lfs::core::Tensor&)>;
    using OptimizerUpdateFn = std::function<void(
        AdamParamState& state,
        const lfs::core::Tensor& new_param)>;

    // Update parameter with optimizer state synchronization
    void update_param_with_optimizer(
        const ParamUpdateFn& param_fn,
        const OptimizerUpdateFn& optimizer_fn,
        std::unique_ptr<AdamOptimizer>& optimizer,
        lfs::core::SplatData& splat_data,
        std::vector<size_t> param_idxs = {0, 1, 2, 3, 4, 5});

} // namespace lfs::training
