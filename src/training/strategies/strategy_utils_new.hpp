/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data_new.hpp"
#include "istrategy_new.hpp"
#include "optimizers/fused_adam.hpp"
#include "optimizers/scheduler.hpp"
#include <memory>

namespace gs::training {

    /**
     * @brief Initialize Gaussians by ensuring gradients are allocated
     * @param splat_data SplatDataNew instance
     */
    void initialize_gaussians_new(gs::SplatDataNew& splat_data);

    /**
     * @brief Create FusedAdam optimizer for SplatDataNew
     * @param splat_data SplatDataNew with parameter tensors
     * @param params Optimization parameters
     * @return Unique pointer to configured FusedAdam optimizer
     */
    std::unique_ptr<FusedAdam> create_optimizer_new(
        gs::SplatDataNew& splat_data,
        const gs::param::OptimizationParameters& params);

    /**
     * @brief Create ExponentialLR scheduler
     * @param params Optimization parameters
     * @param optimizer Pointer to FusedAdam optimizer
     * @param param_group_index Which parameter group to schedule (-1 for all)
     * @return Unique pointer to configured scheduler
     */
    std::unique_ptr<ExponentialLR> create_scheduler_new(
        const gs::param::OptimizationParameters& params,
        FusedAdam* optimizer,
        int param_group_index = -1);

} // namespace gs::training
