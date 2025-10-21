/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "strategy_utils_new.hpp"
#include "core/logger.hpp"
#include "optimizers/fused_adam.hpp"
#include "optimizers/scheduler.hpp"
#include <cstring>
#include <cuda_runtime.h>

namespace gs::training {

    void initialize_gaussians_new(gs::SplatDataNew& splat_data) {
        // Ensure all gradient tensors are allocated
        splat_data.ensure_grad_allocated();
    }

    std::unique_ptr<FusedAdam> create_optimizer_new(
        gs::SplatDataNew& splat_data,
        const gs::param::OptimizationParameters& params) {

        std::vector<FusedAdam::ParamGroup> groups;

        // Create a temporary FusedAdam to get param IDs
        FusedAdam temp_adam({}, FusedAdam::Options{});

        // Helper to create a raw param
        auto create_raw_param = [&temp_adam](float* data_ptr, float* grad_ptr, size_t num_elements) {
            return FusedAdam::RawParam{
                .data_ptr = data_ptr,
                .grad_ptr = grad_ptr,
                .num_elements = num_elements,
                .param_id = temp_adam.get_next_param_id()};
        };

        // Create groups with raw pointers from SplatDataNew
        auto add_param_group = [&groups, &create_raw_param](
                                   float* data_ptr, float* grad_ptr, size_t num_elements, double lr) {
            FusedAdam::Options options;
            options.lr = lr;
            options.eps = 1e-15;
            options.betas = {0.9, 0.999};

            std::vector<FusedAdam::RawParam> params;
            params.push_back(create_raw_param(data_ptr, grad_ptr, num_elements));
            groups.emplace_back(std::move(params), options);
        };

        // Add parameter groups with raw pointers (order: means, sh0, shN, scaling, rotation, opacity)
        add_param_group(
            splat_data.means_raw().ptr<float>(),
            splat_data.means_grad_ptr(),
            splat_data.means_raw().numel(),
            params.means_lr * splat_data.get_scene_scale());

        add_param_group(
            splat_data.sh0_raw().ptr<float>(),
            splat_data.sh0_grad_ptr(),
            splat_data.sh0_raw().numel(),
            params.shs_lr);

        add_param_group(
            splat_data.shN_raw().ptr<float>(),
            splat_data.shN_grad_ptr(),
            splat_data.shN_raw().numel(),
            params.shs_lr / 20.0);

        add_param_group(
            splat_data.scaling_raw().ptr<float>(),
            splat_data.scaling_grad_ptr(),
            splat_data.scaling_raw().numel(),
            params.scaling_lr);

        add_param_group(
            splat_data.rotation_raw().ptr<float>(),
            splat_data.rotation_grad_ptr(),
            splat_data.rotation_raw().numel(),
            params.rotation_lr);

        add_param_group(
            splat_data.opacity_raw().ptr<float>(),
            splat_data.opacity_grad_ptr(),
            splat_data.opacity_raw().numel(),
            params.opacity_lr);

        FusedAdam::Options global_options;
        global_options.lr = 0.0;
        global_options.eps = 1e-15;
        return std::make_unique<FusedAdam>(std::move(groups), global_options);
    }

    std::unique_ptr<ExponentialLR> create_scheduler_new(
        const gs::param::OptimizationParameters& params,
        FusedAdam* optimizer,
        int param_group_index) {

        // Calculate gamma for exponential decay
        // We want lr to decay from initial to 0.01 * initial over params.iterations
        const double gamma = std::pow(0.01, 1.0 / params.iterations);

        return std::make_unique<ExponentialLR>(*optimizer, gamma, param_group_index);
    }

} // namespace gs::training
