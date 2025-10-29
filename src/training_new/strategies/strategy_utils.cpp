/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "strategy_utils.hpp"
#include "core_new/logger.hpp"

namespace lfs::training {

    void initialize_gaussians(lfs::core::SplatData& splat_data) {
        // Tensors are already on GPU in the new framework (created with Device::CUDA by default)
        // Just need to ensure gradients are allocated
        if (!splat_data.has_gradients()) {
            splat_data.allocate_gradients();
        }
    }

    std::unique_ptr<AdamOptimizer> create_optimizer(
        lfs::core::SplatData& splat_data,
        const lfs::core::param::OptimizationParameters& params) {

        // Create Adam config with per-parameter learning rates
        // Note: In the new optimizer, we'll need to set LRs per parameter type
        // For now, we'll create with the means LR and update individual params later
        AdamConfig config;
        config.lr = params.means_lr * splat_data.get_scene_scale();
        config.beta1 = 0.9f;
        config.beta2 = 0.999f;
        config.eps = 1e-15f;

        auto optimizer = std::make_unique<AdamOptimizer>(splat_data, config);

        // TODO: Set per-parameter learning rates
        // This will require extending AdamOptimizer to support per-parameter LRs
        // For now, the AdamOptimizer applies the same LR to all parameters

        return optimizer;
    }

    std::unique_ptr<ExponentialLR> create_scheduler(
        const lfs::core::param::OptimizationParameters& params,
        AdamOptimizer& optimizer) {

        // Python: gamma = 0.01^(1/max_steps)
        // This means after max_steps, lr will be 0.01 * initial_lr
        const double gamma = std::pow(0.01, 1.0 / params.iterations);
        return std::make_unique<ExponentialLR>(optimizer, gamma);
    }

    void update_param_with_optimizer(
        const ParamUpdateFn& param_fn,
        const OptimizerUpdateFn& optimizer_fn,
        std::unique_ptr<AdamOptimizer>& optimizer,
        lfs::core::SplatData& splat_data,
        std::vector<size_t> param_idxs) {

        // CRITICAL: Ensure CUDA device is set for this thread
        // Some operations might spawn TBB threads, and those need CUDA context
        cudaSetDevice(0);

        // Map param index to ParamType
        auto index_to_param_type = [](size_t idx) -> ParamType {
            switch (idx) {
            case 0: return ParamType::Means;
            case 1: return ParamType::Sh0;
            case 2: return ParamType::ShN;
            case 3: return ParamType::Scaling;
            case 4: return ParamType::Rotation;
            case 5: return ParamType::Opacity;
            default:
                LOG_ERROR("Invalid parameter index: {}", idx);
                return ParamType::Means;
            }
        };

        // Get references to all parameters
        std::array<lfs::core::Tensor*, 6> params = {
            &splat_data.means(),
            &splat_data.sh0(),
            &splat_data.shN(),
            &splat_data.scaling_raw(),
            &splat_data.rotation_raw(),
            &splat_data.opacity_raw()};

        std::array<lfs::core::Tensor*, 6> grads = {
            &splat_data.means_grad(),
            &splat_data.sh0_grad(),
            &splat_data.shN_grad(),
            &splat_data.scaling_grad(),
            &splat_data.rotation_grad(),
            &splat_data.opacity_grad()};

        std::array<lfs::core::Tensor, 6> new_params;

        // First pass: Compute new parameters and update optimizer state
        for (auto i : param_idxs) {
            auto param = params[i];
            cudaError_t err_before = cudaGetLastError();
            if (err_before != cudaSuccess) {
                LOG_ERROR("CUDA error before param_fn: {}", cudaGetErrorString(err_before));
            }

            auto new_param = param_fn(i, *param);

            cudaError_t err_after = cudaGetLastError();
            if (err_after != cudaSuccess) {
                LOG_ERROR("CUDA error after param_fn({}): {}", i, cudaGetErrorString(err_after));
                throw std::runtime_error(std::string("CUDA error in param_fn: ") + cudaGetErrorString(err_after));
            }
            new_params[i] = new_param;

            // Get optimizer state for this parameter type
            ParamType param_type = index_to_param_type(i);
            const AdamParamState* state = optimizer->get_state(param_type);

            if (state) {
                // Make a copy to pass to the update function
                AdamParamState state_copy = *state;
                optimizer_fn(state_copy, new_param);
                // Write the modified state back to the optimizer
                optimizer->set_state(param_type, state_copy);
            }
        }

        // Second pass: Update parameters and gradients in SplatData
        for (auto i : param_idxs) {
            if (i == 0) {
                splat_data.means() = new_params[i];
                // Update gradient tensor to match new size
                if (splat_data.has_gradients()) {
                    splat_data.means_grad() = lfs::core::Tensor::zeros(
                        new_params[i].shape(), new_params[i].device(), new_params[i].dtype());
                }
            } else if (i == 1) {
                splat_data.sh0() = new_params[i];
                if (splat_data.has_gradients()) {
                    splat_data.sh0_grad() = lfs::core::Tensor::zeros(
                        new_params[i].shape(), new_params[i].device(), new_params[i].dtype());
                }
            } else if (i == 2) {
                splat_data.shN() = new_params[i];
                if (splat_data.has_gradients()) {
                    splat_data.shN_grad() = lfs::core::Tensor::zeros(
                        new_params[i].shape(), new_params[i].device(), new_params[i].dtype());
                }
            } else if (i == 3) {
                splat_data.scaling_raw() = new_params[i];
                if (splat_data.has_gradients()) {
                    splat_data.scaling_grad() = lfs::core::Tensor::zeros(
                        new_params[i].shape(), new_params[i].device(), new_params[i].dtype());
                }
            } else if (i == 4) {
                splat_data.rotation_raw() = new_params[i];
                if (splat_data.has_gradients()) {
                    splat_data.rotation_grad() = lfs::core::Tensor::zeros(
                        new_params[i].shape(), new_params[i].device(), new_params[i].dtype());
                }
            } else if (i == 5) {
                splat_data.opacity_raw() = new_params[i];
                if (splat_data.has_gradients()) {
                    splat_data.opacity_grad() = lfs::core::Tensor::zeros(
                        new_params[i].shape(), new_params[i].device(), new_params[i].dtype());
                }
            }
        }
    }

} // namespace lfs::training
