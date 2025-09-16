/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "strategy_utils.hpp"
#include "optimizers/fused_adam.hpp"
#include <cstring>
#include <cuda_runtime.h>

namespace gs::training {
    void initialize_gaussians(gs::SplatData& splat_data) {
        // SplatData is already on CUDA (raw memory), but we need to ensure gradients are allocated
        splat_data.ensure_grad_allocated();

        // Allocate densification info if needed - this still uses torch for now
        // but it's only used during densification which is currently disabled
        if (!splat_data._densification_info.defined() || splat_data._densification_info.numel() == 0) {
            const auto dev = torch::kCUDA;
            splat_data._densification_info = torch::zeros({2, splat_data.size()},
                                                          torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        }
    }

    std::unique_ptr<FusedAdam> create_optimizer(
        gs::SplatData& splat_data,
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

        // Create groups with raw pointers from SplatData
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

        // Add parameter groups with raw pointers
        add_param_group(
            splat_data.means_cuda_ptr(),
            splat_data.means_grad_cuda_ptr(),
            splat_data.means().numel(),
            params.means_lr * splat_data.get_scene_scale());

        add_param_group(
            splat_data.sh0_cuda_ptr(),
            splat_data.sh0_grad_cuda_ptr(),
            splat_data.sh0().numel(),
            params.shs_lr);

        add_param_group(
            splat_data.shN_cuda_ptr(),
            splat_data.shN_grad_cuda_ptr(),
            splat_data.shN().numel(),
            params.shs_lr / 20.f);

        add_param_group(
            splat_data.scaling_raw_cuda_ptr(),
            splat_data.scaling_grad_cuda_ptr(),
            splat_data.scaling_raw().numel(),
            params.scaling_lr);

        add_param_group(
            splat_data.rotation_raw_cuda_ptr(),
            splat_data.rotation_grad_cuda_ptr(),
            splat_data.rotation_raw().numel(),
            params.rotation_lr);

        add_param_group(
            splat_data.opacity_raw_cuda_ptr(),
            splat_data.opacity_grad_cuda_ptr(),
            splat_data.opacity_raw().numel(),
            params.opacity_lr);

        FusedAdam::Options global_options;
        global_options.lr = 0.f;
        global_options.eps = 1e-15;
        return std::make_unique<FusedAdam>(std::move(groups), global_options);
    }

    std::unique_ptr<ExponentialLR> create_scheduler(
        const gs::param::OptimizationParameters& params,
        FusedAdam* optimizer,
        int param_group_index) {
        // Python: gamma = 0.01^(1/max_steps)
        // This means after max_steps, lr will be 0.01 * initial_lr
        const double gamma = std::pow(0.01, 1.0 / params.iterations);
        return std::make_unique<ExponentialLR>(*optimizer, gamma, param_group_index);
    }

    void update_param_with_optimizer(
        const ParamUpdateFn& param_fn,
        const OptimizerUpdateFn& optimizer_fn,
        std::unique_ptr<FusedAdam>& optimizer,
        gs::SplatData& splat_data,
        std::vector<size_t> param_idxs) {

        // For now, we need to recreate the optimizer when parameters change size
        // This is because we're using raw memory and need to update all pointers

        // First, apply the parameter transformations using torch tensors
        // (this is still needed for the complex index operations in densification)
        std::array<torch::Tensor*, 6> params;
        params[0] = &splat_data.means();
        params[1] = &splat_data.sh0();
        params[2] = &splat_data.shN();
        params[3] = &splat_data.scaling_raw();
        params[4] = &splat_data.rotation_raw();
        params[5] = &splat_data.opacity_raw();

        // Store old learning rates before we lose the optimizer
        std::vector<double> old_lrs;
        for (const auto& group : optimizer->param_groups()) {
            old_lrs.push_back(group.options.lr);
        }

        // Apply transformations and update tensors
        for (auto i : param_idxs) {
            auto& param = *params[i];
            auto new_param = param_fn(i, param);
            if (new_param.defined()) {
                // Update the tensor in SplatData
                *params[i] = new_param.contiguous();

                // CRITICAL: After updating the tensor, we need to update the raw CUDA pointers
                // The tensor's data_ptr() may have changed after concatenation/selection
                if (i == 0) {
                    // Update means pointers
                    cudaMemcpy(splat_data.means_cuda_ptr(),
                               params[i]->data_ptr<float>(),
                               params[i]->numel() * sizeof(float),
                               cudaMemcpyDeviceToDevice);
                } else if (i == 1) {
                    // Update sh0 pointers
                    cudaMemcpy(splat_data.sh0_cuda_ptr(),
                               params[i]->data_ptr<float>(),
                               params[i]->numel() * sizeof(float),
                               cudaMemcpyDeviceToDevice);
                } else if (i == 2) {
                    // Update shN pointers
                    cudaMemcpy(splat_data.shN_cuda_ptr(),
                               params[i]->data_ptr<float>(),
                               params[i]->numel() * sizeof(float),
                               cudaMemcpyDeviceToDevice);
                } else if (i == 3) {
                    // Update scaling pointers
                    cudaMemcpy(splat_data.scaling_raw_cuda_ptr(),
                               params[i]->data_ptr<float>(),
                               params[i]->numel() * sizeof(float),
                               cudaMemcpyDeviceToDevice);
                } else if (i == 4) {
                    // Update rotation pointers
                    cudaMemcpy(splat_data.rotation_raw_cuda_ptr(),
                               params[i]->data_ptr<float>(),
                               params[i]->numel() * sizeof(float),
                               cudaMemcpyDeviceToDevice);
                } else if (i == 5) {
                    // Update opacity pointers
                    cudaMemcpy(splat_data.opacity_raw_cuda_ptr(),
                               params[i]->data_ptr<float>(),
                               params[i]->numel() * sizeof(float),
                               cudaMemcpyDeviceToDevice);
                }
            }
        }

        // Ensure gradients are allocated for new size
        splat_data.ensure_grad_allocated();

        // IMPORTANT: Synchronize CUDA to ensure all memory operations complete
        cudaDeviceSynchronize();

        // Now recreate the optimizer with the updated pointers
        std::vector<FusedAdam::ParamGroup> groups;

        // Helper to create a raw param with incrementing IDs
        int param_id = 0;
        auto create_raw_param = [&param_id](float* data_ptr, float* grad_ptr, size_t num_elements) {
            return FusedAdam::RawParam{
                .data_ptr = data_ptr,
                .grad_ptr = grad_ptr,
                .num_elements = num_elements,
                .param_id = param_id++};
        };

        // Recreate groups with new raw pointers - these should now be valid
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

        // Add parameter groups with restored learning rates
        add_param_group(
            splat_data.means_cuda_ptr(),
            splat_data.means_grad_cuda_ptr(),
            splat_data.means().numel(),
            old_lrs[0]);

        add_param_group(
            splat_data.sh0_cuda_ptr(),
            splat_data.sh0_grad_cuda_ptr(),
            splat_data.sh0().numel(),
            old_lrs[1]);

        add_param_group(
            splat_data.shN_cuda_ptr(),
            splat_data.shN_grad_cuda_ptr(),
            splat_data.shN().numel(),
            old_lrs[2]);

        add_param_group(
            splat_data.scaling_raw_cuda_ptr(),
            splat_data.scaling_grad_cuda_ptr(),
            splat_data.scaling_raw().numel(),
            old_lrs[3]);

        add_param_group(
            splat_data.rotation_raw_cuda_ptr(),
            splat_data.rotation_grad_cuda_ptr(),
            splat_data.rotation_raw().numel(),
            old_lrs[4]);

        add_param_group(
            splat_data.opacity_raw_cuda_ptr(),
            splat_data.opacity_grad_cuda_ptr(),
            splat_data.opacity_raw().numel(),
            old_lrs[5]);

        FusedAdam::Options global_options;
        global_options.lr = 0.f;
        global_options.eps = 1e-15;

        // Replace the optimizer
        optimizer = std::make_unique<FusedAdam>(std::move(groups), global_options);

        // Note: We lose the optimizer state here, but that's acceptable for densification
        // as it happens infrequently and the state will rebuild quickly
    }
} // namespace gs::training