/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "strategy_utils.hpp"
#include "optimizers/fused_adam.hpp"

namespace gs::training {
    void initialize_gaussians(gs::SplatData& splat_data) {
        const auto dev = torch::kCUDA;

        // SplatData is already on CUDA (raw memory), but we need to ensure gradients are allocated
        splat_data.ensure_grad_allocated();

        // Allocate densification info if needed
        if (!splat_data._densification_info.defined() || splat_data._densification_info.numel() == 0) {
            splat_data._densification_info = torch::zeros({2, splat_data.size()},
                                                         torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        }
    }

    std::unique_ptr<FusedAdam> create_optimizer(
        gs::SplatData& splat_data,
        const gs::param::OptimizationParameters& params) {
        std::vector<FusedAdam::ParamGroup> groups;

        // Create temporary tensors for optimizer initialization
        // These are cloned tensors that own their memory
        torch::Tensor means_tensor = splat_data.means();
        torch::Tensor sh0_tensor = splat_data.sh0();
        torch::Tensor shN_tensor = splat_data.shN();
        torch::Tensor scaling_tensor = splat_data.scaling_raw();
        torch::Tensor rotation_tensor = splat_data.rotation_raw();
        torch::Tensor opacity_tensor = splat_data.opacity_raw();

        // CRITICAL: Set requires_grad so optimizer tracks them
        means_tensor.set_requires_grad(true);
        sh0_tensor.set_requires_grad(true);
        shN_tensor.set_requires_grad(true);
        scaling_tensor.set_requires_grad(true);
        rotation_tensor.set_requires_grad(true);
        opacity_tensor.set_requires_grad(true);

        // Create groups with proper Options
        auto add_param_group = [&groups](torch::Tensor param, double lr) {
            FusedAdam::Options options;
            options.lr = lr;
            options.eps = 1e-15;
            options.betas = {0.9, 0.999};
            groups.emplace_back(std::vector<torch::Tensor>{std::move(param)}, options);
        };

        add_param_group(means_tensor, params.means_lr * splat_data.get_scene_scale());
        add_param_group(sh0_tensor, params.shs_lr);
        add_param_group(shN_tensor, params.shs_lr / 20.f);
        add_param_group(scaling_tensor, params.scaling_lr);
        add_param_group(rotation_tensor, params.rotation_lr);
        add_param_group(opacity_tensor, params.opacity_lr);

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

    // New function to sync gradients from raw memory to optimizer tensors
    void sync_gradients_to_optimizer(gs::SplatData& splat_data, FusedAdam& optimizer) {
        // Get the optimizer's parameter tensors
        auto& means_param = optimizer.param_groups()[0].params[0];
        auto& sh0_param = optimizer.param_groups()[1].params[0];
        auto& shN_param = optimizer.param_groups()[2].params[0];
        auto& scaling_param = optimizer.param_groups()[3].params[0];
        auto& rotation_param = optimizer.param_groups()[4].params[0];
        auto& opacity_param = optimizer.param_groups()[5].params[0];

        // Ensure they have gradient tensors allocated
        if (!means_param.grad().defined()) {
            means_param.mutable_grad() = torch::zeros_like(means_param);
        }
        if (!sh0_param.grad().defined()) {
            sh0_param.mutable_grad() = torch::zeros_like(sh0_param);
        }
        if (!shN_param.grad().defined()) {
            shN_param.mutable_grad() = torch::zeros_like(shN_param);
        }
        if (!scaling_param.grad().defined()) {
            scaling_param.mutable_grad() = torch::zeros_like(scaling_param);
        }
        if (!rotation_param.grad().defined()) {
            rotation_param.mutable_grad() = torch::zeros_like(rotation_param);
        }
        if (!opacity_param.grad().defined()) {
            opacity_param.mutable_grad() = torch::zeros_like(opacity_param);
        }

        // Copy gradients from raw memory to tensor gradients
        cudaMemcpy(means_param.mutable_grad().data_ptr<float>(),
                  splat_data.means_grad_cuda_ptr(),
                  means_param.numel() * sizeof(float),
                  cudaMemcpyDeviceToDevice);

        cudaMemcpy(sh0_param.mutable_grad().data_ptr<float>(),
                  splat_data.sh0_grad_cuda_ptr(),
                  sh0_param.numel() * sizeof(float),
                  cudaMemcpyDeviceToDevice);

        cudaMemcpy(shN_param.mutable_grad().data_ptr<float>(),
                  splat_data.shN_grad_cuda_ptr(),
                  shN_param.numel() * sizeof(float),
                  cudaMemcpyDeviceToDevice);

        cudaMemcpy(scaling_param.mutable_grad().data_ptr<float>(),
                  splat_data.scaling_grad_cuda_ptr(),
                  scaling_param.numel() * sizeof(float),
                  cudaMemcpyDeviceToDevice);

        cudaMemcpy(rotation_param.mutable_grad().data_ptr<float>(),
                  splat_data.rotation_grad_cuda_ptr(),
                  rotation_param.numel() * sizeof(float),
                  cudaMemcpyDeviceToDevice);

        cudaMemcpy(opacity_param.mutable_grad().data_ptr<float>(),
                  splat_data.opacity_grad_cuda_ptr(),
                  opacity_param.numel() * sizeof(float),
                  cudaMemcpyDeviceToDevice);
    }

    // New function to sync parameters from optimizer back to raw memory
    void sync_parameters_from_optimizer(gs::SplatData& splat_data, FusedAdam& optimizer) {
        // Get the updated parameters from optimizer
        auto& means_param = optimizer.param_groups()[0].params[0];
        auto& sh0_param = optimizer.param_groups()[1].params[0];
        auto& shN_param = optimizer.param_groups()[2].params[0];
        auto& scaling_param = optimizer.param_groups()[3].params[0];
        auto& rotation_param = optimizer.param_groups()[4].params[0];
        auto& opacity_param = optimizer.param_groups()[5].params[0];

        // Copy updated parameters back to raw memory
        cudaMemcpy(splat_data.means_cuda_ptr(),
                  means_param.contiguous().data_ptr<float>(),
                  means_param.numel() * sizeof(float),
                  cudaMemcpyDeviceToDevice);

        cudaMemcpy(splat_data.sh0_cuda_ptr(),
                  sh0_param.contiguous().data_ptr<float>(),
                  sh0_param.numel() * sizeof(float),
                  cudaMemcpyDeviceToDevice);

        cudaMemcpy(splat_data.shN_cuda_ptr(),
                  shN_param.contiguous().data_ptr<float>(),
                  shN_param.numel() * sizeof(float),
                  cudaMemcpyDeviceToDevice);

        cudaMemcpy(splat_data.scaling_raw_cuda_ptr(),
                  scaling_param.contiguous().data_ptr<float>(),
                  scaling_param.numel() * sizeof(float),
                  cudaMemcpyDeviceToDevice);

        cudaMemcpy(splat_data.rotation_raw_cuda_ptr(),
                  rotation_param.contiguous().data_ptr<float>(),
                  rotation_param.numel() * sizeof(float),
                  cudaMemcpyDeviceToDevice);

        cudaMemcpy(splat_data.opacity_raw_cuda_ptr(),
                  opacity_param.contiguous().data_ptr<float>(),
                  opacity_param.numel() * sizeof(float),
                  cudaMemcpyDeviceToDevice);
    }

    void update_param_with_optimizer(
        const ParamUpdateFn& param_fn,
        const OptimizerUpdateFn& optimizer_fn,
        std::unique_ptr<FusedAdam>& optimizer,
        gs::SplatData& splat_data,
        std::vector<size_t> param_idxs) {

        // Get current parameters from optimizer (not from SplatData!)
        std::array<torch::Tensor*, 6> params;
        for (size_t i = 0; i < 6; ++i) {
            params[i] = &optimizer->param_groups()[i].params[0];
        }

        std::array<torch::Tensor, 6> new_params;

        // Collect old parameter keys and states
        std::vector<void*> old_param_keys;
        std::array<std::unique_ptr<FusedAdam::AdamState>, 6> saved_states;

        for (auto i : param_idxs) {
            auto& param = *params[i];
            auto new_param = param_fn(i, param);
            new_params[i] = new_param;

            void* old_param_key = param.unsafeGetTensorImpl();
            old_param_keys.push_back(old_param_key);

            // Check if state exists
            auto state_it = optimizer->state().find(old_param_key);
            if (state_it != optimizer->state().end()) {
                auto* adam_state = state_it->second.get();
                auto new_state = optimizer_fn(*adam_state, new_param);
                saved_states[i] = std::move(new_state);
            } else {
                saved_states[i] = nullptr;
            }
        }

        // Now remove all old states
        for (auto key : old_param_keys) {
            optimizer->state().erase(key);
        }

        // Update parameters and add new states
        for (auto i : param_idxs) {
            optimizer->param_groups()[i].params[0] = new_params[i];

            if (saved_states[i]) {
                void* new_param_key = new_params[i].unsafeGetTensorImpl();
                optimizer->state()[new_param_key] = std::move(saved_states[i]);
            }
        }

        // Update SplatData's raw memory with the new data
        for (auto i : param_idxs) {
            if (i == 0 && new_params[i].defined()) {
                cudaMemcpy(splat_data.means_cuda_ptr(),
                          new_params[i].contiguous().data_ptr<float>(),
                          new_params[i].numel() * sizeof(float),
                          cudaMemcpyDeviceToDevice);
            } else if (i == 1 && new_params[i].defined()) {
                cudaMemcpy(splat_data.sh0_cuda_ptr(),
                          new_params[i].contiguous().data_ptr<float>(),
                          new_params[i].numel() * sizeof(float),
                          cudaMemcpyDeviceToDevice);
            } else if (i == 2 && new_params[i].defined()) {
                cudaMemcpy(splat_data.shN_cuda_ptr(),
                          new_params[i].contiguous().data_ptr<float>(),
                          new_params[i].numel() * sizeof(float),
                          cudaMemcpyDeviceToDevice);
            } else if (i == 3 && new_params[i].defined()) {
                cudaMemcpy(splat_data.scaling_raw_cuda_ptr(),
                          new_params[i].contiguous().data_ptr<float>(),
                          new_params[i].numel() * sizeof(float),
                          cudaMemcpyDeviceToDevice);
            } else if (i == 4 && new_params[i].defined()) {
                cudaMemcpy(splat_data.rotation_raw_cuda_ptr(),
                          new_params[i].contiguous().data_ptr<float>(),
                          new_params[i].numel() * sizeof(float),
                          cudaMemcpyDeviceToDevice);
            } else if (i == 5 && new_params[i].defined()) {
                cudaMemcpy(splat_data.opacity_raw_cuda_ptr(),
                          new_params[i].contiguous().data_ptr<float>(),
                          new_params[i].numel() * sizeof(float),
                          cudaMemcpyDeviceToDevice);
            }
        }

        // Ensure gradients are still allocated after parameter update
        splat_data.ensure_grad_allocated();
    }
} // namespace gs::training