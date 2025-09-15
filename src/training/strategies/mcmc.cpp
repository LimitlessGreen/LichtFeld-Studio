/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcmc.hpp"
#include "Ops.h"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "optimizers/fused_adam.hpp"
#include "optimizers/scheduler.hpp"
#include "rasterization/rasterizer.hpp"
#include "strategy_utils.hpp"
#include <iostream>
#include <random>

#ifdef _WIN32
#include <c10/cuda/CUDACachingAllocator.h> //required for emptyCache
#endif

namespace gs::training {
    MCMC::MCMC(gs::SplatData&& splat_data)
        : _splat_data(std::move(splat_data)) {
    }

    torch::Tensor MCMC::multinomial_sample(const torch::Tensor& weights, int n, bool replacement) {
        const int64_t num_elements = weights.size(0);

        // PyTorch's multinomial has a limit of 2^24 elements
        if (num_elements <= (1 << 24)) {
            return torch::multinomial(weights, n, replacement);
        }
        // For larger arrays, we need to implement sampling manually
        auto weights_normalized = weights / weights.sum();
        auto weights_cpu = weights_normalized.cpu();

        std::vector<int64_t> sampled_indices;
        sampled_indices.reserve(n);

        // Create cumulative distribution
        auto cumsum = weights_cpu.cumsum(0);
        auto cumsum_data = cumsum.accessor<float, 1>();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);

        for (int i = 0; i < n; ++i) {
            float u = dis(gen);
            // Binary search for the index
            int64_t idx = 0;
            int64_t left = 0, right = num_elements - 1;
            while (left <= right) {
                int64_t mid = (left + right) / 2;
                if (cumsum_data[mid] < u) {
                    left = mid + 1;
                } else {
                    idx = mid;
                    right = mid - 1;
                }
            }
            sampled_indices.push_back(idx);
        }

        auto result = torch::tensor(sampled_indices, torch::kLong);
        return result.to(weights.device());
    }

    void MCMC::update_optimizer_for_relocate(FusedAdam* optimizer,
                                             const torch::Tensor& sampled_indices,
                                             const torch::Tensor& dead_indices,
                                             int param_position) {
        // Get the parameter from the param group
        auto& param = optimizer->param_groups()[param_position].params[0];
        int param_id = param.param_id;

        // Check if optimizer state exists
        auto state_it = optimizer->state().find(param_id);
        if (state_it == optimizer->state().end()) {
            // No state exists yet - this can happen if optimizer.step() hasn't been called
            // In this case, there's nothing to reset, so we can safely return
            return;
        }

        // Get the optimizer state - now we need to handle raw memory
        auto& adam_state = *state_it->second;

        // Since we have raw pointers, we need to manually set the values at sampled_indices to 0
        // We'll need to use CUDA kernels for this
        // For now, we'll just skip this optimization as relocation happens infrequently
        // The optimizer will naturally adapt to the new parameters

        // Note: In a production system, you'd want to add a CUDA kernel to zero out
        // specific indices in the exp_avg and exp_avg_sq arrays
    }

    int MCMC::relocate_gs() {
        // Get opacities and handle both [N] and [N, 1] shapes
        torch::NoGradGuard no_grad;
        auto opacities = _splat_data.get_opacity();
        if (opacities.dim() == 2 && opacities.size(1) == 1) {
            opacities = opacities.squeeze(-1);
        }

        auto rotation_raw = _splat_data.rotation_raw();
        auto dead_mask = opacities <= _params->min_opacity | (rotation_raw * rotation_raw).sum(-1) < 1e-8f;
        auto dead_indices = dead_mask.nonzero().squeeze(-1);
        int n_dead = dead_indices.numel();

        if (n_dead == 0)
            return 0;

        auto alive_mask = ~dead_mask;
        auto alive_indices = alive_mask.nonzero().squeeze(-1);

        if (alive_indices.numel() == 0)
            return 0;

        // Sample from alive Gaussians based on opacity
        auto probs = opacities.index_select(0, alive_indices);
        auto sampled_idxs_local = multinomial_sample(probs, n_dead, true);
        auto sampled_idxs = alive_indices.index_select(0, sampled_idxs_local);

        // Get parameters for sampled Gaussians
        auto sampled_opacities = opacities.index_select(0, sampled_idxs);
        auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

        // Count occurrences of each sampled index
        auto ratios = torch::ones_like(opacities, torch::kInt32);
        ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kInt32));
        ratios = ratios.index_select(0, sampled_idxs).contiguous();

        // IMPORTANT: Clamp as in Python implementation
        const int n_max = static_cast<int>(_binoms.size(0));
        ratios = torch::clamp_max_(ratios, n_max);

        // Call the CUDA relocation function from gsplat
        auto relocation_result = gsplat::relocation(
            sampled_opacities,
            sampled_scales,
            ratios,
            _binoms,
            n_max);

        auto new_opacities = std::get<0>(relocation_result);
        auto new_scales = std::get<1>(relocation_result);

        // Clamp new opacities
        new_opacities = torch::clamp_(new_opacities, _params->min_opacity, 1.0f - 1e-7f);

        // Update parameters for sampled indices
        // Handle opacity shape properly
        if (_splat_data.opacity_raw().dim() == 2) {
            _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                                 torch::logit(new_opacities).unsqueeze(-1));
        } else {
            _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
        }
        _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

        // Copy from sampled to dead indices
        _splat_data.means().index_put_({dead_indices}, _splat_data.means().index_select(0, sampled_idxs));
        _splat_data.sh0().index_put_({dead_indices}, _splat_data.sh0().index_select(0, sampled_idxs));
        _splat_data.shN().index_put_({dead_indices}, _splat_data.shN().index_select(0, sampled_idxs));
        _splat_data.scaling_raw().index_put_({dead_indices}, _splat_data.scaling_raw().index_select(0, sampled_idxs));
        _splat_data.rotation_raw().index_put_({dead_indices}, _splat_data.rotation_raw().index_select(0, sampled_idxs));
        _splat_data.opacity_raw().index_put_({dead_indices}, _splat_data.opacity_raw().index_select(0, sampled_idxs));

        // Update optimizer states for sampled indices
        for (int i = 0; i < 6; ++i) {
            update_optimizer_for_relocate(_optimizer.get(), sampled_idxs, dead_indices, i);
        }

        return n_dead;
    }

    int MCMC::add_new_gs() {
        // Add this check at the beginning
        torch::NoGradGuard no_grad;
        if (!_optimizer) {
            std::cerr << "Warning: add_new_gs called but optimizer not initialized" << std::endl;
            return 0;
        }

        const int current_n = _splat_data.size();
        const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
        const int n_new = std::max(0, n_target - current_n);

        if (n_new == 0)
            return 0;

        // Get opacities and handle both [N] and [N, 1] shapes
        auto opacities = _splat_data.get_opacity();
        if (opacities.dim() == 2 && opacities.size(1) == 1) {
            opacities = opacities.squeeze(-1);
        }

        auto probs = opacities.flatten();
        auto sampled_idxs = multinomial_sample(probs, n_new, true);

        // Get parameters for sampled Gaussians
        auto sampled_opacities = opacities.index_select(0, sampled_idxs);
        auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

        // Count occurrences
        auto ratios = torch::zeros({opacities.size(0)}, torch::kFloat32).to(torch::kCUDA);
        ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kFloat32));
        ratios = ratios.index_select(0, sampled_idxs) + 1;

        // IMPORTANT: Clamp and convert to int as in Python implementation
        const int n_max = static_cast<int>(_binoms.size(0));
        ratios = torch::clamp(ratios, 1, n_max);
        ratios = ratios.to(torch::kInt32).contiguous(); // Convert to int!

        // Call the CUDA relocation function from gsplat
        auto relocation_result = gsplat::relocation(
            sampled_opacities,
            sampled_scales,
            ratios,
            _binoms,
            n_max);

        auto new_opacities = std::get<0>(relocation_result);
        auto new_scales = std::get<1>(relocation_result);

        // Clamp new opacities
        new_opacities = torch::clamp(new_opacities, _params->min_opacity, 1.0f - 1e-7f);

        // Update existing Gaussians FIRST (before concatenation)
        if (_splat_data.opacity_raw().dim() == 2) {
            _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                                 torch::logit(new_opacities).unsqueeze(-1));
        } else {
            _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
        }
        _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

        // Prepare new Gaussians to concatenate
        auto new_means = _splat_data.means().index_select(0, sampled_idxs);
        auto new_sh0 = _splat_data.sh0().index_select(0, sampled_idxs);
        auto new_shN = _splat_data.shN().index_select(0, sampled_idxs);
        auto new_scaling = _splat_data.scaling_raw().index_select(0, sampled_idxs);
        auto new_rotation = _splat_data.rotation_raw().index_select(0, sampled_idxs);
        auto new_opacity = _splat_data.opacity_raw().index_select(0, sampled_idxs);

        // Step 1: Concatenate all parameters (without requires_grad)
        auto concat_means = torch::cat({_splat_data.means(), new_means}, 0);
        auto concat_sh0 = torch::cat({_splat_data.sh0(), new_sh0}, 0);
        auto concat_shN = torch::cat({_splat_data.shN(), new_shN}, 0);
        auto concat_scaling = torch::cat({_splat_data.scaling_raw(), new_scaling}, 0);
        auto concat_rotation = torch::cat({_splat_data.rotation_raw(), new_rotation}, 0);
        auto concat_opacity = torch::cat({_splat_data.opacity_raw(), new_opacity}, 0);

        // Step 2: Update the model's parameters
        _splat_data.means() = concat_means;
        _splat_data.sh0() = concat_sh0;
        _splat_data.shN() = concat_shN;
        _splat_data.scaling_raw() = concat_scaling;
        _splat_data.rotation_raw() = concat_rotation;
        _splat_data.opacity_raw() = concat_opacity;

        // Ensure gradients are allocated for new parameters
        _splat_data.ensure_grad_allocated();

        // Step 3: Recreate optimizer with new sizes
        // Store old learning rates
        std::vector<double> old_lrs;
        for (const auto& group : _optimizer->param_groups()) {
            old_lrs.push_back(group.options.lr);
        }

        // Create new optimizer with updated pointers and sizes
        _optimizer = create_optimizer(_splat_data, *_params);

        // Restore learning rates
        for (size_t i = 0; i < old_lrs.size() && i < _optimizer->param_groups().size(); ++i) {
            _optimizer->set_lr(old_lrs[i], i);
        }

        noise_buffer_ = torch::empty_like(_splat_data.means());
        return n_new;
    }

    void MCMC::inject_noise() {
        torch::NoGradGuard no_grad;

        // Get current learning rate from optimizer (after scheduler has updated it)
        double current_lr = _optimizer->get_lr(0) * _noise_lr;

        // Generate noise
        noise_buffer_.normal_(0.0f, 1.0f);

        gsplat::add_noise(
            _splat_data.opacity_raw(),
            _splat_data.scaling_raw(),
            _splat_data.rotation_raw(),
            noise_buffer_,
            _splat_data.means(),
            static_cast<float>(current_lr));
    }

    void MCMC::post_backward(int iter, RenderOutput& render_output) {
        // Increment SH degree every 1000 iterations
        torch::NoGradGuard no_grad;
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data.increment_sh_degree();
        }

        // Refine Gaussians
        if (is_refining(iter)) {
            // Relocate dead Gaussians
            relocate_gs();

            // Add new Gaussians
            add_new_gs();
        }

        // Inject noise to positions
        inject_noise();

#ifdef _WIN32
        // Windows doesn't support CUDACachingAllocator expandable_segments
        if (iter % 10 == 0)
            c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    }

    void MCMC::step(int iter) {
        if (iter < _params->iterations) {
            _optimizer->step(iter);
            // Use manual zero_grad instead of optimizer's zero_grad
            _splat_data.zero_grad_manual();
            _scheduler->step();
        }
    }

    void MCMC::remove_gaussians(const torch::Tensor& mask) {
        torch::NoGradGuard no_grad;

        if (mask.sum().item<int>() == 0) {
            LOG_DEBUG("No Gaussians to remove");
            return;
        }

        LOG_DEBUG("MCMC: Removing {} Gaussians", mask.sum().item<int>());

        const torch::Tensor sampled_idxs = mask.logical_not().nonzero().squeeze(-1);

        const auto param_fn = [&sampled_idxs](const int i, const torch::Tensor& param) {
            // Don't set requires_grad anymore
            return param.index_select(0, sampled_idxs);
        };

        const auto optimizer_fn = [&sampled_idxs](
                                      FusedAdam::AdamState& state,
                                      const torch::Tensor& new_param)
            -> std::unique_ptr<FusedAdam::AdamState> {
            // With raw memory, we can't easily subset the state
            // Since this happens infrequently, we'll just reset the state
            // The optimizer will quickly adapt
            return nullptr;
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);
        noise_buffer_ = torch::empty_like(_splat_data.means());
    }

    void MCMC::initialize(const gs::param::OptimizationParameters& optimParams) {
        // DISABLE AUTOGRAD
        torch::NoGradGuard no_grad;

        _params = std::make_unique<const gs::param::OptimizationParameters>(optimParams);

        const auto dev = torch::kCUDA;
        // Move to CUDA WITHOUT setting requires_grad
        _splat_data.means() = _splat_data.means().to(dev);
        _splat_data.scaling_raw() = _splat_data.scaling_raw().to(dev);
        _splat_data.rotation_raw() = _splat_data.rotation_raw().to(dev);
        _splat_data.opacity_raw() = _splat_data.opacity_raw().to(dev);
        _splat_data.sh0() = _splat_data.sh0().to(dev);
        _splat_data.shN() = _splat_data.shN().to(dev);
        _splat_data._densification_info = torch::empty({0});
        noise_buffer_ = torch::empty_like(_splat_data.means());

        // Pre-allocate gradients (without autograd)
        _splat_data.ensure_grad_allocated();

        // Initialize binomial coefficients
        const int n_max = 51;
        _binoms = torch::zeros({n_max, n_max}, torch::kFloat32);
        auto binoms_accessor = _binoms.accessor<float, 2>();
        for (int n = 0; n < n_max; ++n) {
            for (int k = 0; k <= n; ++k) {
                float binom = 1.0f;
                for (int i = 0; i < k; ++i) {
                    binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
                }
                binoms_accessor[n][k] = binom;
            }
        }
        _binoms = _binoms.to(dev);

        // Initialize optimizer using the helper function
        _optimizer = create_optimizer(_splat_data, *_params);

        const double gamma = std::pow(0.01, 1.0 / _params->iterations);
        _scheduler = std::make_unique<ExponentialLR>(*_optimizer, gamma, 0);
    }

    bool MCMC::is_refining(int iter) const {
        return (iter < _params->stop_refine &&
                iter > _params->start_refine &&
                iter % _params->refine_every == 0);
    }
} // namespace gs::training