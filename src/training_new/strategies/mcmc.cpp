/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcmc.hpp"
#include "kernels/mcmc_kernels.hpp"
#include "strategy_utils.hpp"
#include "core_new/logger.hpp"
#include <cmath>

namespace lfs::training {

    MCMC::MCMC(lfs::core::SplatData&& splat_data)
        : _splat_data(std::move(splat_data)) {
    }

    lfs::core::Tensor MCMC::multinomial_sample(const lfs::core::Tensor& weights, int n, bool replacement) {
        // Use the tensor library's built-in multinomial sampling
        return lfs::core::Tensor::multinomial(weights, n, replacement);
    }

    void MCMC::update_optimizer_for_relocate(
        const lfs::core::Tensor& sampled_indices,
        const lfs::core::Tensor& dead_indices,
        ParamType param_type) {

        // Reset optimizer state (exp_avg and exp_avg_sq) for relocated Gaussians
        // Use GPU version for efficiency (indices already on GPU)
        _optimizer->relocate_params_at_indices_gpu(
            param_type,
            sampled_indices.ptr<int64_t>(),
            sampled_indices.numel()
        );
    }

    int MCMC::relocate_gs() {
        using namespace lfs::core;

        // Get opacities (handle both [N] and [N, 1] shapes)
        Tensor opacities = _splat_data.get_opacity();
        if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
            opacities = opacities.squeeze(-1);
        }

        // Find dead Gaussians: opacity <= min_opacity OR rotation magnitude near zero
        Tensor rotation_raw = _splat_data.rotation_raw();
        Tensor rot_mag_sq = (rotation_raw * rotation_raw).sum(-1);
        Tensor dead_mask = (opacities <= _params->min_opacity).logical_or(rot_mag_sq < 1e-8f);

        Tensor dead_indices = dead_mask.nonzero().squeeze(-1);
        int n_dead = dead_indices.numel();

        if (n_dead == 0)
            return 0;

        Tensor alive_mask = dead_mask.logical_not();
        Tensor alive_indices = alive_mask.nonzero().squeeze(-1);

        if (alive_indices.numel() == 0)
            return 0;

        // Sample from alive Gaussians based on opacity
        Tensor probs = opacities.index_select(0, alive_indices);
        Tensor sampled_idxs_local = multinomial_sample(probs, n_dead, true);
        Tensor sampled_idxs = alive_indices.index_select(0, sampled_idxs_local);

        // Get parameters for sampled Gaussians
        Tensor sampled_opacities = opacities.index_select(0, sampled_idxs);
        Tensor sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

        // Count occurrences of each sampled index (how many times each was sampled)
        Tensor ratios = Tensor::ones_like(opacities, DataType::Int32);
        ratios = ratios.index_add_(0, sampled_idxs, Tensor::ones({sampled_idxs.numel()}, Device::CUDA, DataType::Int32));
        ratios = ratios.index_select(0, sampled_idxs).contiguous();

        // Clamp ratios to [1, n_max]
        const int n_max = static_cast<int>(_binoms.shape()[0]);
        ratios = ratios.clamp(1, n_max);

        // Allocate output tensors
        Tensor new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
        Tensor new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);

        // Call CUDA relocation kernel
        mcmc::launch_relocation_kernel(
            sampled_opacities.ptr<float>(),
            sampled_scales.ptr<float>(),
            ratios.ptr<int32_t>(),
            _binoms.ptr<float>(),
            n_max,
            new_opacities.ptr<float>(),
            new_scales.ptr<float>(),
            sampled_opacities.numel()
        );

        // Clamp new opacities
        new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);

        // Update parameters for sampled indices (inverse sigmoid for opacity)
        // logit(x) = log(x / (1-x))
        Tensor new_opacity_raw = (new_opacities / (Tensor::ones_like(new_opacities) - new_opacities)).log();

        // Handle opacity shape
        if (_splat_data.opacity_raw().ndim() == 2) {
            new_opacity_raw = new_opacity_raw.unsqueeze(-1);
        }

        _splat_data.opacity_raw().index_put_(sampled_idxs, new_opacity_raw);
        _splat_data.scaling_raw().index_put_(sampled_idxs, new_scales.log());

        // Copy from sampled to dead indices
        _splat_data.means().index_put_(dead_indices, _splat_data.means().index_select(0, sampled_idxs));
        _splat_data.sh0().index_put_(dead_indices, _splat_data.sh0().index_select(0, sampled_idxs));
        _splat_data.shN().index_put_(dead_indices, _splat_data.shN().index_select(0, sampled_idxs));
        _splat_data.scaling_raw().index_put_(dead_indices, _splat_data.scaling_raw().index_select(0, sampled_idxs));
        _splat_data.rotation_raw().index_put_(dead_indices, _splat_data.rotation_raw().index_select(0, sampled_idxs));
        _splat_data.opacity_raw().index_put_(dead_indices, _splat_data.opacity_raw().index_select(0, sampled_idxs));

        // Update optimizer states for all parameters
        update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Means);
        update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Sh0);
        update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::ShN);
        update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Scaling);
        update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Rotation);
        update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Opacity);

        return n_dead;
    }

    int MCMC::add_new_gs() {
        using namespace lfs::core;

        if (!_optimizer) {
            LOG_ERROR("MCMC::add_new_gs: optimizer not initialized");
            return 0;
        }

        const int current_n = _splat_data.size();
        const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
        const int n_new = std::max(0, n_target - current_n);

        if (n_new == 0)
            return 0;

        // Get opacities (handle both [N] and [N, 1] shapes)
        Tensor opacities = _splat_data.get_opacity();
        if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
            opacities = opacities.squeeze(-1);
        }

        auto probs = opacities.flatten();
        auto sampled_idxs = multinomial_sample(probs, n_new, true);

        // Get parameters for sampled Gaussians
        auto sampled_opacities = opacities.index_select(0, sampled_idxs);
        auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

        // Count occurrences (ratio starts at 0, add 1 for each occurrence, then add 1 more)
        Tensor ratios = Tensor::zeros({opacities.numel()}, Device::CUDA, DataType::Float32);
        ratios = ratios.index_add_(0, sampled_idxs, Tensor::ones({sampled_idxs.numel()}, Device::CUDA));
        ratios = ratios.index_select(0, sampled_idxs) + Tensor::ones_like(ratios.index_select(0, sampled_idxs));

        // Clamp and convert to int32
        const int n_max = static_cast<int>(_binoms.shape()[0]);
        ratios = ratios.clamp(1.0f, static_cast<float>(n_max));
        ratios = ratios.to(DataType::Int32).contiguous();

        // Allocate output tensors
        Tensor new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
        Tensor new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);

        // Call CUDA relocation kernel
        mcmc::launch_relocation_kernel(
            sampled_opacities.ptr<float>(),
            sampled_scales.ptr<float>(),
            ratios.ptr<int32_t>(),
            _binoms.ptr<float>(),
            n_max,
            new_opacities.ptr<float>(),
            new_scales.ptr<float>(),
            sampled_opacities.numel()
        );

        // Clamp new opacities
        new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);

        // Update existing Gaussians FIRST (before concatenation)
        Tensor new_opacity_raw = (new_opacities / (Tensor::ones_like(new_opacities) - new_opacities)).log();

        if (_splat_data.opacity_raw().ndim() == 2) {
            new_opacity_raw = new_opacity_raw.unsqueeze(-1);
        }

        _splat_data.opacity_raw().index_put_(sampled_idxs, new_opacity_raw);
        _splat_data.scaling_raw().index_put_(sampled_idxs, new_scales.log());

        // Prepare new Gaussians to concatenate
        auto new_means = _splat_data.means().index_select(0, sampled_idxs);
        auto new_sh0 = _splat_data.sh0().index_select(0, sampled_idxs);
        auto new_shN = _splat_data.shN().index_select(0, sampled_idxs);
        auto new_scaling = _splat_data.scaling_raw().index_select(0, sampled_idxs);
        auto new_rotation = _splat_data.rotation_raw().index_select(0, sampled_idxs);
        auto new_opacity = _splat_data.opacity_raw().index_select(0, sampled_idxs);

        // Concatenate all parameters using optimizer's add_new_params
        // Note: add_new_params automatically updates both parameters and optimizer state
        _optimizer->add_new_params(ParamType::Means, new_means);
        _optimizer->add_new_params(ParamType::Sh0, new_sh0);
        _optimizer->add_new_params(ParamType::ShN, new_shN);
        _optimizer->add_new_params(ParamType::Scaling, new_scaling);
        _optimizer->add_new_params(ParamType::Rotation, new_rotation);
        _optimizer->add_new_params(ParamType::Opacity, new_opacity);

        return n_new;
    }

    void MCMC::inject_noise() {
        using namespace lfs::core;

        // Get current learning rate from optimizer (after scheduler has updated it)
        const float current_lr = _optimizer->get_lr() * _noise_lr;

        // Generate noise
        Tensor noise = Tensor::randn_like(_splat_data.means());

        // Call CUDA add_noise kernel
        mcmc::launch_add_noise_kernel(
            _splat_data.opacity_raw().ptr<float>(),
            _splat_data.scaling_raw().ptr<float>(),
            _splat_data.rotation_raw().ptr<float>(),
            noise.ptr<float>(),
            _splat_data.means().ptr<float>(),
            current_lr,
            _splat_data.size()
        );
    }

    void MCMC::post_backward(int iter, RenderOutput& render_output) {
        // Increment SH degree every sh_degree_interval iterations
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data.increment_sh_degree();
        }

        // Refine Gaussians
        if (is_refining(iter)) {
            // Relocate dead Gaussians
            int n_relocated = relocate_gs();
            if (n_relocated > 0) {
                LOG_DEBUG("MCMC: Relocated {} dead Gaussians at iteration {}", n_relocated, iter);
            }

            // Add new Gaussians
            int n_added = add_new_gs();
            if (n_added > 0) {
                LOG_DEBUG("MCMC: Added {} new Gaussians at iteration {} (total: {})",
                         n_added, iter, _splat_data.size());
            }
        }

        // Inject noise to positions every iteration
        inject_noise();
    }

    void MCMC::step(int iter) {
        if (iter < _params->iterations) {
            _optimizer->step(iter);
            _optimizer->zero_grad(iter);
            _scheduler->step();
        }
    }

    void MCMC::remove_gaussians(const lfs::core::Tensor& mask) {
        using namespace lfs::core;

        // Convert bool to int32 for sum
        Tensor mask_int = mask.to(DataType::Int32);
        int n_remove = mask_int.sum().item<int>();

        LOG_INFO("MCMC::remove_gaussians called: mask size={}, n_remove={}, current size={}",
                 mask.numel(), n_remove, _splat_data.size());

        if (n_remove == 0) {
            LOG_DEBUG("MCMC: No Gaussians to remove");
            return;
        }

        LOG_DEBUG("MCMC: Removing {} Gaussians", n_remove);

        // Get indices to keep
        Tensor keep_mask = mask.logical_not();
        Tensor keep_indices = keep_mask.nonzero().squeeze(-1);

        // Select only the Gaussians we want to keep
        _splat_data.means() = _splat_data.means().index_select(0, keep_indices).contiguous();
        _splat_data.sh0() = _splat_data.sh0().index_select(0, keep_indices).contiguous();
        _splat_data.shN() = _splat_data.shN().index_select(0, keep_indices).contiguous();
        _splat_data.scaling_raw() = _splat_data.scaling_raw().index_select(0, keep_indices).contiguous();
        _splat_data.rotation_raw() = _splat_data.rotation_raw().index_select(0, keep_indices).contiguous();
        _splat_data.opacity_raw() = _splat_data.opacity_raw().index_select(0, keep_indices).contiguous();

        // Recreate optimizer with reduced parameters
        // This is simpler than trying to manually update optimizer state
        _optimizer = create_optimizer(_splat_data, *_params);

        // Recreate scheduler
        const double gamma = std::pow(0.01, 1.0 / _params->iterations);
        _scheduler = create_scheduler(*_params, *_optimizer);
    }

    void MCMC::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        using namespace lfs::core;

        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        // Initialize binomial coefficients (same as original)
        const int n_max = 51;
        std::vector<float> binoms_data(n_max * n_max, 0.0f);
        for (int n = 0; n < n_max; ++n) {
            for (int k = 0; k <= n; ++k) {
                float binom = 1.0f;
                for (int i = 0; i < k; ++i) {
                    binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
                }
                binoms_data[n * n_max + k] = binom;
            }
        }
        _binoms = Tensor::from_vector(binoms_data, TensorShape({static_cast<size_t>(n_max), static_cast<size_t>(n_max)}), Device::CUDA);

        // Initialize optimizer using strategy_utils helper
        _optimizer = create_optimizer(_splat_data, *_params);

        // Initialize scheduler
        _scheduler = create_scheduler(*_params, *_optimizer);

        LOG_INFO("MCMC strategy initialized with {} Gaussians", _splat_data.size());
    }

    bool MCMC::is_refining(int iter) const {
        return (iter < _params->stop_refine &&
                iter > _params->start_refine &&
                iter % _params->refine_every == 0);
    }

} // namespace lfs::training
