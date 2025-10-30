/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "default_strategy.hpp"
#include "Ops.h"
#include "core_new/logger.hpp"
#include "core_new/parameters.hpp"
#include "kernels/densification_kernels.hpp"
#include "optimizer/render_output.hpp"
#include "strategy_utils.hpp"

namespace lfs::training {
    DefaultStrategy::DefaultStrategy(lfs::core::SplatData&& splat_data)
        : _splat_data(std::move(splat_data)) {
    }

    void DefaultStrategy::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        initialize_gaussians(_splat_data);

        // Initialize optimizer
        _optimizer = create_optimizer(_splat_data, *_params);

        // Initialize exponential scheduler
        _scheduler = create_scheduler(*_params, *_optimizer);

        // Initialize densification info: [2, N] tensor for tracking gradients
        _splat_data._densification_info = lfs::core::Tensor::zeros(
            {2, static_cast<size_t>(_splat_data.size())},
            _splat_data.means().device());
    }

    bool DefaultStrategy::is_refining(int iter) const {
        return (iter > _params->start_refine &&
                iter % _params->refine_every == 0 &&
                iter % _params->reset_every >= _params->pause_refine_after_reset);
    }

    void DefaultStrategy::remove_gaussians(const lfs::core::Tensor& mask) {
        int mask_sum = mask.to(lfs::core::DataType::Int32).sum().template item<int>();

        if (mask_sum == 0) {
            LOG_DEBUG("No Gaussians to remove");
            return;
        }

        LOG_DEBUG("Removing {} Gaussians", mask_sum);
        remove(mask);
    }

    void DefaultStrategy::duplicate(const lfs::core::Tensor& is_duplicated) {
        const lfs::core::Tensor sampled_idxs = is_duplicated.nonzero().squeeze(-1);
        const int N = _splat_data.size();
        const int num_selected = sampled_idxs.shape()[0];

        if (num_selected == 0) {
            return; // Nothing to duplicate
        }

        // Use optimized index_select + cat (same as OLD LibTorch approach, proven faster than custom kernel)
        auto pos_selected = _splat_data.means().index_select(0, sampled_idxs);
        _splat_data.means() = _splat_data.means().cat(pos_selected, 0);

        auto rot_selected = _splat_data.rotation_raw().index_select(0, sampled_idxs);
        _splat_data.rotation_raw() = _splat_data.rotation_raw().cat(rot_selected, 0);

        auto scale_selected = _splat_data.scaling_raw().index_select(0, sampled_idxs);
        _splat_data.scaling_raw() = _splat_data.scaling_raw().cat(scale_selected, 0);

        auto sh0_selected = _splat_data.sh0().index_select(0, sampled_idxs);
        _splat_data.sh0() = _splat_data.sh0().cat(sh0_selected, 0);

        auto shN_selected = _splat_data.shN().index_select(0, sampled_idxs);
        _splat_data.shN() = _splat_data.shN().cat(shN_selected, 0);

        auto op_selected = _splat_data.opacity_raw().index_select(0, sampled_idxs);
        _splat_data.opacity_raw() = _splat_data.opacity_raw().cat(op_selected, 0);

        // Update gradients to match new size
        if (_splat_data.has_gradients()) {
            _splat_data.means_grad() = lfs::core::Tensor::zeros(_splat_data.means().shape(), _splat_data.means().device());
            _splat_data.rotation_grad() = lfs::core::Tensor::zeros(_splat_data.rotation_raw().shape(), _splat_data.rotation_raw().device());
            _splat_data.scaling_grad() = lfs::core::Tensor::zeros(_splat_data.scaling_raw().shape(), _splat_data.scaling_raw().device());
            _splat_data.sh0_grad() = lfs::core::Tensor::zeros(_splat_data.sh0().shape(), _splat_data.sh0().device());
            _splat_data.shN_grad() = lfs::core::Tensor::zeros(_splat_data.shN().shape(), _splat_data.shN().device());
            _splat_data.opacity_grad() = lfs::core::Tensor::zeros(_splat_data.opacity_raw().shape(), _splat_data.opacity_raw().device());
        }

        // Update optimizer states: add zeros for new Gaussians
        auto update_state = [&](ParamType param_type, const lfs::core::Tensor& zeros) {
            if (const auto* state = _optimizer->get_state(param_type)) {
                AdamParamState new_state = *state;
                new_state.exp_avg = state->exp_avg.cat(zeros, 0);
                new_state.exp_avg_sq = state->exp_avg_sq.cat(zeros, 0);
                _optimizer->set_state(param_type, new_state);
            }
        };

        auto zeros_3 = lfs::core::Tensor::zeros({static_cast<size_t>(num_selected), 3}, _splat_data.sh0().device());
        auto zeros_4 = lfs::core::Tensor::zeros({static_cast<size_t>(num_selected), 4}, _splat_data.rotation_raw().device());
        const int shN_dim = _splat_data.shN().shape()[1];
        auto zeros_shN = lfs::core::Tensor::zeros({static_cast<size_t>(num_selected), static_cast<size_t>(shN_dim)}, _splat_data.shN().device());
        auto zeros_opacity = lfs::core::Tensor::zeros({static_cast<size_t>(num_selected), 1}, _splat_data.opacity_raw().device());

        update_state(ParamType::Means, zeros_3);
        update_state(ParamType::Rotation, zeros_4);
        update_state(ParamType::Scaling, zeros_3);
        update_state(ParamType::Sh0, zeros_3);
        update_state(ParamType::ShN, zeros_shN);
        update_state(ParamType::Opacity, zeros_opacity);
    }

    void DefaultStrategy::split(const lfs::core::Tensor& is_split) {
        const lfs::core::Tensor split_idxs = is_split.nonzero().squeeze(-1);
        const lfs::core::Tensor keep_idxs = is_split.logical_not().nonzero().squeeze(-1);

        const int N = _splat_data.size();
        const int num_split = split_idxs.shape()[0];
        const int num_keep = keep_idxs.shape()[0];

        if (num_split == 0) {
            return; // Nothing to split
        }

        constexpr int split_size = 2;

        // Get SH dimensions
        const int shN_dim = _splat_data.shN().shape()[1];

        // Generate random noise [2, num_split, 3]
        const lfs::core::Tensor random_noise = lfs::core::Tensor::randn(
            {split_size, static_cast<size_t>(num_split), 3},
            _splat_data.sh0().device());

        // Allocate output tensors [num_keep + num_split*2, ...]
        const int out_size = num_keep + num_split * split_size;
        auto positions_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 3}, _splat_data.means().device());
        auto rotations_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 4}, _splat_data.rotation_raw().device());
        auto scales_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 3}, _splat_data.scaling_raw().device());
        auto sh0_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 3}, _splat_data.sh0().device());
        auto shN_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), static_cast<size_t>(shN_dim)}, _splat_data.shN().device());
        auto opacities_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 1}, _splat_data.opacity_raw().device());

        // Call custom CUDA kernel (outputs sh0 and shN separately - NO slice/contiguous overhead!)
        kernels::launch_split_gaussians(
            _splat_data.means().ptr<float>(),
            _splat_data.rotation_raw().ptr<float>(),
            _splat_data.scaling_raw().ptr<float>(),
            _splat_data.sh0().ptr<float>(),
            _splat_data.shN().ptr<float>(),
            _splat_data.opacity_raw().ptr<float>(),
            positions_out.ptr<float>(),
            rotations_out.ptr<float>(),
            scales_out.ptr<float>(),
            sh0_out.ptr<float>(),
            shN_out.ptr<float>(),
            opacities_out.ptr<float>(),
            split_idxs.ptr<int64_t>(),
            keep_idxs.ptr<int64_t>(),
            random_noise.ptr<float>(),
            N,
            num_split,
            num_keep,
            shN_dim,
            _params->revised_opacity,
            nullptr // default stream
        );

        // Update SplatData with new tensors (already contiguous from kernel!)
        _splat_data.means() = positions_out;
        _splat_data.rotation_raw() = rotations_out;
        _splat_data.scaling_raw() = scales_out;
        _splat_data.sh0() = sh0_out;
        _splat_data.shN() = shN_out;
        _splat_data.opacity_raw() = opacities_out.squeeze(-1);

        // Update gradients to match new size
        if (_splat_data.has_gradients()) {
            _splat_data.means_grad() = lfs::core::Tensor::zeros(positions_out.shape(), positions_out.device());
            _splat_data.rotation_grad() = lfs::core::Tensor::zeros(rotations_out.shape(), rotations_out.device());
            _splat_data.scaling_grad() = lfs::core::Tensor::zeros(scales_out.shape(), scales_out.device());
            _splat_data.sh0_grad() = lfs::core::Tensor::zeros(sh0_out.shape(), sh0_out.device());
            _splat_data.shN_grad() = lfs::core::Tensor::zeros(shN_out.shape(), shN_out.device());
            _splat_data.opacity_grad() = lfs::core::Tensor::zeros(opacities_out.squeeze(-1).shape(), opacities_out.device());
        }

        // Update optimizer states: keep old states for kept Gaussians, add zeros for split Gaussians
        auto update_optimizer_state = [&](ParamType param_type, size_t param_dim) {
            if (const auto* state = _optimizer->get_state(param_type)) {
                // Keep states for kept Gaussians
                lfs::core::Tensor keep_exp_avg = state->exp_avg.index_select(0, keep_idxs);
                lfs::core::Tensor keep_exp_avg_sq = state->exp_avg_sq.index_select(0, keep_idxs);

                // Add zeros for split Gaussians
                auto zeros = lfs::core::Tensor::zeros(
                    {static_cast<size_t>(num_split * split_size), param_dim},
                    state->exp_avg.device(),
                    state->exp_avg.dtype());

                // Create new state and set it
                AdamParamState new_state = *state;
                new_state.exp_avg = keep_exp_avg.cat(zeros, 0);
                new_state.exp_avg_sq = keep_exp_avg_sq.cat(zeros, 0);
                _optimizer->set_state(param_type, new_state);
            }
        };

        update_optimizer_state(ParamType::Means, 3);
        update_optimizer_state(ParamType::Rotation, 4);
        update_optimizer_state(ParamType::Scaling, 3);
        update_optimizer_state(ParamType::Sh0, 3);
        update_optimizer_state(ParamType::ShN, shN_dim);
        update_optimizer_state(ParamType::Opacity, 1);
    }

    void DefaultStrategy::grow_gs(int iter) {
        lfs::core::Tensor numer = _splat_data._densification_info[1];
        lfs::core::Tensor denom = _splat_data._densification_info[0];
        const lfs::core::Tensor grads = numer / denom.clamp_min(1.0f);

        const lfs::core::Tensor is_grad_high = grads > _params->grad_threshold;

        // Get max along last dimension
        const lfs::core::Tensor max_values = _splat_data.get_scaling().max(-1, false);
        const lfs::core::Tensor is_small = max_values <= _params->grow_scale3d * _splat_data.get_scene_scale();
        const lfs::core::Tensor is_duplicated = is_grad_high.logical_and(is_small);

        const auto num_duplicates = static_cast<int64_t>(is_duplicated.sum_scalar());

        const lfs::core::Tensor is_large = is_small.logical_not();
        lfs::core::Tensor is_split = is_grad_high.logical_and(is_large);
        const auto num_split = static_cast<int64_t>(is_split.sum_scalar());

        // First duplicate
        if (num_duplicates > 0) {
            duplicate(is_duplicated);
        }

        // New Gaussians added by duplication will not be split
        auto zeros_to_concat = lfs::core::Tensor::zeros_bool({static_cast<size_t>(num_duplicates)}, is_split.device());
        is_split = is_split.cat(zeros_to_concat, 0);

        if (num_split > 0) {
            split(is_split);
        }
    }

    void DefaultStrategy::remove(const lfs::core::Tensor& is_prune) {
        const lfs::core::Tensor sampled_idxs = is_prune.logical_not().nonzero().squeeze(-1);

        const auto param_fn = [&sampled_idxs](const int i, const lfs::core::Tensor& param) {
            return param.index_select(0, sampled_idxs);
        };

        const auto optimizer_fn = [&sampled_idxs](
                                      AdamParamState& state,
                                      const lfs::core::Tensor& new_param) {
            // For remove, we select only the surviving Gaussians' optimizer state
            state.exp_avg = state.exp_avg.index_select(0, sampled_idxs);
            state.exp_avg_sq = state.exp_avg_sq.index_select(0, sampled_idxs);
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);
    }

    void DefaultStrategy::prune_gs(int iter) {
        // Check for low opacity
        lfs::core::Tensor is_prune = _splat_data.get_opacity() < _params->prune_opacity;

        auto rotation_raw = _splat_data.rotation_raw();
        is_prune = is_prune.logical_or((rotation_raw * rotation_raw).sum(-1, false) < 1e-8f);

        // Check for too large Gaussians
        if (iter > _params->reset_every) {
            const lfs::core::Tensor max_values = _splat_data.get_scaling().max(-1, false);
            lfs::core::Tensor is_too_big = max_values > _params->prune_scale3d * _splat_data.get_scene_scale();
            is_prune = is_prune.logical_or(is_too_big);
        }

        const auto num_prunes = static_cast<int64_t>(is_prune.sum_scalar());
        if (num_prunes > 0) {
            remove(is_prune);
        }
    }

    void DefaultStrategy::reset_opacity() {
        const auto threshold = 2.0f * _params->prune_opacity;

        const auto param_fn = [&threshold](const int i, const lfs::core::Tensor& param) {
            if (i == 5) {
                // For opacity parameter, clamp to logit(threshold)
                const float logit_threshold = std::log(threshold / (1.0f - threshold));
                return param.clamp_max(logit_threshold);
            }
            LOG_ERROR("Invalid parameter index for reset_opacity: {}", i);
            return param;
        };

        const auto optimizer_fn = [](AdamParamState& state, const lfs::core::Tensor& new_param) {
            // Reset optimizer state for opacity to zeros
            state.exp_avg = lfs::core::Tensor::zeros_like(state.exp_avg);
            state.exp_avg_sq = lfs::core::Tensor::zeros_like(state.exp_avg_sq);
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data, {5});
    }

    void DefaultStrategy::post_backward(int iter, RenderOutput& render_output) {
        // Increment SH degree every 1000 iterations
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data.increment_sh_degree();
        }

        if (iter == _params->stop_refine) {
            // Reset densification info at the end of refinement. Saves memory and processing time.
            _splat_data._densification_info = lfs::core::Tensor::empty({0});
        }

        if (iter >= _params->stop_refine) {
            return;
        }

        if (is_refining(iter)) {
            grow_gs(iter);
            prune_gs(iter);

            _splat_data._densification_info = lfs::core::Tensor::zeros(
                {2, static_cast<size_t>(_splat_data.size())},
                _splat_data.means().device());
        }

        if (iter % _params->reset_every == 0 && iter > 0) {
            reset_opacity();
        }
    }

    void DefaultStrategy::step(int iter) {
        if (iter < _params->iterations) {
            _optimizer->step(iter);
            _optimizer->zero_grad(iter);
            _scheduler->step();
        }
    }
} // namespace lfs::training
