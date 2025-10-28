/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "default_strategy.hpp"
#include "Ops.h"
#include "core_new/logger.hpp"
#include "core_new/parameters.hpp"
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

        const auto param_fn = [&sampled_idxs](const int i, const lfs::core::Tensor& param) {
            const lfs::core::Tensor new_param = param.index_select(0, sampled_idxs);
            return param.cat(new_param, 0);
        };

        const auto optimizer_fn = [&sampled_idxs](
            AdamParamState& state,
            const lfs::core::Tensor& full_param) {
            // For duplicate, we add zeros to the optimizer state for new Gaussians
            auto new_shape = full_param.shape();
            auto zeros_to_add = lfs::core::Tensor::zeros(
                {sampled_idxs.shape()[0], new_shape[1]},
                state.exp_avg.device(),
                state.exp_avg.dtype());

            state.exp_avg = state.exp_avg.cat(zeros_to_add, 0);
            state.exp_avg_sq = state.exp_avg_sq.cat(zeros_to_add, 0);
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);
    }

    void DefaultStrategy::split(const lfs::core::Tensor& is_split) {
        const lfs::core::Tensor sampled_idxs = is_split.nonzero().squeeze(-1);
        const lfs::core::Tensor rest_idxs = is_split.logical_not().nonzero().squeeze(-1);

        const lfs::core::Tensor sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);
        const lfs::core::Tensor sampled_quats = _splat_data.get_rotation().index_select(0, sampled_idxs);

        // Convert quaternions to rotation matrices manually
        // sampled_quats: [N, 4] with [w, x, y, z]
        const auto w = sampled_quats.slice(1, 0, 1).squeeze(-1); // [N]
        const auto x = sampled_quats.slice(1, 1, 2).squeeze(-1); // [N]
        const auto y = sampled_quats.slice(1, 2, 3).squeeze(-1); // [N]
        const auto z = sampled_quats.slice(1, 3, 4).squeeze(-1); // [N]

        // Compute rotation matrix elements
        // R = [[1-2(y²+z²), 2(xy-wz), 2(xz+wy)],
        //      [2(xy+wz), 1-2(x²+z²), 2(yz-wx)],
        //      [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]]
        const auto two = lfs::core::Tensor::full_like(w, 2.0f);
        const auto one = lfs::core::Tensor::ones_like(w);

        const auto r00 = one - two * (y * y + z * z);
        const auto r01 = two * (x * y - w * z);
        const auto r02 = two * (x * z + w * y);
        const auto r10 = two * (x * y + w * z);
        const auto r11 = one - two * (x * x + z * z);
        const auto r12 = two * (y * z - w * x);
        const auto r20 = two * (x * z - w * y);
        const auto r21 = two * (y * z + w * x);
        const auto r22 = one - two * (x * x + y * y);

        // Stack to form rotation matrices [N, 3, 3]
        // Stack rows first, then stack the 3 rows
        auto row0 = r00.unsqueeze(-1).cat(r01.unsqueeze(-1), -1).cat(r02.unsqueeze(-1), -1); // [N, 3]
        auto row1 = r10.unsqueeze(-1).cat(r11.unsqueeze(-1), -1).cat(r12.unsqueeze(-1), -1); // [N, 3]
        auto row2 = r20.unsqueeze(-1).cat(r21.unsqueeze(-1), -1).cat(r22.unsqueeze(-1), -1); // [N, 3]

        // Stack rows to get [N, 3, 3]
        const auto rotmats = row0.unsqueeze(1).cat(row1.unsqueeze(1), 1).cat(row2.unsqueeze(1), 1);

        const auto num_split_gaussians = sampled_idxs.shape()[0];
        constexpr auto split_size = 2;

        // Generate random samples [split_size, N, 3]
        const lfs::core::Tensor randn = lfs::core::Tensor::randn(
            {split_size, num_split_gaussians, 3},
            sampled_quats.device());

        // Compute einsum manually: samples[b,n,i] = rotmats[n,i,j] * scales[n,j] * randn[b,n,j]
        // This is: samples = rotmats @ diag(scales) @ randn[b]
        // For each split b, compute: rotmats[n] @ (scales[n] * randn[b,n])
        lfs::core::Tensor samples_list[split_size];
        for (int b = 0; b < split_size; ++b) {
            // randn[b] has shape [N, 3]
            auto randn_b = randn[b]; // [N, 3]
            // Element-wise multiply with scales: [N, 3]
            auto scaled_randn = sampled_scales * randn_b; // [N, 3]
            // Batch matrix-vector multiply: rotmats @ scaled_randn
            // rotmats: [N, 3, 3], scaled_randn: [N, 3] -> need [N, 3, 1]
            auto scaled_randn_col = scaled_randn.unsqueeze(-1); // [N, 3, 1]
            auto rotated = rotmats.bmm(scaled_randn_col).squeeze(-1); // [N, 3]
            samples_list[b] = rotated;
        }

        // Stack samples: [split_size, N, 3]
        lfs::core::Tensor samples = samples_list[0].unsqueeze(0);
        for (int b = 1; b < split_size; ++b) {
            samples = samples.cat(samples_list[b].unsqueeze(0), 0);
        }

        const auto param_fn = [this, &sampled_idxs, &rest_idxs, &samples, &sampled_scales](
                                  const int i, const lfs::core::Tensor& param) {
            const lfs::core::Tensor sampled_param = param.index_select(0, sampled_idxs);
            lfs::core::Tensor split_param;

            if (i == 0) {
                // means: add offset to each split
                split_param = (sampled_param.unsqueeze(0) + samples).reshape({-1, 3}); // [split_size * N, 3]
            } else if (i == 3) {
                // scaling: divide by 1.6 and duplicate
                auto new_scales = (sampled_scales / 1.6f).log();
                // Duplicate split_size times
                split_param = new_scales;
                for (int s = 1; s < split_size; ++s) {
                    split_param = split_param.cat(new_scales, 0);
                }
            } else if (i == 5 && _params->revised_opacity) {
                // opacity: revised formula
                // new_opacity = 1 - sqrt(1 - sigmoid(sampled_param))
                const lfs::core::Tensor sigmoid_vals = sampled_param.sigmoid();
                const lfs::core::Tensor one_minus_sigmoid = lfs::core::Tensor::ones_like(sigmoid_vals) - sigmoid_vals;
                const lfs::core::Tensor new_opacities = lfs::core::Tensor::ones_like(sigmoid_vals) - one_minus_sigmoid.sqrt();
                auto logit_opacities = new_opacities.logit();
                // Duplicate split_size times
                split_param = logit_opacities;
                for (int s = 1; s < split_size; ++s) {
                    split_param = split_param.cat(logit_opacities, 0);
                }
            } else {
                // other parameters: just duplicate
                split_param = sampled_param;
                for (int s = 1; s < split_size; ++s) {
                    split_param = split_param.cat(sampled_param, 0);
                }
            }

            const lfs::core::Tensor rest_param = param.index_select(0, rest_idxs);
            return rest_param.cat(split_param, 0);
        };

        const auto optimizer_fn = [&sampled_idxs, &rest_idxs](
            AdamParamState& state,
            const lfs::core::Tensor& full_param) {
            // For split, we keep the non-split states and add zeros for split Gaussians
            auto rest_exp_avg = state.exp_avg.index_select(0, rest_idxs);
            auto rest_exp_avg_sq = state.exp_avg_sq.index_select(0, rest_idxs);

            // Create new shape for zeros
            std::vector<size_t> zero_shape_vec;
            for (size_t i = 0; i < full_param.ndim(); ++i) {
                if (i == 0) {
                    zero_shape_vec.push_back(sampled_idxs.shape()[0] * split_size);
                } else {
                    zero_shape_vec.push_back(full_param.shape()[i]);
                }
            }
            auto zeros_to_add = lfs::core::Tensor::zeros(
                lfs::core::TensorShape(zero_shape_vec),
                state.exp_avg.device(),
                state.exp_avg.dtype());

            state.exp_avg = rest_exp_avg.cat(zeros_to_add, 0);
            state.exp_avg_sq = rest_exp_avg_sq.cat(zeros_to_add, 0);
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);
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
