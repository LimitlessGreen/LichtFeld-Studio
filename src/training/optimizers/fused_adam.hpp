/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <torch/torch.h>
#include <vector>
#include <unordered_map>

namespace gs::training {
    /**
     * @brief FusedAdam optimizer using composition instead of inheritance
     *
     * This optimizer manages its own parameter groups and state without
     * inheriting from torch::optim::Optimizer
     */
    class FusedAdam {
    public:
        struct Options {
            double lr = 1e-3;
            std::tuple<double, double> betas = {0.9, 0.999};
            double eps = 1e-8;
            double weight_decay = 0.0;

            Options() = default;
            Options(double learning_rate) : lr(learning_rate) {}

            Options& set_lr(double val) { lr = val; return *this; }
            Options& set_betas(std::tuple<double, double> val) { betas = val; return *this; }
            Options& set_eps(double val) { eps = val; return *this; }
            Options& set_weight_decay(double val) { weight_decay = val; return *this; }
        };

        struct ParamGroup {
            std::vector<torch::Tensor> params;
            Options options;

            ParamGroup(std::vector<torch::Tensor> p, Options opt)
                : params(std::move(p)), options(opt) {}

            ParamGroup(std::vector<torch::Tensor> p)
                : params(std::move(p)), options() {}
        };

        struct AdamState {
            torch::Tensor exp_avg;
            torch::Tensor exp_avg_sq;
            torch::Tensor max_exp_avg_sq; // For amsgrad variant
            int64_t step_count = 0;
        };

    private:
        std::vector<ParamGroup> param_groups_;
        std::unordered_map<void*, std::unique_ptr<AdamState>> state_;
        Options global_options_;

    public:
        // Constructors
        explicit FusedAdam(std::vector<ParamGroup> param_groups, Options global_options)
            : param_groups_(std::move(param_groups)), global_options_(global_options) {}

        explicit FusedAdam(std::vector<ParamGroup> param_groups)
            : param_groups_(std::move(param_groups)), global_options_() {}

        explicit FusedAdam(std::vector<torch::Tensor> params, Options options)
            : param_groups_{{std::move(params), options}}, global_options_(options) {}

        explicit FusedAdam(std::vector<torch::Tensor> params)
            : param_groups_{{std::move(params), Options()}}, global_options_() {}

        // Main interface
        void step(int iteration = 0);
        void zero_grad(bool set_to_none = true, int iteration = 0);

        // Access to internals
        std::vector<ParamGroup>& param_groups() { return param_groups_; }
        const std::vector<ParamGroup>& param_groups() const { return param_groups_; }

        std::unordered_map<void*, std::unique_ptr<AdamState>>& state() { return state_; }
        const std::unordered_map<void*, std::unique_ptr<AdamState>>& state() const { return state_; }

        // Get/set learning rate for specific group
        void set_lr(double lr, int group_idx = -1) {
            if (group_idx < 0) {
                global_options_.lr = lr;
                for (auto& group : param_groups_) {
                    group.options.lr = lr;
                }
            } else if (group_idx < static_cast<int>(param_groups_.size())) {
                param_groups_[group_idx].options.lr = lr;
            }
        }

        double get_lr(int group_idx = 0) const {
            if (group_idx < static_cast<int>(param_groups_.size())) {
                return param_groups_[group_idx].options.lr;
            }
            return global_options_.lr;
        }
    };
} // namespace gs::training