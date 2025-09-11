/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scheduler.hpp"
#include "fused_adam.hpp"
#include <cmath>

namespace gs::training {
    void ExponentialLR::step() {
        if (param_group_index_ >= 0) {
            double current_lr = optimizer_.get_lr(param_group_index_);
            optimizer_.set_lr(current_lr * gamma_, param_group_index_);
        } else {
            // Update all param groups
            for (size_t i = 0; i < optimizer_.param_groups().size(); ++i) {
                double current_lr = optimizer_.get_lr(i);
                optimizer_.set_lr(current_lr * gamma_, i);
            }
        }
    }

    WarmupExponentialLR::WarmupExponentialLR(
        FusedAdam& optimizer,
        double gamma,
        int warmup_steps,
        double warmup_start_factor,
        int param_group_index)
        : torch_optimizer_(nullptr),
          fused_adam_(&optimizer),
          gamma_(gamma),
          warmup_steps_(warmup_steps),
          warmup_start_factor_(warmup_start_factor),
          param_group_index_(param_group_index),
          current_step_(0) {
        // Store initial learning rates for all param groups
        for (size_t i = 0; i < optimizer.param_groups().size(); ++i) {
            initial_lrs_.push_back(optimizer.get_lr(i));
        }
    }

    void WarmupExponentialLR::step() {
        current_step_++;

        auto update_group = [this](int group_idx) {
            double initial_lr = initial_lrs_[group_idx];
            double new_lr;

            if (current_step_ <= warmup_steps_) {
                // Linear warmup from start_factor to 1.0
                double progress = static_cast<double>(current_step_) / warmup_steps_;
                double factor = warmup_start_factor_ + (1.0 - warmup_start_factor_) * progress;
                new_lr = initial_lr * factor;
            } else {
                // Exponential decay after warmup
                int decay_steps = current_step_ - warmup_steps_;
                new_lr = initial_lr * std::pow(gamma_, decay_steps);
            }

            // Update learning rate
            if (fused_adam_) {
                fused_adam_->set_lr(new_lr, group_idx);
            } else if (torch_optimizer_) {
                auto& group = torch_optimizer_->param_groups()[group_idx];
                if (auto* adam_options = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
                    adam_options->lr(new_lr);
                }
            }
        };

        if (param_group_index_ >= 0) {
            // Update specific param group
            update_group(param_group_index_);
        } else {
            // Update all param groups
            size_t num_groups = fused_adam_ ? fused_adam_->param_groups().size()
                                            : torch_optimizer_->param_groups().size();
            for (size_t i = 0; i < num_groups; ++i) {
                update_group(i);
            }
        }
    }
} // namespace gs::training