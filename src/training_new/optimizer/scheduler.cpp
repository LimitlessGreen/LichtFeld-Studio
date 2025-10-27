/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scheduler.hpp"
#include "adam_optimizer.hpp"
#include <cmath>

namespace lfs::training {

    WarmupExponentialLR::WarmupExponentialLR(
        AdamOptimizer& optimizer,
        double gamma,
        int warmup_steps,
        double warmup_start_factor)
        : optimizer_(optimizer),
          gamma_(gamma),
          warmup_steps_(warmup_steps),
          warmup_start_factor_(warmup_start_factor),
          current_step_(0) {
        // Store initial learning rate
        initial_lr_ = optimizer.get_lr();
    }

    void ExponentialLR::step() {
        double current_lr = optimizer_.get_lr();
        double new_lr = current_lr * gamma_;
        optimizer_.set_lr(static_cast<float>(new_lr));
    }

    void WarmupExponentialLR::step() {
        current_step_++;

        double new_lr;

        if (current_step_ <= warmup_steps_) {
            // Linear warmup from start_factor to 1.0
            double progress = static_cast<double>(current_step_) / warmup_steps_;
            double factor = warmup_start_factor_ + (1.0 - warmup_start_factor_) * progress;
            new_lr = initial_lr_ * factor;
        } else {
            // Exponential decay after warmup
            int decay_steps = current_step_ - warmup_steps_;
            new_lr = initial_lr_ * std::pow(gamma_, decay_steps);
        }

        optimizer_.set_lr(static_cast<float>(new_lr));
    }

} // namespace lfs::training
