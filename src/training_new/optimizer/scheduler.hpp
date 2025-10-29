/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <vector>

namespace lfs::training {

    class AdamOptimizer; // Forward declaration

    /**
     * Simple Exponential Learning Rate Scheduler
     *
     * Multiplies the learning rate by gamma at each step:
     *   lr_new = lr_current * gamma
     *
     * Example:
     *   AdamOptimizer optimizer(...);
     *   ExponentialLR scheduler(optimizer, 0.99);  // Decay by 1% each step
     *
     *   for (int iter = 0; iter < 1000; iter++) {
     *       optimizer.step(iter);
     *       scheduler.step();  // Update learning rate
     *   }
     */
    class ExponentialLR {
    public:
        ExponentialLR(AdamOptimizer& optimizer, double gamma)
            : optimizer_(optimizer), gamma_(gamma) {
        }

        void step();

    private:
        AdamOptimizer& optimizer_;
        double gamma_;
    };

    /**
     * Exponential Learning Rate Scheduler with Linear Warmup
     *
     * Phase 1 (Warmup): Linearly increase LR from (initial_lr * warmup_start_factor) to initial_lr
     *   lr = initial_lr * (warmup_start_factor + (1 - warmup_start_factor) * progress)
     *   where progress = current_step / warmup_steps
     *
     * Phase 2 (Decay): Exponentially decay LR
     *   lr = initial_lr * gamma^(current_step - warmup_steps)
     *
     * Example:
     *   AdamOptimizer optimizer(...);
     *   WarmupExponentialLR scheduler(optimizer,
     *                                  gamma=0.995,           // Exponential decay rate
     *                                  warmup_steps=100,      // 100 steps warmup
     *                                  warmup_start_factor=0.1);  // Start at 10% of initial LR
     *
     *   for (int iter = 0; iter < 1000; iter++) {
     *       optimizer.step(iter);
     *       scheduler.step();  // Update learning rate
     *   }
     */
    class WarmupExponentialLR {
    public:
        WarmupExponentialLR(
            AdamOptimizer& optimizer,
            double gamma,
            int warmup_steps = 0,
            double warmup_start_factor = 1.0);

        void step();

        // Get current step count
        int get_step() const { return current_step_; }

    private:
        AdamOptimizer& optimizer_;
        double gamma_;
        int warmup_steps_;
        double warmup_start_factor_;
        int current_step_;
        double initial_lr_;
    };

} // namespace lfs::training
