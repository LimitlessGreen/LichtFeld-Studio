/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace gs::training {
    /**
     * @brief FusedAdam optimizer using raw CUDA memory instead of torch tensors
     *
     * This optimizer manages its own parameter groups and state using raw CUDA memory
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

            Options& set_lr(double val) {
                lr = val;
                return *this;
            }
            Options& set_betas(std::tuple<double, double> val) {
                betas = val;
                return *this;
            }
            Options& set_eps(double val) {
                eps = val;
                return *this;
            }
            Options& set_weight_decay(double val) {
                weight_decay = val;
                return *this;
            }
        };

        // Raw parameter info - no torch dependency
        struct RawParam {
            float* data_ptr;     // Parameter data pointer
            float* grad_ptr;     // Gradient data pointer
            size_t num_elements; // Number of elements
            int param_id;        // Unique ID for this parameter
        };

        struct ParamGroup {
            std::vector<RawParam> params;
            Options options;

            ParamGroup(std::vector<RawParam> p, Options opt)
                : params(std::move(p)),
                  options(opt) {}

            ParamGroup(std::vector<RawParam> p)
                : params(std::move(p)),
                  options() {}
        };

        // Raw CUDA memory for Adam state
        struct AdamState {
            float* exp_avg = nullptr;        // Raw CUDA memory
            float* exp_avg_sq = nullptr;     // Raw CUDA memory
            float* max_exp_avg_sq = nullptr; // For amsgrad variant (unused)
            size_t num_elements = 0;
            int64_t step_count = 0;

            AdamState() = default;

            AdamState(size_t n_elements);

            ~AdamState();

            // Delete copy, allow move
            AdamState(const AdamState&) = delete;
            AdamState& operator=(const AdamState&) = delete;

            AdamState(AdamState&& other) noexcept;
            AdamState& operator=(AdamState&& other) noexcept;

            void allocate(size_t n_elements);
            void free();
        };

    private:
        std::vector<ParamGroup> param_groups_;
        std::unordered_map<int, std::unique_ptr<AdamState>> state_; // Use param_id as key
        Options global_options_;
        int next_param_id_ = 0;

    public:
        // Constructors
        explicit FusedAdam(std::vector<ParamGroup> param_groups, Options global_options)
            : param_groups_(std::move(param_groups)),
              global_options_(global_options) {}

        explicit FusedAdam(std::vector<ParamGroup> param_groups)
            : param_groups_(std::move(param_groups)),
              global_options_() {}

        // Main interface
        void step(int iteration = 0);
        void zero_grad(bool set_to_none = true, int iteration = 0);

        // Access to internals
        std::vector<ParamGroup>& param_groups() { return param_groups_; }
        const std::vector<ParamGroup>& param_groups() const { return param_groups_; }

        std::unordered_map<int, std::unique_ptr<AdamState>>& state() { return state_; }
        const std::unordered_map<int, std::unique_ptr<AdamState>>& state() const { return state_; }

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

        // Helper to get next param ID
        int get_next_param_id() { return next_param_id_++; }
    };
} // namespace gs::training