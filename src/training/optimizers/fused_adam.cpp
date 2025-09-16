/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fused_adam.hpp"
#include "adam_api.h"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

// TODO: This is just a gimmick for the bounty. I don't think it should be integrated into the main codebase.
// TODO: Removing the SH step skipping also means that the custom zero_grad() method is no longer needed.
// TODO: All skipping conditions assume that iteration count starts at 1 (which is it currently does).
// Between iteration 1000 and 25000, we can skip every second step for higher degree SH coefficients.
// This does push the bounty benchmark below 20 minutes, but I don't really like the practical implications.
// It also causes a *very* small drop in quality metrics and robustness. Thus, I disable it by default.
#define SKIP_SH_STEPS false

namespace gs::training {

    // AdamState implementation
    FusedAdam::AdamState::AdamState(size_t n_elements) {
        allocate(n_elements);
    }

    FusedAdam::AdamState::~AdamState() {
        free();
    }

    FusedAdam::AdamState::AdamState(AdamState&& other) noexcept
        : exp_avg(other.exp_avg),
          exp_avg_sq(other.exp_avg_sq),
          max_exp_avg_sq(other.max_exp_avg_sq),
          num_elements(other.num_elements),
          step_count(other.step_count) {
        other.exp_avg = nullptr;
        other.exp_avg_sq = nullptr;
        other.max_exp_avg_sq = nullptr;
        other.num_elements = 0;
    }

    FusedAdam::AdamState& FusedAdam::AdamState::operator=(AdamState&& other) noexcept {
        if (this != &other) {
            free();
            exp_avg = other.exp_avg;
            exp_avg_sq = other.exp_avg_sq;
            max_exp_avg_sq = other.max_exp_avg_sq;
            num_elements = other.num_elements;
            step_count = other.step_count;

            other.exp_avg = nullptr;
            other.exp_avg_sq = nullptr;
            other.max_exp_avg_sq = nullptr;
            other.num_elements = 0;
        }
        return *this;
    }

    void FusedAdam::AdamState::allocate(size_t n_elements) {
        if (exp_avg || exp_avg_sq) {
            free();
        }

        num_elements = n_elements;
        size_t bytes = n_elements * sizeof(float);

        cudaError_t err = cudaMalloc(&exp_avg, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate exp_avg: " + std::string(cudaGetErrorString(err)));
        }

        err = cudaMalloc(&exp_avg_sq, bytes);
        if (err != cudaSuccess) {
            cudaFree(exp_avg);
            exp_avg = nullptr;
            throw std::runtime_error("Failed to allocate exp_avg_sq: " + std::string(cudaGetErrorString(err)));
        }

        // Initialize to zero
        cudaMemset(exp_avg, 0, bytes);
        cudaMemset(exp_avg_sq, 0, bytes);
    }

    void FusedAdam::AdamState::free() {
        if (exp_avg) {
            cudaFree(exp_avg);
            exp_avg = nullptr;
        }
        if (exp_avg_sq) {
            cudaFree(exp_avg_sq);
            exp_avg_sq = nullptr;
        }
        if (max_exp_avg_sq) {
            cudaFree(max_exp_avg_sq);
            max_exp_avg_sq = nullptr;
        }
        num_elements = 0;
    }

    void FusedAdam::step(int iteration) {
        int i = 0; // HACK: counter to track what Gaussian parameter we are on
        for (auto& group : param_groups_) {
            ++i;

            // Use group's options
            double lr = group.options.lr;
            double eps = group.options.eps;
            auto [beta1, beta2] = group.options.betas;

            for (auto& param : group.params) {
                if (!param.grad_ptr) {
                    continue;
                }

                // Lazy state initialization using param_id
                auto state_ptr = state_.find(param.param_id);
                if (state_ptr == state_.end()) {
                    auto new_state = std::make_unique<AdamState>(param.num_elements);
                    new_state->step_count = 0;
                    state_[param.param_id] = std::move(new_state);
                    state_ptr = state_.find(param.param_id);
                }

                auto& state = *state_ptr->second;

                // Increment step
                state.step_count++;

                // Higher degree SH coefficients are not used in the first 1000 iterations so this is a free speed up
                if (i == 3 && iteration <= 1000)
                    continue;

                if constexpr (SKIP_SH_STEPS) {
                    // Skip every second step during training except for the last 5000 iterations
                    if (i == 3 && (iteration % 2 != 0 && iteration <= 25000))
                        continue;
                }

                auto bias_correction1_rcp = 1.0 / (1.0 - std::pow(beta1, state.step_count));
                auto bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(beta2, state.step_count));

                // Call the pure CUDA kernel from fastgs
                fast_gs::optimizer::adam_step_raw(
                    param.data_ptr,
                    state.exp_avg,
                    state.exp_avg_sq,
                    param.grad_ptr,
                    param.num_elements,
                    static_cast<float>(lr),
                    static_cast<float>(beta1),
                    static_cast<float>(beta2),
                    static_cast<float>(eps),
                    static_cast<float>(bias_correction1_rcp),
                    static_cast<float>(bias_correction2_sqrt_rcp));
            }
        }
    }

    void FusedAdam::zero_grad(bool set_to_none, int iteration) {
        // Zero gradients using raw CUDA memory
        for (auto& group : param_groups_) {
            for (auto& param : group.params) {
                if (param.grad_ptr) {
                    cudaMemset(param.grad_ptr, 0, param.num_elements * sizeof(float));
                }
            }
        }
    }
} // namespace gs::training