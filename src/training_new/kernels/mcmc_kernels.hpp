/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>

namespace lfs::training::mcmc {

    /**
     * Relocation kernel - Equation (9) from "3D Gaussian Splatting as Markov Chain Monte Carlo"
     *
     * Computes new opacities and scales for relocated Gaussians based on their sampling ratios.
     *
     * @param opacities [N] - Original opacity values
     * @param scales [N, 3] - Original scale values
     * @param ratios [N] - Number of times each Gaussian was sampled (int32)
     * @param binoms [n_max, n_max] - Precomputed binomial coefficients
     * @param n_max - Maximum ratio value (size of binomial table)
     * @param new_opacities [N] - Output: relocated opacity values
     * @param new_scales [N, 3] - Output: relocated scale values
     * @param N - Number of Gaussians
     * @param stream - CUDA stream for async execution
     */
    void launch_relocation_kernel(
        const float* opacities,
        const float* scales,
        const int32_t* ratios,
        const float* binoms,
        int n_max,
        float* new_opacities,
        float* new_scales,
        size_t N,
        void* stream = nullptr);

    /**
     * Add noise kernel - Injects position noise scaled by covariance
     *
     * Adds Gaussian noise to mean positions, scaled by the Gaussian covariance
     * and learning rate. Used for MCMC exploration.
     *
     * @param raw_opacities [N] - Raw (pre-sigmoid) opacity values
     * @param raw_scales [N, 3] - Raw (pre-exp) scale values
     * @param raw_quats [N, 4] - Raw quaternion rotation values
     * @param noise [N, 3] - Random noise from N(0,1)
     * @param means [N, 3] - Mean positions (modified in-place)
     * @param current_lr - Current learning rate for noise scaling
     * @param N - Number of Gaussians
     * @param stream - CUDA stream for async execution
     */
    void launch_add_noise_kernel(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        const float* noise,
        float* means,
        float current_lr,
        size_t N,
        void* stream = nullptr);

} // namespace lfs::training::mcmc
