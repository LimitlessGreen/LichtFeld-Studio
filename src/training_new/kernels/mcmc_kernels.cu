/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcmc_kernels.hpp"
#include <cuda_runtime.h>

namespace lfs::training::mcmc {

    // Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
    __global__ void relocation_kernel(
        const float* opacities,
        const float* scales,
        const int32_t* ratios,
        const float* binoms,
        int n_max,
        float* new_opacities,
        float* new_scales,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        int n_idx = ratios[idx];

        // Safety check: n_idx must be >= 1
        if (n_idx < 1) {
            new_opacities[idx] = opacities[idx];
            for (int i = 0; i < 3; ++i) {
                new_scales[idx * 3 + i] = scales[idx * 3 + i];
            }
            return;
        }

        float denom_sum = 0.0f;

        // Compute new opacity: 1 - (1 - old_opacity)^(1/n_idx)
        float opacity_base = fmaxf(0.0f, fminf(1.0f, opacities[idx]));  // Clamp to [0, 1]
        new_opacities[idx] = 1.0f - powf(1.0f - opacity_base, 1.0f / static_cast<float>(n_idx));

        // Compute new scale
        for (int i = 1; i <= n_idx; ++i) {
            for (int k = 0; k <= (i - 1); ++k) {
                float bin_coeff = binoms[(i - 1) * n_max + k];
                float term = (powf(-1.0f, k) / sqrtf(static_cast<float>(k + 1))) *
                             powf(new_opacities[idx], k + 1);
                denom_sum += (bin_coeff * term);
            }
        }

        // Safety check: avoid division by zero
        float coeff = (fabsf(denom_sum) > 1e-8f) ? (opacity_base / denom_sum) : 1.0f;
        for (int i = 0; i < 3; ++i) {
            new_scales[idx * 3 + i] = coeff * scales[idx * 3 + i];
        }
    }

    void launch_relocation_kernel(
        const float* opacities,
        const float* scales,
        const int32_t* ratios,
        const float* binoms,
        int n_max,
        float* new_opacities,
        float* new_scales,
        size_t N,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        relocation_kernel<<<grid, threads, 0, cuda_stream>>>(
            opacities,
            scales,
            ratios,
            binoms,
            n_max,
            new_opacities,
            new_scales,
            N);
    }

    // Helper: Convert raw quaternion to rotation matrix
    __device__ inline void raw_quat_to_rotmat(const float* raw_quat, float* R) {
        float w = raw_quat[0], x = raw_quat[1], y = raw_quat[2], z = raw_quat[3];

        // Normalize
        float inv_norm = fminf(rsqrtf(x * x + y * y + z * z + w * w), 1e+12f);
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
        w *= inv_norm;

        float x2 = x * x, y2 = y * y, z2 = z * z;
        float xy = x * y, xz = x * z, yz = y * z;
        float wx = w * x, wy = w * y, wz = w * z;

        // Column-major order (matching glm)
        R[0] = 1.f - 2.f * (y2 + z2);
        R[1] = 2.f * (xy + wz);
        R[2] = 2.f * (xz - wy);

        R[3] = 2.f * (xy - wz);
        R[4] = 1.f - 2.f * (x2 + z2);
        R[5] = 2.f * (yz + wx);

        R[6] = 2.f * (xz + wy);
        R[7] = 2.f * (yz - wx);
        R[8] = 1.f - 2.f * (x2 + y2);
    }

    // Helper: Matrix-vector multiplication (3x3 * 3x1)
    __device__ inline void matvec3(const float* M, const float* v, float* result) {
        result[0] = M[0] * v[0] + M[3] * v[1] + M[6] * v[2];
        result[1] = M[1] * v[0] + M[4] * v[1] + M[7] * v[2];
        result[2] = M[2] * v[0] + M[5] * v[1] + M[8] * v[2];
    }

    __global__ void add_noise_kernel(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        const float* noise,
        float* means,
        float current_lr,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        size_t idx_3d = 3 * idx;
        size_t idx_4d = 4 * idx;

        // Compute S^2 (diagonal matrix from exp(2 * raw_scale))
        float S2[9] = {0};
        S2[0] = __expf(2.f * raw_scales[idx_3d + 0]);
        S2[4] = __expf(2.f * raw_scales[idx_3d + 1]);
        S2[8] = __expf(2.f * raw_scales[idx_3d + 2]);

        // Get rotation matrix R from quaternion
        float R[9];
        raw_quat_to_rotmat(raw_quats + idx_4d, R);

        // Compute R * S^2 (temp storage)
        float RS2[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                RS2[i * 3 + j] = R[i * 3 + 0] * S2[0 * 3 + j] +
                                 R[i * 3 + 1] * S2[1 * 3 + j] +
                                 R[i * 3 + 2] * S2[2 * 3 + j];
            }
        }

        // Compute covariance = R * S^2 * R^T
        float covariance[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                covariance[i * 3 + j] = RS2[i * 3 + 0] * R[j * 3 + 0] +
                                        RS2[i * 3 + 1] * R[j * 3 + 1] +
                                        RS2[i * 3 + 2] * R[j * 3 + 2];
            }
        }

        // Transform noise: transformed_noise = covariance * noise
        float transformed_noise[3];
        float noise_vec[3] = {noise[idx_3d], noise[idx_3d + 1], noise[idx_3d + 2]};
        matvec3(covariance, noise_vec, transformed_noise);

        // Compute opacity-based scaling factor
        float opacity = __frcp_rn(1.f + __expf(-raw_opacities[idx]));  // sigmoid
        float op_sigmoid = __frcp_rn(1.f + __expf(100.f * opacity - 0.5f));
        float noise_factor = current_lr * op_sigmoid;

        // Add scaled noise to means
        means[idx_3d + 0] += noise_factor * transformed_noise[0];
        means[idx_3d + 1] += noise_factor * transformed_noise[1];
        means[idx_3d + 2] += noise_factor * transformed_noise[2];
    }

    void launch_add_noise_kernel(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        const float* noise,
        float* means,
        float current_lr,
        size_t N,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        add_noise_kernel<<<grid, threads, 0, cuda_stream>>>(
            raw_opacities,
            raw_scales,
            raw_quats,
            noise,
            means,
            current_lr,
            N);
    }

} // namespace lfs::training::mcmc
