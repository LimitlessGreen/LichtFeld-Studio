/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "regularization.hpp"
#include "lfs/kernels/regularization.cuh" // LibTorch-free CUDA kernels
#include <format>

namespace lfs::training::losses {

    std::expected<float, std::string> ScaleRegularization::forward(
        const lfs::core::Tensor& scaling_raw,
        lfs::core::Tensor& scaling_raw_grad,
        const Params& params) {
        try {
            if (params.weight <= 0.0f) {
                return 0.0f;
            }

            // Validate inputs
            if (scaling_raw.device() != lfs::core::Device::CUDA) {
                return std::unexpected("scaling_raw must be on CUDA device");
            }
            if (scaling_raw_grad.device() != lfs::core::Device::CUDA) {
                return std::unexpected("scaling_raw_grad must be on CUDA device");
            }
            if (scaling_raw.shape() != scaling_raw_grad.shape()) {
                return std::unexpected("scaling_raw and scaling_raw_grad must have same shape");
            }

            size_t n = scaling_raw.numel();
            if (n == 0) {
                return 0.0f;
            }

            // Allocate temporary buffers
            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::CUDA);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);

            // Launch LibTorch-free fused kernel with warp reductions
            lfs::training::kernels::launch_fused_scale_regularization(
                scaling_raw.ptr<float>(),
                scaling_raw_grad.ptr<float>(),
                loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                n,
                params.weight,
                nullptr);

            // Copy result to host
            float loss = loss_tensor.item<float>();
            return loss;

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in ScaleRegularization::forward: {}", e.what()));
        }
    }

    std::expected<float, std::string> OpacityRegularization::forward(
        const lfs::core::Tensor& opacity_raw,
        lfs::core::Tensor& opacity_raw_grad,
        const Params& params) {
        try {
            if (params.weight <= 0.0f) {
                return 0.0f;
            }

            // Validate inputs
            if (opacity_raw.device() != lfs::core::Device::CUDA) {
                return std::unexpected("opacity_raw must be on CUDA device");
            }
            if (opacity_raw_grad.device() != lfs::core::Device::CUDA) {
                return std::unexpected("opacity_raw_grad must be on CUDA device");
            }
            if (opacity_raw.shape() != opacity_raw_grad.shape()) {
                return std::unexpected("opacity_raw and opacity_raw_grad must have same shape");
            }

            size_t n = opacity_raw.numel();
            if (n == 0) {
                return 0.0f;
            }

            // Allocate temporary buffers
            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::CUDA);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);

            // Launch LibTorch-free fused kernel with warp reductions
            lfs::training::kernels::launch_fused_opacity_regularization(
                opacity_raw.ptr<float>(),
                opacity_raw_grad.ptr<float>(),
                loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                n,
                params.weight,
                nullptr);

            // Copy result to host
            float loss = loss_tensor.item<float>();
            return loss;

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in OpacityRegularization::forward: {}", e.what()));
        }
    }

} // namespace lfs::training::losses
