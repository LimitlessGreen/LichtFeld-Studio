/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "regularization.hpp"
#include "kernels/regularization.cuh"  // CUDA kernels that use torch::Tensor
#include <format>
#include <torch/torch.h>

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

        // Create torch::Tensor views (zero-copy) that wrap our lfs::core::Tensor data
        // The CUDA kernel expects torch tensors, so we wrap the raw pointers
        std::vector<int64_t> torch_shape;
        for (size_t i = 0; i < scaling_raw.ndim(); i++) {
            torch_shape.push_back(static_cast<int64_t>(scaling_raw.shape()[i]));
        }

        // Wrap the raw data pointers as torch tensors (no copy!)
        // We need a mutable wrapper for the CUDA kernel to write gradients
        auto torch_scaling_raw = torch::from_blob(
            const_cast<float*>(scaling_raw.ptr<float>()),
            torch_shape,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Create a torch tensor that has a grad buffer pointing to our grad tensor
        torch_scaling_raw.set_requires_grad(true);

        // Manually set the grad tensor to point to our existing gradient buffer
        auto torch_grad = torch::from_blob(
            scaling_raw_grad.ptr<float>(),
            torch_shape,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        torch_scaling_raw.mutable_grad() = torch_grad;

        // Call the existing CUDA kernel (it accumulates to .grad())
        float loss = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
            torch_scaling_raw,
            params.weight);

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

        // Create torch::Tensor views (zero-copy)
        std::vector<int64_t> torch_shape;
        for (size_t i = 0; i < opacity_raw.ndim(); i++) {
            torch_shape.push_back(static_cast<int64_t>(opacity_raw.shape()[i]));
        }

        auto torch_opacity_raw = torch::from_blob(
            const_cast<float*>(opacity_raw.ptr<float>()),
            torch_shape,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        torch_opacity_raw.set_requires_grad(true);

        auto torch_grad = torch::from_blob(
            opacity_raw_grad.ptr<float>(),
            torch_shape,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        torch_opacity_raw.mutable_grad() = torch_grad;

        // Call the existing CUDA kernel
        float loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
            torch_opacity_raw,
            params.weight);

        return loss;

    } catch (const std::exception& e) {
        return std::unexpected(std::format("Error in OpacityRegularization::forward: {}", e.what()));
    }
}

} // namespace lfs::training::losses
