/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "kernels/bilateral_grid.cuh"
#include <torch/torch.h>

namespace gs::training {

    class BilateralGrid {
    public:
        BilateralGrid(int num_images, int grid_W = 16, int grid_H = 16, int grid_L = 8);

        // Apply bilateral grid to rendered image (OLD - uses autograd)
        torch::Tensor apply(const torch::Tensor& rgb, int image_idx);

        // MANUAL FORWARD: Apply bilateral grid without autograd
        // Returns: (output_image, context)
        std::pair<torch::Tensor, bilateral_grid::BilateralGridSliceContext> apply_forward(
            const torch::Tensor& rgb, int image_idx);

        // MANUAL BACKWARD: Compute gradients manually
        // Returns: gradient w.r.t. rgb (grad_rgb)
        // NOTE: grad_grid is accumulated into grids_.grad()
        torch::Tensor apply_backward(
            const bilateral_grid::BilateralGridSliceContext& ctx,
            const torch::Tensor& grad_output,
            int image_idx);

        // Compute total variation loss (OLD - uses autograd)
        torch::Tensor tv_loss() const;

        // MANUAL TV LOSS: Compute TV loss without autograd
        // Returns: (loss_value, context)
        std::pair<float, bilateral_grid::BilateralGridTVContext> tv_loss_forward() const;

        // MANUAL TV BACKWARD: Compute TV loss gradients manually
        // NOTE: gradients are accumulated into grids_.grad()
        void tv_loss_backward(
            const bilateral_grid::BilateralGridTVContext& ctx,
            float grad_loss);

        // Get parameters for optimizer
        torch::Tensor parameters() { return grids_; }
        const torch::Tensor& parameters() const { return grids_; }

        // Grid dimensions
        int grid_width() const { return grid_width_; }
        int grid_height() const { return grid_height_; }
        int grid_guidance() const { return grid_guidance_; }

    private:
        torch::Tensor grids_; // [N, 12, L, H, W]
        int num_images_;
        int grid_width_;
        int grid_height_;
        int grid_guidance_;
    };

} // namespace gs::training