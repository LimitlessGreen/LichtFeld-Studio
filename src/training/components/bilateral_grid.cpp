/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "bilateral_grid.hpp"
#include "kernels/bilateral_grid.cuh"

namespace gs::training {

    // Autograd function for bilateral grid slicing
    class BilateralGridSliceFunction : public torch::autograd::Function<BilateralGridSliceFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor grid,
            torch::Tensor rgb) {
            // Input validation
            TORCH_CHECK(grid.dim() == 4 && grid.size(0) == 12,
                        "Grid must be [12, L, H, W]");
            TORCH_CHECK(rgb.dim() == 3 && rgb.size(2) == 3,
                        "RGB must be [H, W, 3]");
            TORCH_CHECK(grid.is_cuda() && rgb.is_cuda(),
                        "Tensors must be on CUDA");

            auto output = torch::empty_like(rgb);

            // Call CUDA kernel
            bilateral_grid::slice_forward_cuda(
                grid.contiguous(),
                rgb.contiguous(),
                output,
                true // use uniform coordinates
            );

            ctx->save_for_backward({grid, rgb});
            return {output};
        }

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {
            auto saved = ctx->get_saved_variables();
            auto grid = saved[0];
            auto rgb = saved[1];
            auto grad_output = grad_outputs[0];

            auto [grad_grid, grad_rgb] = bilateral_grid::slice_backward_cuda(
                grid, rgb, grad_output.contiguous());

            return {grad_grid, grad_rgb};
        }
    };

    // Autograd function for total variation loss
    class BilateralGridTVLossFunction : public torch::autograd::Function<BilateralGridTVLossFunction> {
    public:
        static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor grids) {
            ctx->save_for_backward({grids});
            return bilateral_grid::tv_loss_forward_cuda(grids.contiguous());
        }

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {
            auto grids = ctx->get_saved_variables()[0];
            auto grad_output = grad_outputs[0];

            auto grad_grids = bilateral_grid::tv_loss_backward_cuda(
                grids, grad_output);

            return {grad_grids};
        }
    };

    // BilateralGrid implementation
    BilateralGrid::BilateralGrid(int num_images, int grid_W, int grid_H, int grid_L)
        : num_images_(num_images),
          grid_width_(grid_W),
          grid_height_(grid_H),
          grid_guidance_(grid_L) {

        // Initialize with identity transformation
        auto eye = torch::eye(4, torch::kFloat32).slice(0, 0, 3);
        auto grid = eye.repeat({grid_L * grid_H * grid_W, 1});
        grid = grid.reshape({1, grid_L, grid_H, grid_W, 12});
        grid = grid.permute({0, 4, 1, 2, 3});

        grids_ = grid.repeat({num_images, 1, 1, 1, 1}).to(torch::kCUDA);
        grids_.set_requires_grad(true);
    }

    torch::Tensor BilateralGrid::apply(const torch::Tensor& rgb, int image_idx) {
        TORCH_CHECK(image_idx >= 0 && image_idx < num_images_,
                    "Invalid image index: ", image_idx);

        // Handle different input formats
        torch::Tensor rgb_processed;
        if (rgb.dim() == 4 && rgb.size(0) == 1) {
            // Input is [1, C, H, W] - squeeze batch dimension
            rgb_processed = rgb.squeeze(0); // Now [C, H, W]
        } else if (rgb.dim() == 3) {
            // Input is already [C, H, W]
            rgb_processed = rgb;
        } else {
            TORCH_CHECK(false, "RGB must be [C, H, W] or [1, C, H, W], got ", rgb.sizes());
        }

        rgb_processed = torch::clamp(rgb_processed, 0, 1);
        // Convert from [C, H, W] to [H, W, C]
        auto rgb_hwc = rgb_processed.permute({1, 2, 0}).contiguous();

        // Apply bilateral grid
        auto grid = grids_[image_idx];
        auto output = BilateralGridSliceFunction::apply(grid, rgb_hwc)[0];

        // Convert back to [C, H, W]
        auto result = output.permute({2, 0, 1}).contiguous();

        // If input had batch dimension, add it back
        if (rgb.dim() == 4) {
            result = result.unsqueeze(0);
        }

        return result;
    }

    torch::Tensor BilateralGrid::tv_loss() const {
        return BilateralGridTVLossFunction::apply(grids_);
    }

    // ============= MANUAL FORWARD/BACKWARD IMPLEMENTATIONS (no autograd) =============

    std::pair<torch::Tensor, bilateral_grid::BilateralGridSliceContext> BilateralGrid::apply_forward(
        const torch::Tensor& rgb, int image_idx) {
        TORCH_CHECK(image_idx >= 0 && image_idx < num_images_,
                    "Invalid image index: ", image_idx);

        // Handle different input formats
        torch::Tensor rgb_processed;
        if (rgb.dim() == 4 && rgb.size(0) == 1) {
            // Input is [1, C, H, W] - squeeze batch dimension
            rgb_processed = rgb.squeeze(0); // Now [C, H, W]
        } else if (rgb.dim() == 3) {
            // Input is already [C, H, W]
            rgb_processed = rgb;
        } else {
            TORCH_CHECK(false, "RGB must be [C, H, W] or [1, C, H, W], got ", rgb.sizes());
        }

        rgb_processed = torch::clamp(rgb_processed, 0, 1);
        // Convert from [C, H, W] to [H, W, C]
        auto rgb_hwc = rgb_processed.permute({1, 2, 0}).contiguous();

        // Apply bilateral grid using manual forward
        auto grid = grids_[image_idx];
        auto [output, ctx] = bilateral_grid::bilateral_grid_slice_forward(grid, rgb_hwc);

        // Convert back to [C, H, W]
        auto result = output.permute({2, 0, 1}).contiguous();

        // If input had batch dimension, add it back
        if (rgb.dim() == 4) {
            result = result.unsqueeze(0);
        }

        return {result, ctx};
    }

    torch::Tensor BilateralGrid::apply_backward(
        const bilateral_grid::BilateralGridSliceContext& ctx,
        const torch::Tensor& grad_output,
        int image_idx) {
        TORCH_CHECK(image_idx >= 0 && image_idx < num_images_,
                    "Invalid image index: ", image_idx);

        // Handle different input formats (reverse of forward)
        torch::Tensor grad_processed;
        bool had_batch_dim = false;

        if (grad_output.dim() == 4 && grad_output.size(0) == 1) {
            grad_processed = grad_output.squeeze(0); // [C, H, W]
            had_batch_dim = true;
        } else if (grad_output.dim() == 3) {
            grad_processed = grad_output; // [C, H, W]
        } else {
            TORCH_CHECK(false, "grad_output must be [C, H, W] or [1, C, H, W], got ", grad_output.sizes());
        }

        // Convert from [C, H, W] to [H, W, C]
        auto grad_hwc = grad_processed.permute({1, 2, 0}).contiguous();

        // Compute gradients using manual backward
        auto [grad_grid, grad_rgb] = bilateral_grid::bilateral_grid_slice_backward(ctx, grad_hwc);

        // Accumulate grid gradients into the parameter
        if (!grids_.grad().defined()) {
            grids_.mutable_grad() = torch::zeros_like(grids_);
        }
        grids_.mutable_grad()[image_idx].add_(grad_grid);

        // Convert grad_rgb back to [C, H, W]
        auto result = grad_rgb.permute({2, 0, 1}).contiguous();

        // Add batch dimension back if needed
        if (had_batch_dim) {
            result = result.unsqueeze(0);
        }

        return result;
    }

    std::pair<float, bilateral_grid::BilateralGridTVContext> BilateralGrid::tv_loss_forward() const {
        return bilateral_grid::bilateral_grid_tv_forward(grids_.contiguous());
    }

    void BilateralGrid::tv_loss_backward(
        const bilateral_grid::BilateralGridTVContext& ctx,
        float grad_loss) {

        // Compute gradients
        auto grad_grids = bilateral_grid::bilateral_grid_tv_backward(ctx, grad_loss);

        // Accumulate into grids_ gradient
        if (!grids_.grad().defined()) {
            grids_.mutable_grad() = torch::zeros_like(grids_);
        }
        grids_.mutable_grad().add_(grad_grids);
    }

} // namespace gs::training