/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fast_rasterizer.hpp"
#include "cuda_memory.hpp"
#include "kernels/training_kernels.cuh"
#include "rasterization_api.h"
#include <cuda_runtime.h>
#include <iostream> // Add for debugging
#include <stdexcept>

namespace gs::training {

    // Check the actual size of ForwardContext
    static_assert(sizeof(fast_gs::rasterization::ForwardContext) <= RenderOutput::FORWARD_CONTEXT_SIZE,
                  "ForwardContext size exceeds allocated storage. Update FORWARD_CONTEXT_SIZE in rasterizer.hpp");

    RenderOutput fast_rasterize(
        CameraNew& viewpoint_camera,
        SplatDataNew& gaussian_model,
        float* bg_color,
        TrainingMemory& cuda_memory) {

        // Get camera parameters
        const int width = static_cast<int>(viewpoint_camera.image_width());
        const int height = static_cast<int>(viewpoint_camera.image_height());
        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        // Ensure CUDA memory is allocated for this size
        cuda_memory.ensure_size(width, height, 3, 1);

        // Get raw Gaussian pointers - LibTorch-free!
        float* means_ptr = gaussian_model.means_raw().ptr<float>();
        float* raw_opacities_ptr = gaussian_model.opacity_raw().ptr<float>();
        float* raw_scales_ptr = gaussian_model.scaling_raw().ptr<float>();
        float* raw_rotations_ptr = gaussian_model.rotation_raw().ptr<float>();
        float* sh0_ptr = gaussian_model.sh0_raw().ptr<float>();
        float* shN_ptr = gaussian_model.shN_raw().ptr<float>();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);
        const int n_primitives = gaussian_model.size();
        const int total_bases_sh_rest = gaussian_model.shN_raw().size(1);

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        // Get raw camera pointers - LibTorch-free!
        const float* w2c_ptr = viewpoint_camera.world_view_transform().ptr<float>();
        const float* cam_position_ptr = viewpoint_camera.cam_position().ptr<float>();

        // Get DEDICATED render buffers (not gradient buffers!)
        float* image_buffer = cuda_memory.render_image();
        float* alpha_buffer = cuda_memory.render_alpha();

        // Clear buffers - IMPORTANT!
        cudaMemset(image_buffer, 0, 3 * width * height * sizeof(float));
        cudaMemset(alpha_buffer, 0, width * height * sizeof(float));

        // Debug: Check background values
        static int debug_counter = 0;
        if (debug_counter++ < 10) {
            float bg_cpu[3];
            cudaMemcpy(bg_cpu, bg_color, 3 * sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "Background RGB: [" << bg_cpu[0] << ", " << bg_cpu[1] << ", " << bg_cpu[2] << "]" << std::endl;
        }

        // Call raw CUDA implementation directly with raw pointers
        fast_gs::rasterization::ForwardContext forward_ctx = fast_gs::rasterization::forward_raw(
            means_ptr,
            raw_scales_ptr,
            raw_rotations_ptr,
            raw_opacities_ptr,
            sh0_ptr,
            shN_ptr,
            w2c_ptr,
            cam_position_ptr,
            image_buffer,
            alpha_buffer,
            n_primitives,
            active_sh_bases,
            total_bases_sh_rest,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near_plane,
            far_plane);

        // Debug: Check image values BEFORE blending
        if (debug_counter <= 10) {
            float sample_pixel[3];
            int center_pixel = (height / 2) * width + (width / 2);
            for (int c = 0; c < 3; ++c) {
                cudaMemcpy(&sample_pixel[c], image_buffer + c * width * height + center_pixel,
                           sizeof(float), cudaMemcpyDeviceToHost);
            }
            float sample_alpha;
            cudaMemcpy(&sample_alpha, alpha_buffer + center_pixel, sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "Before blend - Center pixel RGB: [" << sample_pixel[0] << ", "
                      << sample_pixel[1] << ", " << sample_pixel[2] << "], Alpha: " << sample_alpha << std::endl;
        }

        // Blend with background using our kernel from training_kernels.cu
        launch_blend_image_with_background(
            image_buffer,
            alpha_buffer,
            bg_color,
            width,
            height,
            0 // default stream
        );
        cudaDeviceSynchronize();

        // Debug: Check image values AFTER blending
        if (debug_counter <= 10) {
            float sample_pixel[3];
            int center_pixel = (height / 2) * width + (width / 2);
            for (int c = 0; c < 3; ++c) {
                cudaMemcpy(&sample_pixel[c], image_buffer + c * width * height + center_pixel,
                           sizeof(float), cudaMemcpyDeviceToHost);
            }
            std::cout << "After blend - Center pixel RGB: [" << sample_pixel[0] << ", "
                      << sample_pixel[1] << ", " << sample_pixel[2] << "]" << std::endl;
        }

        // Create output
        RenderOutput output;
        output.image = image_buffer;
        output.alpha = alpha_buffer;
        output.width = width;
        output.height = height;
        output.channels = 3;
        output.set_context(forward_ctx);

        return output;
    }

    void fast_rasterize_backward(
        const float* grad_image,
        const float* grad_alpha,
        const RenderOutput& render_output,
        SplatDataNew& gaussian_model,
        CameraNew& viewpoint_camera) {

        if (!render_output.has_context) {
            throw std::runtime_error("RenderOutput does not have a valid forward context for backward pass");
        }

        // Get the context
        const auto* forward_ctx = render_output.get_context<fast_gs::rasterization::ForwardContext>();

        // Get camera parameters
        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        // Get raw camera pointers - LibTorch-free!
        const float* w2c_ptr = viewpoint_camera.world_view_transform().ptr<float>();
        const float* cam_position_ptr = viewpoint_camera.cam_position().ptr<float>();

        const int n_primitives = gaussian_model.size();
        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);
        const int total_bases_sh_rest = gaussian_model.shN_raw().size(1);

        // Get raw gradient pointers - LibTorch-free!
        float* grad_means_ptr = gaussian_model.means_grad_ptr();
        float* grad_scales_raw_ptr = gaussian_model.scaling_grad_ptr();
        float* grad_rotations_raw_ptr = gaussian_model.rotation_grad_ptr();
        float* grad_opacities_raw_ptr = gaussian_model.opacity_grad_ptr();
        float* grad_sh0_ptr = gaussian_model.sh0_grad_ptr();
        float* grad_shN_ptr = gaussian_model.shN_grad_ptr();

        // We don't support camera gradients
        float* grad_w2c_ptr = nullptr;

        // Get raw data pointers for forward data - LibTorch-free!
        float* means_ptr = gaussian_model.means_raw().ptr<float>();
        float* scales_raw_ptr = gaussian_model.scaling_raw().ptr<float>();
        float* rotations_raw_ptr = gaussian_model.rotation_raw().ptr<float>();
        float* shN_ptr = gaussian_model.shN_raw().ptr<float>();

        // Call raw CUDA backward with raw pointers
        fast_gs::rasterization::BackwardOutputs outputs = fast_gs::rasterization::backward_raw(
            gaussian_model.densification_info.is_valid() ? gaussian_model.densification_info.ptr<float>() : nullptr,
            grad_image,
            grad_alpha,
            render_output.image,
            render_output.alpha,
            means_ptr,
            scales_raw_ptr,
            rotations_raw_ptr,
            shN_ptr,
            w2c_ptr,
            cam_position_ptr,
            *forward_ctx,
            grad_means_ptr,
            grad_scales_raw_ptr,
            grad_rotations_raw_ptr,
            grad_opacities_raw_ptr,
            grad_sh0_ptr,
            grad_shN_ptr,
            grad_w2c_ptr,
            n_primitives,
            active_sh_bases,
            total_bases_sh_rest,
            render_output.width,
            render_output.height,
            fx,
            fy,
            cx,
            cy);

        if (!outputs.success) {
            throw std::runtime_error(std::string("Backward pass failed: ") +
                                     (outputs.error_message ? outputs.error_message : "Unknown error"));
        }
    }
} // namespace gs::training