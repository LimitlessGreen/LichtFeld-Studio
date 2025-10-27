/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backward.h"
#include "forward.h"
#include "helper_math.h"
#include "rasterization_api.h"
#include "rasterization_config.h"
#include "rasterizer_memory_arena.h"
#include "buffer_utils.h"
#include "utils.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <functional>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace fast_gs::rasterization {

ForwardContext forward_raw(
    const float* means_ptr,
    const float* scales_raw_ptr,
    const float* rotations_raw_ptr,
    const float* opacities_raw_ptr,
    const float* sh_coefficients_0_ptr,
    const float* sh_coefficients_rest_ptr,
    const float* w2c_ptr,
    const float* cam_position_ptr,
    float* image_ptr,
    float* alpha_ptr,
    int n_primitives,
    int active_sh_bases,
    int total_bases_sh_rest,
    int width,
    int height,
    float focal_x,
    float focal_y,
    float center_x,
    float center_y,
    float near_plane,
    float far_plane) {

    // Validate inputs using pure CUDA validation
    CHECK_CUDA_PTR(means_ptr, "means_ptr");
    CHECK_CUDA_PTR(scales_raw_ptr, "scales_raw_ptr");
    CHECK_CUDA_PTR(rotations_raw_ptr, "rotations_raw_ptr");
    CHECK_CUDA_PTR(opacities_raw_ptr, "opacities_raw_ptr");
    CHECK_CUDA_PTR(sh_coefficients_0_ptr, "sh_coefficients_0_ptr");
    CHECK_CUDA_PTR(sh_coefficients_rest_ptr, "sh_coefficients_rest_ptr");
    CHECK_CUDA_PTR(w2c_ptr, "w2c_ptr");
    CHECK_CUDA_PTR(cam_position_ptr, "cam_position_ptr");
    CHECK_CUDA_PTR(image_ptr, "image_ptr");
    CHECK_CUDA_PTR(alpha_ptr, "alpha_ptr");

    if (n_primitives <= 0 || width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid dimensions in forward_raw");
    }

    // Calculate grid dimensions
    const dim3 grid(div_round_up(width, config::tile_width),
                    div_round_up(height, config::tile_height), 1);
    const int n_tiles = grid.x * grid.y;

    // Get global arena and begin frame
    auto& arena = GlobalArenaManager::instance().get_arena();
    uint64_t frame_id = arena.begin_frame();

    // Get arena allocator for this frame
    auto arena_allocator = arena.get_allocator(frame_id);

    // Allocate buffers through arena
    size_t per_primitive_size = required<PerPrimitiveBuffers>(n_primitives);
    size_t per_tile_size = required<PerTileBuffers>(n_tiles);

    char* per_primitive_buffers_blob = arena_allocator(per_primitive_size);
    char* per_tile_buffers_blob = arena_allocator(per_tile_size);

    if (!per_primitive_buffers_blob || !per_tile_buffers_blob) {
        arena.end_frame(frame_id);
        throw std::runtime_error("Failed to allocate buffers from arena");
    }

    // Allocate helper buffers for backward pass upfront to avoid allocation failures later
    const size_t grad_mean2d_size = n_primitives * 2 * sizeof(float);
    const size_t grad_conic_size = n_primitives * 3 * sizeof(float);

    char* grad_mean2d_helper = arena_allocator(grad_mean2d_size);
    char* grad_conic_helper = arena_allocator(grad_conic_size);

    if (!grad_mean2d_helper || !grad_conic_helper) {
        arena.end_frame(frame_id);
        throw std::runtime_error("Failed to allocate backward helper buffers from arena");
    }

    // Create allocation wrappers
    std::function<char*(size_t)> per_primitive_buffers_func =
        [&per_primitive_buffers_blob](size_t size) -> char* {
            // Already allocated, just return the pointer
            return per_primitive_buffers_blob;
        };

    std::function<char*(size_t)> per_tile_buffers_func =
        [&per_tile_buffers_blob](size_t size) -> char* {
            return per_tile_buffers_blob;
        };

    // These will be allocated later based on n_instances
    char* per_instance_buffers_blob = nullptr;
    char* per_bucket_buffers_blob = nullptr;
    size_t per_instance_size = 0;
    size_t per_bucket_size = 0;

    std::function<char*(size_t)> per_instance_buffers_func =
        [&arena_allocator, &per_instance_buffers_blob, &per_instance_size](size_t size) -> char* {
            per_instance_size = size;
            per_instance_buffers_blob = arena_allocator(size);
            if (!per_instance_buffers_blob) {
                throw std::runtime_error("Failed to allocate instance buffers");
            }
            return per_instance_buffers_blob;
        };

    std::function<char*(size_t)> per_bucket_buffers_func =
        [&arena_allocator, &per_bucket_buffers_blob, &per_bucket_size](size_t size) -> char* {
            per_bucket_size = size;
            per_bucket_buffers_blob = arena_allocator(size);
            if (!per_bucket_buffers_blob) {
                throw std::runtime_error("Failed to allocate bucket buffers");
            }
            return per_bucket_buffers_blob;
        };

    try {
        // Call the actual forward implementation
        auto [n_visible_primitives, n_instances, n_buckets,
              primitive_primitive_indices_selector,
              instance_primitive_indices_selector] = forward(
            per_primitive_buffers_func,
            per_tile_buffers_func,
            per_instance_buffers_func,
            per_bucket_buffers_func,
            reinterpret_cast<const float3*>(means_ptr),
            reinterpret_cast<const float3*>(scales_raw_ptr),
            reinterpret_cast<const float4*>(rotations_raw_ptr),
            opacities_raw_ptr,
            reinterpret_cast<const float3*>(sh_coefficients_0_ptr),
            reinterpret_cast<const float3*>(sh_coefficients_rest_ptr),
            reinterpret_cast<const float4*>(w2c_ptr),
            reinterpret_cast<const float3*>(cam_position_ptr),
            image_ptr,
            alpha_ptr,
            n_primitives,
            active_sh_bases,
            total_bases_sh_rest,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            near_plane,
            far_plane);

        // Verify allocations happened
        if (n_instances > 0 && !per_instance_buffers_blob) {
            arena.end_frame(frame_id);
            throw std::runtime_error("Instance buffers were not allocated despite n_instances > 0");
        }
        if (n_buckets > 0 && !per_bucket_buffers_blob) {
            arena.end_frame(frame_id);
            throw std::runtime_error("Bucket buffers were not allocated despite n_buckets > 0");
        }

        // Create and return context
        ForwardContext ctx;
        ctx.per_primitive_buffers = per_primitive_buffers_blob;
        ctx.per_tile_buffers = per_tile_buffers_blob;
        ctx.per_instance_buffers = per_instance_buffers_blob;
        ctx.per_bucket_buffers = per_bucket_buffers_blob;
        ctx.per_primitive_buffers_size = per_primitive_size;
        ctx.per_tile_buffers_size = per_tile_size;
        ctx.per_instance_buffers_size = per_instance_size;
        ctx.per_bucket_buffers_size = per_bucket_size;
        ctx.n_visible_primitives = n_visible_primitives;
        ctx.n_instances = n_instances;
        ctx.n_buckets = n_buckets;
        ctx.primitive_primitive_indices_selector = primitive_primitive_indices_selector;
        ctx.instance_primitive_indices_selector = instance_primitive_indices_selector;
        ctx.frame_id = frame_id;
        ctx.grad_mean2d_helper = grad_mean2d_helper;
        ctx.grad_conic_helper = grad_conic_helper;

        return ctx;

    } catch (const std::exception& e) {
        // Clean up frame on error
        arena.end_frame(frame_id);
        throw;
    }
}

BackwardOutputs backward_raw(
    float* densification_info_ptr,
    const float* grad_image_ptr,
    const float* grad_alpha_ptr,
    const float* image_ptr,
    const float* alpha_ptr,
    const float* means_ptr,
    const float* scales_raw_ptr,
    const float* rotations_raw_ptr,
    const float* sh_coefficients_rest_ptr,
    const float* w2c_ptr,
    const float* cam_position_ptr,
    const ForwardContext& forward_ctx,
    float* grad_means_ptr,
    float* grad_scales_raw_ptr,
    float* grad_rotations_raw_ptr,
    float* grad_opacities_raw_ptr,
    float* grad_sh_coefficients_0_ptr,
    float* grad_sh_coefficients_rest_ptr,
    float* grad_w2c_ptr,
    int n_primitives,
    int active_sh_bases,
    int total_bases_sh_rest,
    int width,
    int height,
    float focal_x,
    float focal_y,
    float center_x,
    float center_y) {

    BackwardOutputs outputs;
    outputs.success = false;
    outputs.error_message = nullptr;

    // Validate required inputs using pure CUDA validation
    CHECK_CUDA_PTR(grad_image_ptr, "grad_image_ptr");
    CHECK_CUDA_PTR(grad_alpha_ptr, "grad_alpha_ptr");
    CHECK_CUDA_PTR(image_ptr, "image_ptr");
    CHECK_CUDA_PTR(alpha_ptr, "alpha_ptr");
    CHECK_CUDA_PTR(means_ptr, "means_ptr");
    CHECK_CUDA_PTR(scales_raw_ptr, "scales_raw_ptr");
    CHECK_CUDA_PTR(rotations_raw_ptr, "rotations_raw_ptr");
    CHECK_CUDA_PTR(sh_coefficients_rest_ptr, "sh_coefficients_rest_ptr");
    CHECK_CUDA_PTR(w2c_ptr, "w2c_ptr");
    CHECK_CUDA_PTR(cam_position_ptr, "cam_position_ptr");

    // Validate required outputs
    CHECK_CUDA_PTR(grad_means_ptr, "grad_means_ptr");
    CHECK_CUDA_PTR(grad_scales_raw_ptr, "grad_scales_raw_ptr");
    CHECK_CUDA_PTR(grad_rotations_raw_ptr, "grad_rotations_raw_ptr");
    CHECK_CUDA_PTR(grad_opacities_raw_ptr, "grad_opacities_raw_ptr");
    CHECK_CUDA_PTR(grad_sh_coefficients_0_ptr, "grad_sh_coefficients_0_ptr");
    CHECK_CUDA_PTR(grad_sh_coefficients_rest_ptr, "grad_sh_coefficients_rest_ptr");
    
    // Optional pointer
    CHECK_CUDA_PTR_OPTIONAL(densification_info_ptr, "densification_info_ptr");
    CHECK_CUDA_PTR_OPTIONAL(grad_w2c_ptr, "grad_w2c_ptr");

    // Validate forward context
    if (!forward_ctx.per_primitive_buffers || !forward_ctx.per_tile_buffers) {
        outputs.error_message = "Invalid forward context buffers";
        return outputs;
    }

    if (forward_ctx.n_instances > 0 && !forward_ctx.per_instance_buffers) {
        outputs.error_message = "Missing instance buffers in forward context";
        return outputs;
    }

    if (forward_ctx.n_buckets > 0 && !forward_ctx.per_bucket_buffers) {
        outputs.error_message = "Missing bucket buffers in forward context";
        return outputs;
    }

    // Use pre-allocated helper buffers from forward context
    if (!forward_ctx.grad_mean2d_helper || !forward_ctx.grad_conic_helper) {
        outputs.error_message = "Missing pre-allocated helper buffers in forward context";
        return outputs;
    }

    float* grad_mean2d_helper = static_cast<float*>(forward_ctx.grad_mean2d_helper);
    float* grad_conic_helper = static_cast<float*>(forward_ctx.grad_conic_helper);

    // Zero out helper buffers
    const size_t grad_mean2d_size = n_primitives * 2 * sizeof(float);
    const size_t grad_conic_size = n_primitives * 3 * sizeof(float);
    cudaMemset(grad_mean2d_helper, 0, grad_mean2d_size);
    cudaMemset(grad_conic_helper, 0, grad_conic_size);

    // Zero out output gradients
    cudaMemset(grad_means_ptr, 0, n_primitives * 3 * sizeof(float));
    cudaMemset(grad_scales_raw_ptr, 0, n_primitives * 3 * sizeof(float));
    cudaMemset(grad_rotations_raw_ptr, 0, n_primitives * 4 * sizeof(float));
    cudaMemset(grad_opacities_raw_ptr, 0, n_primitives * sizeof(float));
    cudaMemset(grad_sh_coefficients_0_ptr, 0, n_primitives * 3 * sizeof(float));
    cudaMemset(grad_sh_coefficients_rest_ptr, 0,
               n_primitives * total_bases_sh_rest * 3 * sizeof(float));

    if (grad_w2c_ptr) {
        cudaMemset(grad_w2c_ptr, 0, 4 * 4 * sizeof(float));
    }

    try {
        // Call the actual backward implementation
        backward(
            grad_image_ptr,
            grad_alpha_ptr,
            image_ptr,
            alpha_ptr,
            reinterpret_cast<const float3*>(means_ptr),
            reinterpret_cast<const float3*>(scales_raw_ptr),
            reinterpret_cast<const float4*>(rotations_raw_ptr),
            reinterpret_cast<const float3*>(sh_coefficients_rest_ptr),
            reinterpret_cast<const float4*>(w2c_ptr),
            reinterpret_cast<const float3*>(cam_position_ptr),
            static_cast<char*>(forward_ctx.per_primitive_buffers),
            static_cast<char*>(forward_ctx.per_tile_buffers),
            static_cast<char*>(forward_ctx.per_instance_buffers),
            static_cast<char*>(forward_ctx.per_bucket_buffers),
            reinterpret_cast<float3*>(grad_means_ptr),
            reinterpret_cast<float3*>(grad_scales_raw_ptr),
            reinterpret_cast<float4*>(grad_rotations_raw_ptr),
            grad_opacities_raw_ptr,
            reinterpret_cast<float3*>(grad_sh_coefficients_0_ptr),
            reinterpret_cast<float3*>(grad_sh_coefficients_rest_ptr),
            reinterpret_cast<float2*>(grad_mean2d_helper),
            grad_conic_helper,
            grad_w2c_ptr ? reinterpret_cast<float4*>(grad_w2c_ptr) : nullptr,
            densification_info_ptr,
            n_primitives,
            forward_ctx.n_visible_primitives,
            forward_ctx.n_instances,
            forward_ctx.n_buckets,
            forward_ctx.primitive_primitive_indices_selector,
            forward_ctx.instance_primitive_indices_selector,
            active_sh_bases,
            total_bases_sh_rest,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y);

        // Mark frame as complete
        auto& arena = GlobalArenaManager::instance().get_arena();
        arena.end_frame(forward_ctx.frame_id);

        outputs.success = true;
        return outputs;

    } catch (const std::exception& e) {
        // Clean up on error
        auto& arena = GlobalArenaManager::instance().get_arena();
        arena.end_frame(forward_ctx.frame_id);

        outputs.error_message = e.what();
        return outputs;
    }
}

} // namespace fast_gs::rasterization
