/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "geometry/bounding_box.hpp"
#include <torch/torch.h>
#include <memory>
#include <cstring>

namespace gs::training {
    struct RenderOutput {
        torch::Tensor image;      // [..., channels, H, W]
        torch::Tensor alpha;      // [..., C, H, W, 1]
        torch::Tensor depth;      // [..., C, H, W, 1] - accumulated or expected depth
        torch::Tensor means2d;    // [..., C, N, 2]
        torch::Tensor depths;     // [..., N] - per-gaussian depths
        torch::Tensor radii;      // [..., N]
        torch::Tensor visibility; // [..., N]
        int width;
        int height;

        // Store forward context as raw bytes to avoid incomplete type issues
        // We know ForwardContext is a POD struct with fixed size
        static constexpr size_t FORWARD_CONTEXT_SIZE = 256; // Adjust based on actual size
        alignas(8) char forward_context_storage[FORWARD_CONTEXT_SIZE];
        bool has_context = false;

        // Helper to get context as the actual type (used only in fast_rasterizer.cpp where type is complete)
        template<typename T>
        T* get_context() {
            static_assert(sizeof(T) <= FORWARD_CONTEXT_SIZE, "Context size mismatch");
            return reinterpret_cast<T*>(forward_context_storage);
        }

        template<typename T>
        const T* get_context() const {
            static_assert(sizeof(T) <= FORWARD_CONTEXT_SIZE, "Context size mismatch");
            return reinterpret_cast<const T*>(forward_context_storage);
        }

        template<typename T>
        void set_context(const T& ctx) {
            static_assert(sizeof(T) <= FORWARD_CONTEXT_SIZE, "Context size mismatch");
            std::memcpy(forward_context_storage, &ctx, sizeof(T));
            has_context = true;
        }
    };

    enum class RenderMode {
        RGB,            // Color only
        D,              // Accumulated depth only
        ED,             // Expected depth only
        RGB_D,          // Color + accumulated depth
        RGB_ED,         // Color + expected depth
        POINT_CLOUD = 5 // Point cloud rendering mode
    };

    // Helper function to check if render mode includes depth
    static inline bool renderModeHasDepth(RenderMode mode) {
        return mode != RenderMode::RGB;
    }

    // Helper function to check if render mode includes RGB
    static inline bool renderModeHasRGB(RenderMode mode) {
        return mode == RenderMode::RGB ||
               mode == RenderMode::RGB_D ||
               mode == RenderMode::RGB_ED;
    }

    static inline RenderMode stringToRenderMode(const std::string& mode) {
        if (mode == "RGB")
            return RenderMode::RGB;
        else if (mode == "D")
            return RenderMode::D;
        else if (mode == "ED")
            return RenderMode::ED;
        else if (mode == "RGB_D")
            return RenderMode::RGB_D;
        else if (mode == "RGB_ED")
            return RenderMode::RGB_ED;
        else
            throw std::runtime_error("Invalid render mode: " + mode);
    }

    // Wrapper function to use gsplat backend for rendering
    RenderOutput rasterize(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier = 1.0,
        bool packed = false,
        bool antialiased = false,
        RenderMode render_mode = RenderMode::RGB,
        const gs::geometry::BoundingBox* = nullptr);
} // namespace gs::training