/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "geometry/bounding_box.hpp"
#include <cstring>
#include <memory>

namespace gs::training {
    // Raw render output without torch dependency
    struct RenderOutput {
        // Raw pointers to CUDA memory
        float* image = nullptr; // [C, H, W] in CUDA memory
        float* alpha = nullptr; // [1, H, W] in CUDA memory

        // Dimensions
        int width = 0;
        int height = 0;
        int channels = 3;

        // Store forward context as raw bytes for backward pass
        static constexpr size_t FORWARD_CONTEXT_SIZE = 256;
        alignas(8) char forward_context_storage[FORWARD_CONTEXT_SIZE];
        bool has_context = false;

        // Helper to get context as the actual type
        template <typename T>
        T* get_context() {
            static_assert(sizeof(T) <= FORWARD_CONTEXT_SIZE, "Context size mismatch");
            return reinterpret_cast<T*>(forward_context_storage);
        }

        template <typename T>
        const T* get_context() const {
            static_assert(sizeof(T) <= FORWARD_CONTEXT_SIZE, "Context size mismatch");
            return reinterpret_cast<const T*>(forward_context_storage);
        }

        template <typename T>
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
} // namespace gs::training