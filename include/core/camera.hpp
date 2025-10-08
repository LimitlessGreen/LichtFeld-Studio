/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "Common.h"
#include <array>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <future>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace gs {

    class Camera {
    public:
        Camera() = default;

        Camera(const torch::Tensor& R,
               const torch::Tensor& T,
               float focal_x, float focal_y,
               float center_x, float center_y,
               torch::Tensor radial_distortion,
               torch::Tensor tangential_distortion,
               gsplat::CameraModelType camera_model_type,
               const std::string& image_name,
               const std::filesystem::path& image_path,
               int camera_width, int camera_height,
               int uid);
        Camera(const Camera&, const torch::Tensor& transform);

        // Delete copy, allow move
        Camera(const Camera&) = delete;
        Camera& operator=(const Camera&) = delete;
        Camera(Camera&&) noexcept;
        Camera& operator=(Camera&&) noexcept;

        ~Camera();

        // Initialize GPU tensors on demand
        void initialize_cuda_tensors();

        // Load image from disk just to populate _image_width/_image_height
        void load_image_size(int resize_factor = -1);

        // Get number of bytes in the image file
        size_t get_num_bytes_from_file() const;

        // ========== TORCH INTERFACE (for compatibility) ==========
        torch::Tensor world_view_transform() const;
        torch::Tensor cam_position() const;
        torch::Tensor R() const;
        torch::Tensor T() const;
        torch::Tensor K() const;
        torch::Tensor radial_distortion() const noexcept;
        torch::Tensor tangential_distortion() const noexcept;

        // ========== RAW INTERFACE (for fast_rasterizer) ==========
        // These return GPU pointers directly - no torch overhead
        const float* world_view_transform_cuda_ptr() const {
            ensure_cuda_allocated();
            return _world_view_transform_cuda;
        }

        const float* R_cpu_ptr() const { return _R_cpu.data(); }
        const float* T_cpu_ptr() const { return _T_cpu.data(); }
        const float* cam_position_cuda_ptr() const {
            ensure_cuda_allocated();
            return _cam_position_cuda;
        }

        // Get intrinsics directly
        std::tuple<float, float, float, float> get_intrinsics() const;

        // Common getters
        int image_height() const noexcept { return _image_height; }
        int image_width() const noexcept { return _image_width; }
        int camera_height() const noexcept { return _camera_height; }
        int camera_width() const noexcept { return _camera_width; }
        float focal_x() const noexcept { return _focal_x; }
        float focal_y() const noexcept { return _focal_y; }
        gsplat::CameraModelType camera_model_type() const noexcept { return _camera_model_type; }
        const std::string& image_name() const noexcept { return _image_name; }
        const std::filesystem::path& image_path() const noexcept { return _image_path; }
        int uid() const noexcept { return _uid; }

        float FoVx() const noexcept { return _FoVx; }
        float FoVy() const noexcept { return _FoVy; }

    private:
        // Helper functions for matrix operations
        void compute_world_view_transform();
        void compute_camera_position();
        void ensure_cuda_allocated() const;
        void free_cuda_memory();

        // IDs and intrinsics
        float _FoVx = 0.f;
        float _FoVy = 0.f;
        int _uid = -1;
        float _focal_x = 0.f;
        float _focal_y = 0.f;
        float _center_x = 0.f;
        float _center_y = 0.f;

        // Store R and T as raw CPU arrays (3x3 and 3x1)
        std::array<float, 9> _R_cpu;
        std::array<float, 3> _T_cpu;

        // Store distortion parameters as vectors
        std::vector<float> _radial_distortion_cpu;
        std::vector<float> _tangential_distortion_cpu;
        gsplat::CameraModelType _camera_model_type = gsplat::CameraModelType::PINHOLE;

        // Image info
        std::string _image_name;
        std::filesystem::path _image_path;
        int _camera_width = 0;
        int _camera_height = 0;
        int _image_width = 0;
        int _image_height = 0;

        // CPU storage (computed once)
        std::array<float, 16> _world_view_transform_cpu; // 4x4 matrix
        std::array<float, 3> _cam_position_cpu;

        // GPU storage (allocated on demand)
        mutable float* _world_view_transform_cuda = nullptr;
        mutable float* _cam_position_cuda = nullptr;
        mutable bool _cuda_allocated = false;

        // CUDA stream for async operations
        at::cuda::CUDAStream _stream = at::cuda::getStreamFromPool(false);
    };

    inline float focal2fov(float focal, int pixels) {
        return 2.0f * std::atan(pixels / (2.0f * focal));
    }

    inline float fov2focal(float fov, int pixels) {
        float tan_fov = std::tan(fov * 0.5f);
        return pixels / (2.0f * tan_fov);
    }

} // namespace gs