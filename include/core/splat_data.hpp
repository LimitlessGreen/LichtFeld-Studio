/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/point_cloud.hpp"
#include <expected>
#include <filesystem>
#include <future>
#include <glm/glm.hpp>
#include <mutex>
#include <string>
#include <torch/torch.h>
#include <vector>
#include <cuda_runtime.h>

namespace gs {
    namespace param {
        struct TrainingParameters;
    }

    class SplatData {
    public:
        SplatData() = default;
        ~SplatData();

        // Delete copy operations
        SplatData(const SplatData&) = delete;
        SplatData& operator=(const SplatData&) = delete;

        // Custom move operations (needed because of mutex and raw pointers)
        SplatData(SplatData&& other) noexcept;
        SplatData& operator=(SplatData&& other) noexcept;

        // Constructor
        SplatData(int sh_degree,
                  torch::Tensor means,
                  torch::Tensor sh0,
                  torch::Tensor shN,
                  torch::Tensor scaling,
                  torch::Tensor rotation,
                  torch::Tensor opacity,
                  float scene_scale);

        // Static factory method to create from PointCloud
        static std::expected<SplatData, std::string> init_model_from_pointcloud(
            const gs::param::TrainingParameters& params,
            torch::Tensor scene_center,
            const PointCloud& point_cloud);

        // Computed getters (return torch tensors for compatibility)
        torch::Tensor get_means() const;
        torch::Tensor get_opacity() const;
        torch::Tensor get_rotation() const;
        torch::Tensor get_scaling() const;
        torch::Tensor get_shs() const;

        // Transform (still needs implementation)
        SplatData& transform(const glm::mat4& transform_matrix);

        // Simple inline getters
        int get_active_sh_degree() const { return _active_sh_degree; }
        float get_scene_scale() const { return _scene_scale; }
        int64_t size() const { return _num_points; }

        // ========== TORCH INTERFACE (persistent tensor wrappers) ==========
        torch::Tensor& means() { return _means_tensor; }
        const torch::Tensor& means() const { return _means_tensor; }

        torch::Tensor& opacity_raw() { return _opacity_tensor; }
        const torch::Tensor& opacity_raw() const { return _opacity_tensor; }

        torch::Tensor& rotation_raw() { return _rotation_tensor; }
        const torch::Tensor& rotation_raw() const { return _rotation_tensor; }

        torch::Tensor& scaling_raw() { return _scaling_tensor; }
        const torch::Tensor& scaling_raw() const { return _scaling_tensor; }

        torch::Tensor& sh0() { return _sh0_tensor; }
        const torch::Tensor& sh0() const { return _sh0_tensor; }

        torch::Tensor& shN() { return _shN_tensor; }
        const torch::Tensor& shN() const { return _shN_tensor; }

        // ========== RAW INTERFACE (for fast_rasterizer) ==========
        float* means_cuda_ptr() { return _means_cuda; }
        const float* means_cuda_ptr() const { return _means_cuda; }

        float* opacity_raw_cuda_ptr() { return _opacity_cuda; }
        const float* opacity_raw_cuda_ptr() const { return _opacity_cuda; }

        float* rotation_raw_cuda_ptr() { return _rotation_cuda; }
        const float* rotation_raw_cuda_ptr() const { return _rotation_cuda; }

        float* scaling_raw_cuda_ptr() { return _scaling_cuda; }
        const float* scaling_raw_cuda_ptr() const { return _scaling_cuda; }

        float* sh0_cuda_ptr() { return _sh0_cuda; }
        const float* sh0_cuda_ptr() const { return _sh0_cuda; }

        float* shN_cuda_ptr() { return _shN_cuda; }
        const float* shN_cuda_ptr() const { return _shN_cuda; }

        // Gradient raw access
        float* means_grad_cuda_ptr() { return _means_grad_cuda; }
        float* opacity_grad_cuda_ptr() { return _opacity_grad_cuda; }
        float* rotation_grad_cuda_ptr() { return _rotation_grad_cuda; }
        float* scaling_grad_cuda_ptr() { return _scaling_grad_cuda; }
        float* sh0_grad_cuda_ptr() { return _sh0_grad_cuda; }
        float* shN_grad_cuda_ptr() { return _shN_grad_cuda; }

        // Utility methods
        void increment_sh_degree();

        // Ensure all parameter gradients are allocated
        void ensure_grad_allocated();

        // Manual zero_grad without autograd
        void zero_grad_manual();

        // Export methods - join_threads controls sync vs async
        void save_ply(const std::filesystem::path& root, int iteration, bool join_threads = true) const;
        void save_sog(const std::filesystem::path& root, int iteration, int kmeans_iterations = 10, bool join_threads = true) const;

        // Get attribute names for the PLY format
        std::vector<std::string> get_attribute_names() const;

    public:
        // Holds the magnitude of the screen space gradient (still torch for now)
        torch::Tensor _densification_info = torch::empty({0});

        // Store dimensions for creating torch wrappers
        int _max_sh_degree = 0;

    private:
        void allocate_cuda_memory_and_tensors(int64_t num_points,
                                              int64_t sh0_dim1, int64_t sh0_dim2,
                                              int64_t shN_dim1, int64_t shN_dim2);
        void free_cuda_memory();

        int _active_sh_degree = 0;
        float _scene_scale = 0.f;
        int64_t _num_points = 0;

        // Dimensions for tensor reconstruction
        int64_t _sh0_dims[3] = {0, 0, 0};  // [N, channels, coeffs]
        int64_t _shN_dims[3] = {0, 0, 0};  // [N, channels, coeffs]

        // Raw CUDA memory for data
        float* _means_cuda = nullptr;          // [N, 3]
        float* _sh0_cuda = nullptr;            // [N, 3, 1]
        float* _shN_cuda = nullptr;            // [N, 3, (max_degree+1)^2-1]
        float* _scaling_cuda = nullptr;        // [N, 3]
        float* _rotation_cuda = nullptr;       // [N, 4]
        float* _opacity_cuda = nullptr;        // [N, 1]

        // Raw CUDA memory for gradients
        float* _means_grad_cuda = nullptr;
        float* _sh0_grad_cuda = nullptr;
        float* _shN_grad_cuda = nullptr;
        float* _scaling_grad_cuda = nullptr;
        float* _rotation_grad_cuda = nullptr;
        float* _opacity_grad_cuda = nullptr;

        // Persistent tensor wrappers (non-owning views of raw memory)
        torch::Tensor _means_tensor;
        torch::Tensor _opacity_tensor;
        torch::Tensor _rotation_tensor;
        torch::Tensor _scaling_tensor;
        torch::Tensor _sh0_tensor;
        torch::Tensor _shN_tensor;

        // Async save management
        mutable std::mutex _save_mutex;
        mutable std::vector<std::future<void>> _save_futures;

        // Convert to point cloud for export
        PointCloud to_point_cloud() const;

        // Helper methods for async save management
        void wait_for_saves() const;
        void cleanup_finished_saves() const;
    };
} // namespace gs