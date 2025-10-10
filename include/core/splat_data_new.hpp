/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/point_cloud_new.hpp"
#include "core/tensor.hpp"
#include <expected>
#include <filesystem>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace gs {
    namespace param {
        struct TrainingParameters;
    }

    // Forward declaration for sogs_new
    namespace core {
        struct SogWriteOptionsNew;
    }

    /**
     * @brief Modern SplatData implementation using Tensor library (no libtorch)
     *
     * This replaces the old SplatData class with a pure Tensor-based implementation.
     * All parameters are stored as Tensors on GPU, with automatic gradient management.
     */
    class SplatDataNew {
    public:
        // Core Gaussian parameters (all on GPU)
        Tensor means;      // [N, 3] float32 - positions
        Tensor opacity;    // [N, 1] float32 - raw (before sigmoid)
        Tensor rotation;   // [N, 4] float32 - quaternion (raw, before normalization)
        Tensor scaling;    // [N, 3] float32 - log(scale)
        Tensor sh0;        // [N, 3, 1] float32 - DC spherical harmonics
        Tensor shN;        // [N, 3, K] float32 - higher-order SH coefficients

        // Training state
        Tensor densification_info; // [N] float32 - gradient magnitude for densification

        // Metadata
        int max_sh_degree = 0;
        int active_sh_degree = 0;
        float scene_scale = 0.0f;

        // Default constructor
        SplatDataNew() = default;

        /**
         * @brief Constructor from tensors
         */
        SplatDataNew(int sh_degree,
                     Tensor means,
                     Tensor sh0,
                     Tensor shN,
                     Tensor scaling,
                     Tensor rotation,
                     Tensor opacity,
                     float scene_scale);

        // Rule of 5: use default implementations since Tensor handles memory
        ~SplatDataNew() = default;
        SplatDataNew(const SplatDataNew&) = delete;
        SplatDataNew& operator=(const SplatDataNew&) = delete;
        SplatDataNew(SplatDataNew&&) noexcept = default;
        SplatDataNew& operator=(SplatDataNew&&) noexcept = default;

        /**
         * @brief Initialize from point cloud
         */
        static std::expected<SplatDataNew, std::string> init_model_from_pointcloud(
            const param::TrainingParameters& params,
            const Tensor& scene_center,
            const PointCloudNew& point_cloud);

        // Computed getters (apply activations) - defined in .cpp
        Tensor get_means() const { return means; }
        Tensor get_opacity() const;
        Tensor get_rotation() const;
        Tensor get_scaling() const;
        Tensor get_shs() const;

        // Raw getters (for SOG export which needs pre-activation values)
        const Tensor& means_raw() const { return means; }
        const Tensor& opacity_raw() const { return opacity; }
        const Tensor& rotation_raw() const { return rotation; }
        const Tensor& scaling_raw() const { return scaling; }
        const Tensor& sh0_raw() const { return sh0; }
        const Tensor& shN_raw() const { return shN; }

        // Simple getters
        int get_active_sh_degree() const { return active_sh_degree; }
        int get_max_sh_degree() const { return max_sh_degree; }
        float get_scene_scale() const { return scene_scale; }
        size_t size() const { return means.is_valid() ? means.size(0) : 0; }

        // Modifiers
        void increment_sh_degree();
        SplatDataNew& transform(const glm::mat4& transform_matrix);

        // Export methods
        void save_ply(const std::filesystem::path& root, int iteration) const;
        void save_sog(const std::filesystem::path& root, int iteration, int kmeans_iterations = 10) const;

        // Conversion
        PointCloudNew to_point_cloud() const;

        /**
         * @brief Get attribute names for PLY format
         */
        std::vector<std::string> get_attribute_names() const;

        /**
         * @brief Validate the splat data structure
         */
        bool is_valid() const;

        /**
         * @brief Get diagnostic string
         */
        std::string str() const;
    };

} // namespace gs