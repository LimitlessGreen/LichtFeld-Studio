/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/point_cloud_new.hpp"
#include "core/tensor.hpp"
#include <expected>
#include <filesystem>
#include <future>
#include <geometry/bounding_box.hpp>
#include <glm/glm.hpp>
#include <mutex>
#include <string>
#include <vector>

namespace gs {
    namespace param {
        struct TrainingParameters;
    }

    /**
     * @brief Gaussian Splat data structure using custom Tensor library (no libtorch)
     *
     * This is a modern replacement for SplatData that uses our custom Tensor
     * implementation instead of torch tensors.
     */
    class SplatDataNew {
    public:
        SplatDataNew() = default;
        ~SplatDataNew();

        // Delete copy operations - but add debug messages if accidentally called
        SplatDataNew(const SplatDataNew&) = delete;
        SplatDataNew& operator=(const SplatDataNew&) = delete;

        // Custom move operations (needed because of mutex)
        SplatDataNew(SplatDataNew&& other) noexcept;
        SplatDataNew& operator=(SplatDataNew&& other) noexcept;

        /**
         * @brief Constructor from tensors
         * @param sh_degree Maximum spherical harmonics degree
         * @param means Gaussian positions [N, 3]
         * @param sh0 DC spherical harmonics coefficients [N, 1, 3] (coeffs, channels) - ACTUAL REF FORMAT
         * @param shN Higher-order SH coefficients [N, (degree+1)^2-1, 3] (coeffs, channels) - ACTUAL REF FORMAT
         * @param scaling Log-space scale values [N, 3]
         * @param rotation Quaternion rotations [N, 4] (unnormalized)
         * @param opacity Logit-space opacity values [N, 1]
         * @param scene_scale Scene scaling factor
         */
        SplatDataNew(int sh_degree,
                     Tensor means,
                     Tensor sh0,
                     Tensor shN,
                     Tensor scaling,
                     Tensor rotation,
                     Tensor opacity,
                     float scene_scale);

        /**
         * @brief Static factory method to create from PointCloudNew
         * @param params Training parameters
         * @param scene_center Center of the scene [3]
         * @param point_cloud Point cloud data
         * @return SplatDataNew instance or error message
         */
        static std::expected<SplatDataNew, std::string> init_model_from_pointcloud(
            const param::TrainingParameters& params,
            const Tensor& scene_center,
            const PointCloudNew& point_cloud);

        // ========== COMPUTED GETTERS ==========

        /**
         * @brief Get gaussian positions (no transformation applied)
         * @return Tensor [N, 3]
         */
        Tensor get_means() const;

        /**
         * @brief Get opacity values with sigmoid activation
         * @return Tensor [N] (squeezed)
         */
        Tensor get_opacity() const;

        /**
         * @brief Get normalized rotation quaternions
         * @return Tensor [N, 4]
         */
        Tensor get_rotation() const;

        /**
         * @brief Get scale values with exp activation
         * @return Tensor [N, 3]
         */
        Tensor get_scaling() const;

        /**
         * @brief Get concatenated SH coefficients (sh0 + shN)
         * @return Tensor [N, (degree+1)^2, 3] (coeffs, channels) - ACTUAL REF FORMAT
         */
        Tensor get_shs() const;

        // ========== TRANSFORMATION ==========

        /**
         * @brief Apply transformation matrix to gaussians
         * @param transform_matrix 4x4 transformation matrix
         * @return Reference to this for chaining
         */
        SplatDataNew& transform(const glm::mat4& transform_matrix);

        // ========== SIMPLE GETTERS ==========

        int get_active_sh_degree() const { return active_sh_degree; }
        int get_max_sh_degree() const { return max_sh_degree; }
        float get_scene_scale() const { return scene_scale; }
        int64_t size() const { return means.is_valid() ? means.size(0) : 0; }

        // ========== RAW TENSOR ACCESS ==========

        Tensor& means_raw() { return means; }
        const Tensor& means_raw() const { return means; }

        Tensor& opacity_raw() { return opacity; }
        const Tensor& opacity_raw() const { return opacity; }

        Tensor& rotation_raw() { return rotation; }
        const Tensor& rotation_raw() const { return rotation; }

        Tensor& scaling_raw() { return scaling; }
        const Tensor& scaling_raw() const { return scaling; }

        Tensor& sh0_raw() { return sh0; }
        const Tensor& sh0_raw() const { return sh0; }

        Tensor& shN_raw() { return shN; }
        const Tensor& shN_raw() const { return shN; }

        // ========== UTILITY METHODS ==========

        /**
         * @brief Increment active SH degree (capped at max_sh_degree)
         */
        void increment_sh_degree();

        /**
         * @brief Set active SH degree (capped at max_sh_degree)
         * @param sh_degree Desired SH degree
         */
        void set_active_sh_degree(int sh_degree);

        /**
         * @brief Get attribute names for PLY format
         * @return Vector of attribute names
         */
        std::vector<std::string> get_attribute_names() const;

        /**
         * @brief Validate internal state
         * @return True if all tensors are valid and have correct shapes
         */
        bool is_valid() const;

        // ========== EXPORT METHODS ==========

        /**
         * @brief Save as PLY file
         * @param root Output directory
         * @param iteration Iteration number (for filename)
         * @param join_threads If true, wait for completion (synchronous)
         * @param stem Optional filename stem (default: "splat_{iteration}")
         */
        void save_ply(const std::filesystem::path& root,
                      int iteration,
                      bool join_threads = true,
                      std::string stem = "") const;

        /**
         * @brief Save as SOG format
         * @param root Output directory
         * @param iteration Iteration number (for filename)
         * @param kmeans_iterations Number of k-means iterations
         * @param join_threads If true, wait for completion (synchronous)
         * @return Path to saved SOG file
         */
        std::filesystem::path save_sog(const std::filesystem::path& root,
                                       int iteration,
                                       int kmeans_iterations = 10,
                                       bool join_threads = true) const;

        /**
         * @brief Crop gaussians by bounding box
         * @param bounding_box Bounding box to crop with
         * @return New SplatDataNew with cropped gaussians
         */
        SplatDataNew crop_by_cropbox(const geometry::BoundingBox& bounding_box) const;

        /**
         * @brief Convert to PointCloudNew for export
         * @return PointCloudNew instance
         */
        PointCloudNew to_point_cloud() const;

    public:
        // Holds the magnitude of the screen space gradient
        Tensor densification_info = Tensor();

    private:
        int active_sh_degree = 0;
        int max_sh_degree = 0;
        float scene_scale = 0.0f;

        // Core gaussian attributes
        Tensor means;      // [N, 3] - positions
        Tensor sh0;        // [N, 1, 3] - DC SH coefficients (coeffs, channels) - ACTUAL REF FORMAT
        Tensor shN;        // [N, (degree+1)^2-1, 3] - higher-order SH (coeffs, channels) - ACTUAL REF FORMAT
        Tensor scaling;    // [N, 3] - log-space scales
        Tensor rotation;   // [N, 4] - quaternions (unnormalized)
        Tensor opacity;    // [N, 1] - logit-space opacity

        // Async save management
        mutable std::mutex save_mutex;
        mutable std::vector<std::future<void>> save_futures;

        // Helper methods for async save management
        void wait_for_saves() const;
        void cleanup_finished_saves() const;
    };

} // namespace gs