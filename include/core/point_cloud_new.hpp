/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <string>
#include <vector>

namespace gs {

    /**
     * @brief Point cloud structure using Tensor library (no libtorch)
     *
     * This is a modern replacement for PointCloud that uses our custom Tensor
     * implementation instead of raw CUDA memory and torch tensors.
     */
    class PointCloudNew {
    public:
        // Basic point cloud data
        Tensor means;  // [N, 3] float32 - positions
        Tensor colors; // [N, 3] float32 - normalized to [0,1]

        // For Gaussian point clouds (optional)
        Tensor normals;  // [N, 3] float32
        Tensor sh0;      // [N, 3, 1] float32 (channels, coeffs) - ACTUAL REF FORMAT
        Tensor shN;      // [N, 3, coeffs] float32 (channels, coeffs) - ACTUAL REF FORMAT
        Tensor opacity;  // [N, 1] float32
        Tensor scaling;  // [N, 3] float32
        Tensor rotation; // [N, 4] float32 - quaternion

        // Metadata
        std::vector<std::string> attribute_names;

        // Default constructor
        PointCloudNew() = default;

        /**
         * @brief Constructor for basic point cloud (positions + colors only)
         * @param host_positions Host array of positions [N*3]
         * @param host_colors Host array of colors [N*3] normalized to [0,1]
         * @param n_points Number of points
         */
        PointCloudNew(const float* host_positions, const float* host_colors, size_t n_points) {
            if (n_points == 0)
                return;

            // Create tensors from host data - allocate on CUDA
            std::vector<float> pos_vec(host_positions, host_positions + n_points * 3);
            std::vector<float> col_vec(host_colors, host_colors + n_points * 3);

            std::vector<size_t> shape = {n_points, 3};
            means = Tensor::from_vector(pos_vec, TensorShape(shape), Device::CUDA);
            colors = Tensor::from_vector(col_vec, TensorShape(shape), Device::CUDA);
        }

        /**
         * @brief Constructor that takes ownership of existing tensors
         */
        PointCloudNew(Tensor positions, Tensor point_colors)
            : means(std::move(positions)),
              colors(std::move(point_colors)) {

            if (means.is_valid() && colors.is_valid()) {
                if (means.shape() != colors.shape()) {
                    LOG_ERROR("PointCloudNew: positions and colors must have same shape");
                }
                if (means.ndim() != 2 || means.size(1) != 3) {
                    LOG_ERROR("PointCloudNew: positions must be [N, 3]");
                }
            }
        }

        // Rule of 5: use default implementations since Tensor handles memory
        ~PointCloudNew() = default;
        PointCloudNew(const PointCloudNew&) = delete;
        PointCloudNew& operator=(const PointCloudNew&) = delete;
        PointCloudNew(PointCloudNew&&) noexcept = default;
        PointCloudNew& operator=(PointCloudNew&&) noexcept = default;

        /**
         * @brief Check if this is a Gaussian point cloud
         */
        bool is_gaussian() const {
            return sh0.is_valid() && opacity.is_valid();
        }

        /**
         * @brief Get number of points
         */
        size_t size() const {
            return means.is_valid() ? means.size(0) : 0;
        }

        /**
         * @brief Allocate memory for basic point cloud
         * @param n_points Number of points to allocate
         * @param device Device to allocate on (default: CUDA)
         */
        void allocate_basic(size_t n_points, Device device = Device::CUDA) {
            if (n_points == 0) {
                means = Tensor();
                colors = Tensor();
                return;
            }

            means = Tensor::zeros({n_points, 3}, device, DataType::Float32);
            colors = Tensor::zeros({n_points, 3}, device, DataType::Float32);

            // Clear Gaussian attributes
            normals = Tensor();
            sh0 = Tensor();
            shN = Tensor();
            opacity = Tensor();
            scaling = Tensor();
            rotation = Tensor();
        }

        /**
         * @brief Allocate memory for Gaussian point cloud
         * @param n_points Number of points
         * @param sh0_channels Number of color channels in sh0 (usually 3)
         * @param sh0_coeffs Number of coefficients in sh0 (usually 1)
         * @param shN_channels Number of color channels in shN (usually 3)
         * @param shN_coeffs Number of coefficients in shN (depends on SH degree)
         * @param device Device to allocate on (default: CUDA)
         *
         * NOTE: Parameter order matches old PointCloud: (n, channels, coeffs, channels, coeffs)
         */
        void allocate_gaussian(size_t n_points,
                               size_t sh0_channels, size_t sh0_coeffs,
                               size_t shN_channels, size_t shN_coeffs,
                               Device device = Device::CUDA) {

            // Allocate basic attributes
            allocate_basic(n_points, device);

            if (n_points == 0)
                return;

            // Allocate Gaussian attributes
            // Internal format: [N, channels, coeffs] (matches old PointCloud reference format)
            normals = Tensor::zeros({n_points, 3}, device, DataType::Float32);
            sh0 = Tensor::zeros({n_points, sh0_channels, sh0_coeffs}, device, DataType::Float32);
            shN = Tensor::zeros({n_points, shN_channels, shN_coeffs}, device, DataType::Float32);
            opacity = Tensor::zeros({n_points, 1}, device, DataType::Float32);
            scaling = Tensor::zeros({n_points, 3}, device, DataType::Float32);
            rotation = Tensor::zeros({n_points, 4}, device, DataType::Float32);
        }

        /**
         * @brief Move point cloud to different device
         */
        PointCloudNew to_device(Device device) const {
            PointCloudNew pc;

            // Move basic attributes
            if (means.is_valid())
                pc.means = means.to(device);
            if (colors.is_valid())
                pc.colors = colors.to(device);

            // Move Gaussian attributes if present
            if (normals.is_valid())
                pc.normals = normals.to(device);
            if (sh0.is_valid())
                pc.sh0 = sh0.to(device);
            if (shN.is_valid())
                pc.shN = shN.to(device);
            if (opacity.is_valid())
                pc.opacity = opacity.to(device);
            if (scaling.is_valid())
                pc.scaling = scaling.to(device);
            if (rotation.is_valid())
                pc.rotation = rotation.to(device);

            // Copy metadata
            pc.attribute_names = attribute_names;

            return pc;
        }

        /**
         * @brief Copy to CPU
         */
        PointCloudNew cpu() const {
            return to_device(Device::CPU);
        }

        /**
         * @brief Copy to CUDA
         */
        PointCloudNew cuda() const {
            return to_device(Device::CUDA);
        }

        /**
         * @brief Copy positions and colors to host arrays
         */
        void copy_to_host(float* host_means, float* host_colors) const {
            if (!means.is_valid() || !colors.is_valid()) {
                LOG_ERROR("PointCloudNew::copy_to_host: invalid tensors");
                return;
            }

            auto means_cpu = means.cpu();
            auto colors_cpu = colors.cpu();

            if (host_means) {
                std::memcpy(host_means, means_cpu.ptr<float>(),
                            means_cpu.numel() * sizeof(float));
            }

            if (host_colors) {
                std::memcpy(host_colors, colors_cpu.ptr<float>(),
                            colors_cpu.numel() * sizeof(float));
            }
        }

        /**
         * @brief Get positions as host vector
         */
        std::vector<float> get_positions() const {
            if (!means.is_valid())
                return {};
            return means.cpu().to_vector();
        }

        /**
         * @brief Get colors as host vector
         */
        std::vector<float> get_colors() const {
            if (!colors.is_valid())
                return {};
            return colors.cpu().to_vector();
        }

        /**
         * @brief Validate point cloud structure
         */
        bool is_valid() const {
            if (!means.is_valid() || !colors.is_valid()) {
                return false;
            }

            if (means.shape() != colors.shape()) {
                LOG_ERROR("PointCloudNew: shape mismatch between means and colors");
                return false;
            }

            if (means.ndim() != 2 || means.size(1) != 3) {
                LOG_ERROR("PointCloudNew: means must be [N, 3]");
                return false;
            }

            // Validate Gaussian attributes if present
            if (is_gaussian()) {
                size_t n = means.size(0);

                if (normals.is_valid() &&
                    (normals.ndim() != 2 || normals.size(0) != n || normals.size(1) != 3)) {
                    LOG_ERROR("PointCloudNew: normals must be [N, 3]");
                    return false;
                }

                if (sh0.is_valid() && (sh0.ndim() != 3 || sh0.size(0) != n || sh0.size(1) != 3 || sh0.size(2) != 1)) {
                    LOG_ERROR("PointCloudNew: sh0 must be [N, 3, 1] but got [{}]", sh0.shape().str());
                    return false;
                }

                if (shN.is_valid() && (shN.ndim() != 3 || shN.size(0) != n || shN.size(1) != 3)) {
                    LOG_ERROR("PointCloudNew: shN must be [N, 3, coeffs]");
                    return false;
                }

                if (opacity.is_valid() &&
                    (opacity.ndim() != 2 || opacity.size(0) != n)) {
                    LOG_ERROR("PointCloudNew: opacity must be [N, 1] or [N]");
                    return false;
                }

                if (scaling.is_valid() &&
                    (scaling.ndim() != 2 || scaling.size(0) != n || scaling.size(1) != 3)) {
                    LOG_ERROR("PointCloudNew: scaling must be [N, 3]");
                    return false;
                }

                if (rotation.is_valid() &&
                    (rotation.ndim() != 2 || rotation.size(0) != n || rotation.size(1) != 4)) {
                    LOG_ERROR("PointCloudNew: rotation must be [N, 4]");
                    return false;
                }
            }

            return true;
        }

        /**
         * @brief Clone the point cloud (deep copy)
         */
        PointCloudNew clone() const {
            PointCloudNew pc;

            if (means.is_valid())
                pc.means = means.clone();
            if (colors.is_valid())
                pc.colors = colors.clone();
            if (normals.is_valid())
                pc.normals = normals.clone();
            if (sh0.is_valid())
                pc.sh0 = sh0.clone();
            if (shN.is_valid())
                pc.shN = shN.clone();
            if (opacity.is_valid())
                pc.opacity = opacity.clone();
            if (scaling.is_valid())
                pc.scaling = scaling.clone();
            if (rotation.is_valid())
                pc.rotation = rotation.clone();

            pc.attribute_names = attribute_names;

            return pc;
        }

        /**
         * @brief Get diagnostic string
         */
        std::string str() const {
            std::ostringstream oss;
            oss << "PointCloudNew(";
            oss << "points=" << size();

            if (is_gaussian()) {
                oss << ", gaussian=true";
                if (sh0.is_valid()) {
                    oss << ", sh0=" << sh0.shape().str();
                }
                if (shN.is_valid()) {
                    oss << ", shN=" << shN.shape().str();
                }
            } else {
                oss << ", gaussian=false";
            }

            oss << ")";
            return oss.str();
        }
    };

} // namespace gs