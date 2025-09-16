/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace gs {
    // Torch-free point cloud structure using raw CUDA memory
    struct PointCloud {
        // Basic point cloud data
        float* means_cuda = nullptr;  // [N, 3] float32
        float* colors_cuda = nullptr; // [N, 3] float32 (normalized to [0,1])

        // For Gaussian point clouds (optional, can be nullptr for basic point clouds)
        float* normals_cuda = nullptr;  // [N, 3] float32
        float* sh0_cuda = nullptr;      // [N, channels*coeffs] float32 (flattened)
        float* shN_cuda = nullptr;      // [N, channels*coeffs] float32 (flattened)
        float* opacity_cuda = nullptr;  // [N, 1] float32
        float* scaling_cuda = nullptr;  // [N, 3] float32
        float* rotation_cuda = nullptr; // [N, 4] float32

        // Dimensions
        size_t num_points = 0;
        size_t sh0_dims[3] = {0, 0, 0}; // [N, channels, coeffs] - for reconstruction
        size_t shN_dims[3] = {0, 0, 0}; // [N, channels, coeffs] - for reconstruction

        // Metadata
        std::vector<std::string> attribute_names;

        // Default constructor
        PointCloud() = default;

        // Constructor for basic point cloud (positions + colors only)
        PointCloud(const float* host_positions, const float* host_colors, size_t n_points)
            : num_points(n_points) {
            if (n_points > 0) {
                // Allocate and copy positions
                size_t pos_bytes = n_points * 3 * sizeof(float);
                cudaMalloc(&means_cuda, pos_bytes);
                cudaMemcpy(means_cuda, host_positions, pos_bytes, cudaMemcpyHostToDevice);

                // Allocate and copy colors
                size_t col_bytes = n_points * 3 * sizeof(float);
                cudaMalloc(&colors_cuda, col_bytes);
                cudaMemcpy(colors_cuda, host_colors, col_bytes, cudaMemcpyHostToDevice);
            }
        }

        // Constructor that takes ownership of already allocated CUDA memory
        PointCloud(float* means, float* colors, size_t n_points)
            : means_cuda(means),
              colors_cuda(colors),
              num_points(n_points) {}

        // Destructor
        ~PointCloud() {
            free_memory();
        }

        // Delete copy operations
        PointCloud(const PointCloud&) = delete;
        PointCloud& operator=(const PointCloud&) = delete;

        // Move constructor
        PointCloud(PointCloud&& other) noexcept
            : means_cuda(other.means_cuda),
              colors_cuda(other.colors_cuda),
              normals_cuda(other.normals_cuda),
              sh0_cuda(other.sh0_cuda),
              shN_cuda(other.shN_cuda),
              opacity_cuda(other.opacity_cuda),
              scaling_cuda(other.scaling_cuda),
              rotation_cuda(other.rotation_cuda),
              num_points(other.num_points),
              attribute_names(std::move(other.attribute_names)) {

            std::memcpy(sh0_dims, other.sh0_dims, sizeof(sh0_dims));
            std::memcpy(shN_dims, other.shN_dims, sizeof(shN_dims));

            // Clear source
            other.means_cuda = nullptr;
            other.colors_cuda = nullptr;
            other.normals_cuda = nullptr;
            other.sh0_cuda = nullptr;
            other.shN_cuda = nullptr;
            other.opacity_cuda = nullptr;
            other.scaling_cuda = nullptr;
            other.rotation_cuda = nullptr;
            other.num_points = 0;
        }

        // Move assignment
        PointCloud& operator=(PointCloud&& other) noexcept {
            if (this != &other) {
                free_memory();

                means_cuda = other.means_cuda;
                colors_cuda = other.colors_cuda;
                normals_cuda = other.normals_cuda;
                sh0_cuda = other.sh0_cuda;
                shN_cuda = other.shN_cuda;
                opacity_cuda = other.opacity_cuda;
                scaling_cuda = other.scaling_cuda;
                rotation_cuda = other.rotation_cuda;
                num_points = other.num_points;
                attribute_names = std::move(other.attribute_names);

                std::memcpy(sh0_dims, other.sh0_dims, sizeof(sh0_dims));
                std::memcpy(shN_dims, other.shN_dims, sizeof(shN_dims));

                // Clear source
                other.means_cuda = nullptr;
                other.colors_cuda = nullptr;
                other.normals_cuda = nullptr;
                other.sh0_cuda = nullptr;
                other.shN_cuda = nullptr;
                other.opacity_cuda = nullptr;
                other.scaling_cuda = nullptr;
                other.rotation_cuda = nullptr;
                other.num_points = 0;
            }
            return *this;
        }

        // Check if this is a Gaussian point cloud (has additional attributes)
        bool is_gaussian() const {
            return sh0_cuda != nullptr && opacity_cuda != nullptr;
        }

        // Get number of points
        size_t size() const {
            return num_points;
        }

        // Allocate memory for basic point cloud
        void allocate_basic(size_t n_points) {
            free_memory();

            num_points = n_points;
            if (n_points > 0) {
                cudaMalloc(&means_cuda, n_points * 3 * sizeof(float));
                cudaMalloc(&colors_cuda, n_points * 3 * sizeof(float));

                // Initialize to zero
                cudaMemset(means_cuda, 0, n_points * 3 * sizeof(float));
                cudaMemset(colors_cuda, 0, n_points * 3 * sizeof(float));
            }
        }

        // Allocate memory for Gaussian point cloud
        void allocate_gaussian(size_t n_points, size_t sh0_c, size_t sh0_d, size_t shN_c, size_t shN_d) {
            allocate_basic(n_points);

            if (n_points > 0) {
                // Store dimensions
                sh0_dims[0] = n_points;
                sh0_dims[1] = sh0_c; // channels
                sh0_dims[2] = sh0_d; // coefficients

                shN_dims[0] = n_points;
                shN_dims[1] = shN_c; // channels
                shN_dims[2] = shN_d; // coefficients

                // Allocate additional Gaussian attributes
                cudaMalloc(&normals_cuda, n_points * 3 * sizeof(float));
                cudaMalloc(&sh0_cuda, n_points * sh0_c * sh0_d * sizeof(float));
                cudaMalloc(&shN_cuda, n_points * shN_c * shN_d * sizeof(float));
                cudaMalloc(&opacity_cuda, n_points * sizeof(float));
                cudaMalloc(&scaling_cuda, n_points * 3 * sizeof(float));
                cudaMalloc(&rotation_cuda, n_points * 4 * sizeof(float));

                // Initialize to zero
                cudaMemset(normals_cuda, 0, n_points * 3 * sizeof(float));
                cudaMemset(sh0_cuda, 0, n_points * sh0_c * sh0_d * sizeof(float));
                cudaMemset(shN_cuda, 0, n_points * shN_c * shN_d * sizeof(float));
                cudaMemset(opacity_cuda, 0, n_points * sizeof(float));
                cudaMemset(scaling_cuda, 0, n_points * 3 * sizeof(float));
                cudaMemset(rotation_cuda, 0, n_points * 4 * sizeof(float));
            }
        }

        // Copy to device (if data is on different device)
        PointCloud to_device(int device_id) const {
            int current_device;
            cudaGetDevice(&current_device);

            if (current_device == device_id) {
                // Already on correct device, return copy of pointers
                PointCloud pc;
                pc.means_cuda = means_cuda;
                pc.colors_cuda = colors_cuda;
                pc.normals_cuda = normals_cuda;
                pc.sh0_cuda = sh0_cuda;
                pc.shN_cuda = shN_cuda;
                pc.opacity_cuda = opacity_cuda;
                pc.scaling_cuda = scaling_cuda;
                pc.rotation_cuda = rotation_cuda;
                pc.num_points = num_points;
                pc.attribute_names = attribute_names;
                std::memcpy(pc.sh0_dims, sh0_dims, sizeof(sh0_dims));
                std::memcpy(pc.shN_dims, shN_dims, sizeof(shN_dims));
                return pc;
            }

            // Need to copy to different device
            cudaSetDevice(device_id);

            PointCloud pc;
            pc.num_points = num_points;
            pc.attribute_names = attribute_names;
            std::memcpy(pc.sh0_dims, sh0_dims, sizeof(sh0_dims));
            std::memcpy(pc.shN_dims, shN_dims, sizeof(shN_dims));

            if (num_points > 0) {
                // Copy basic data
                if (means_cuda) {
                    cudaMalloc(&pc.means_cuda, num_points * 3 * sizeof(float));
                    cudaMemcpy(pc.means_cuda, means_cuda, num_points * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                if (colors_cuda) {
                    cudaMalloc(&pc.colors_cuda, num_points * 3 * sizeof(float));
                    cudaMemcpy(pc.colors_cuda, colors_cuda, num_points * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
                }

                // Copy Gaussian data if present
                if (normals_cuda) {
                    cudaMalloc(&pc.normals_cuda, num_points * 3 * sizeof(float));
                    cudaMemcpy(pc.normals_cuda, normals_cuda, num_points * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                if (sh0_cuda) {
                    size_t sh0_size = sh0_dims[0] * sh0_dims[1] * sh0_dims[2] * sizeof(float);
                    cudaMalloc(&pc.sh0_cuda, sh0_size);
                    cudaMemcpy(pc.sh0_cuda, sh0_cuda, sh0_size, cudaMemcpyDeviceToDevice);
                }
                if (shN_cuda) {
                    size_t shN_size = shN_dims[0] * shN_dims[1] * shN_dims[2] * sizeof(float);
                    cudaMalloc(&pc.shN_cuda, shN_size);
                    cudaMemcpy(pc.shN_cuda, shN_cuda, shN_size, cudaMemcpyDeviceToDevice);
                }
                if (opacity_cuda) {
                    cudaMalloc(&pc.opacity_cuda, num_points * sizeof(float));
                    cudaMemcpy(pc.opacity_cuda, opacity_cuda, num_points * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                if (scaling_cuda) {
                    cudaMalloc(&pc.scaling_cuda, num_points * 3 * sizeof(float));
                    cudaMemcpy(pc.scaling_cuda, scaling_cuda, num_points * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                if (rotation_cuda) {
                    cudaMalloc(&pc.rotation_cuda, num_points * 4 * sizeof(float));
                    cudaMemcpy(pc.rotation_cuda, rotation_cuda, num_points * 4 * sizeof(float), cudaMemcpyDeviceToDevice);
                }
            }

            cudaSetDevice(current_device);
            return pc;
        }

        // Copy to host memory (for export)
        void copy_to_host(float* host_means, float* host_colors) const {
            if (means_cuda && host_means) {
                cudaMemcpy(host_means, means_cuda, num_points * 3 * sizeof(float), cudaMemcpyDeviceToHost);
            }
            if (colors_cuda && host_colors) {
                cudaMemcpy(host_colors, colors_cuda, num_points * 3 * sizeof(float), cudaMemcpyDeviceToHost);
            }
        }

    private:
        void free_memory() {
            if (means_cuda) {
                cudaFree(means_cuda);
                means_cuda = nullptr;
            }
            if (colors_cuda) {
                cudaFree(colors_cuda);
                colors_cuda = nullptr;
            }
            if (normals_cuda) {
                cudaFree(normals_cuda);
                normals_cuda = nullptr;
            }
            if (sh0_cuda) {
                cudaFree(sh0_cuda);
                sh0_cuda = nullptr;
            }
            if (shN_cuda) {
                cudaFree(shN_cuda);
                shN_cuda = nullptr;
            }
            if (opacity_cuda) {
                cudaFree(opacity_cuda);
                opacity_cuda = nullptr;
            }
            if (scaling_cuda) {
                cudaFree(scaling_cuda);
                scaling_cuda = nullptr;
            }
            if (rotation_cuda) {
                cudaFree(rotation_cuda);
                rotation_cuda = nullptr;
            }
            num_points = 0;
        }
    };
} // namespace gs