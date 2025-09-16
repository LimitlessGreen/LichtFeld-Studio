/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace gs::loader::internal {

    // Simple CUDA memory wrapper
    template <typename T>
    struct CudaBuffer {
        T* data = nullptr;
        size_t size = 0;

        CudaBuffer() = default;
        CudaBuffer(size_t n) : size(n) {
            if (n > 0) {
                cudaMalloc(&data, n * sizeof(T));
            }
        }

        ~CudaBuffer() {
            if (data)
                cudaFree(data);
        }

        // Delete copy, allow move
        CudaBuffer(const CudaBuffer&) = delete;
        CudaBuffer& operator=(const CudaBuffer&) = delete;
        CudaBuffer(CudaBuffer&& other) noexcept
            : data(other.data),
              size(other.size) {
            other.data = nullptr;
            other.size = 0;
        }
        CudaBuffer& operator=(CudaBuffer&& other) noexcept {
            if (this != &other) {
                if (data)
                    cudaFree(data);
                data = other.data;
                size = other.size;
                other.data = nullptr;
                other.size = 0;
            }
            return *this;
        }

        void upload(const T* host_data, size_t count) {
            if (count > size) {
                if (data)
                    cudaFree(data);
                cudaMalloc(&data, count * sizeof(T));
                size = count;
            }
            cudaMemcpy(data, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
        }

        void download(T* host_data, size_t count) const {
            size_t copy_size = (count < size) ? count : size;
            cudaMemcpy(host_data, data, copy_size * sizeof(T), cudaMemcpyDeviceToHost);
        }
    };

    // Internal point cloud representation
    struct CudaPointCloud {
        CudaBuffer<float> positions; // [N*3] flattened
        CudaBuffer<uint8_t> colors;  // [N*3] flattened
        size_t num_points = 0;

        CudaPointCloud() = default;
        CudaPointCloud(size_t n)
            : positions(n * 3),
              colors(n * 3),
              num_points(n) {}
    };

    // Internal splat representation
    struct CudaSplatData {
        size_t num_points = 0;
        int sh_degree = 0;
        float scene_scale = 1.0f;

        CudaBuffer<float> means;     // [N*3] flattened
        CudaBuffer<float> sh0;       // [N*1*3] flattened
        CudaBuffer<float> shN;       // [N*rest_coeffs*3] flattened
        CudaBuffer<float> scales;    // [N*3] flattened
        CudaBuffer<float> rotations; // [N*4] flattened
        CudaBuffer<float> opacity;   // [N*1] flattened

        // SH dimensions
        size_t sh0_dim1 = 0, sh0_dim2 = 0;
        size_t shN_dim1 = 0, shN_dim2 = 0;
    };

    // Camera data without torch
    struct CudaCameraData {
        uint32_t camera_id = 0;

        // Host data (doesn't need CUDA)
        float R[9]; // 3x3 rotation matrix, row-major
        float T[3]; // translation vector
        float focal_x = 0.f;
        float focal_y = 0.f;
        float center_x = 0.f;
        float center_y = 0.f;

        std::string image_name;
        std::filesystem::path image_path;
        int width = 0;
        int height = 0;

        // Distortion parameters (optional)
        std::vector<float> radial_distortion;
        std::vector<float> tangential_distortion;

        int camera_model_type = 0; // Store as int instead of enum

        // Raw params from COLMAP
        std::vector<float> params;

        // COLMAP camera model
        int camera_model = 0;
    };

    // Helper to compute scene center from camera positions
    inline void compute_scene_center(const std::vector<CudaCameraData>& cameras, float center[3]) {
        center[0] = center[1] = center[2] = 0.0f;

        if (cameras.empty())
            return;

        for (const auto& cam : cameras) {
            // Camera location in world space = -R^T * T
            float loc[3];
            // R^T * T
            loc[0] = cam.R[0] * cam.T[0] + cam.R[3] * cam.T[1] + cam.R[6] * cam.T[2];
            loc[1] = cam.R[1] * cam.T[0] + cam.R[4] * cam.T[1] + cam.R[7] * cam.T[2];
            loc[2] = cam.R[2] * cam.T[0] + cam.R[5] * cam.T[1] + cam.R[8] * cam.T[2];

            center[0] -= loc[0];
            center[1] -= loc[1];
            center[2] -= loc[2];
        }

        float inv_n = 1.0f / cameras.size();
        center[0] *= inv_n;
        center[1] *= inv_n;
        center[2] *= inv_n;
    }

} // namespace gs::loader::internal