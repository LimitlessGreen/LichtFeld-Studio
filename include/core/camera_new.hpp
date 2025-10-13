/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "Common.h"
#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <string>
#include <tuple>

namespace gs {

/**
 * @brief Camera class using custom Tensor library (no libtorch)
 *
 * This is a modern replacement for CameraRef that uses our custom Tensor
 * implementation instead of torch tensors.
 */
class CameraNew {
public:
    CameraNew() = default;

    /**
     * @brief Construct a camera from rotation, translation, and intrinsics
     * @param R Rotation matrix [3, 3]
     * @param T Translation vector [3]
     * @param focal_x Focal length in x (pixels)
     * @param focal_y Focal length in y (pixels)
     * @param center_x Principal point x (pixels)
     * @param center_y Principal point y (pixels)
     * @param radial_distortion Radial distortion coefficients [k1, k2, k3, k4]
     * @param tangential_distortion Tangential distortion coefficients [p1, p2]
     * @param camera_model_type Camera model type (PINHOLE, OPENCV, etc.)
     * @param image_name Name of the image
     * @param image_path Path to the image file
     * @param camera_width Original camera width
     * @param camera_height Original camera height
     * @param uid Unique identifier
     */
    CameraNew(const Tensor& R,
              const Tensor& T,
              float focal_x, float focal_y,
              float center_x, float center_y,
              const Tensor& radial_distortion,
              const Tensor& tangential_distortion,
              gsplat::CameraModelType camera_model_type,
              const std::string& image_name,
              const std::filesystem::path& image_path,
              int camera_width, int camera_height,
              int uid);

    /**
     * @brief Construct a camera from another camera with a transform applied
     * @param other Source camera
     * @param transform World-to-view transformation matrix [4, 4]
     */
    CameraNew(const CameraNew& other, const Tensor& transform);

    // Rule of 5: Delete copy, allow move
    CameraNew(const CameraNew&) = delete;
    CameraNew& operator=(const CameraNew&) = delete;
    CameraNew(CameraNew&&) noexcept = default;
    CameraNew& operator=(CameraNew&&) noexcept = default;
    ~CameraNew();

    // ========== IMAGE LOADING ==========

    /**
     * @brief Load image from disk and return it
     * @param resize_factor Downsampling factor (e.g., 2 = half size)
     * @param max_width Maximum width/height constraint
     * @return Image tensor [3, H, W] normalized to [0, 1]
     */
    Tensor load_and_get_image(int resize_factor = -1, int max_width = 3840);

    /**
     * @brief Load image dimensions without loading the full image
     * @param resize_factor Downsampling factor
     * @param max_width Maximum width/height constraint
     */
    void load_image_size(int resize_factor = -1, int max_width = 3840);

    /**
     * @brief Get the number of bytes the image will occupy in memory
     * @param resize_factor Downsampling factor
     * @param max_width Maximum width/height constraint
     * @return Size in bytes
     */
    size_t get_num_bytes_from_file(int resize_factor = -1, int max_width = 3840) const;
    size_t get_num_bytes_from_file() const;

    // ========== ACCESSORS ==========

    const Tensor& world_view_transform() const { return world_view_transform_; }
    const Tensor& cam_position() const { return cam_position_; }
    const Tensor& R() const { return R_; }
    const Tensor& T() const { return T_; }

    /**
     * @brief Get camera intrinsic matrix K [3, 3]
     * @return Intrinsic matrix tensor
     */
    Tensor K() const;

    /**
     * @brief Get scaled intrinsic parameters
     * @return Tuple of (fx, fy, cx, cy) scaled to current image size
     */
    std::tuple<float, float, float, float> get_intrinsics() const;

    // Simple getters
    int image_height() const noexcept { return image_height_; }
    int image_width() const noexcept { return image_width_; }
    int camera_height() const noexcept { return camera_height_; }
    int camera_width() const noexcept { return camera_width_; }
    float focal_x() const noexcept { return focal_x_; }
    float focal_y() const noexcept { return focal_y_; }
    float center_x() const noexcept { return center_x_; }
    float center_y() const noexcept { return center_y_; }
    const Tensor& radial_distortion() const noexcept { return radial_distortion_; }
    const Tensor& tangential_distortion() const noexcept { return tangential_distortion_; }
    gsplat::CameraModelType camera_model_type() const noexcept { return camera_model_type_; }
    const std::string& image_name() const noexcept { return image_name_; }
    const std::filesystem::path& image_path() const noexcept { return image_path_; }
    int uid() const noexcept { return uid_; }
    float FoVx() const noexcept { return FoVx_; }
    float FoVy() const noexcept { return FoVy_; }

    // ========== VALIDATION ==========

    /**
     * @brief Check if camera state is valid
     * @return True if all required tensors are valid
     */
    bool is_valid() const;

    /**
     * @brief Get diagnostic string
     */
    std::string str() const;

private:
    /**
     * @brief Compute world-to-view transformation matrix
     * @param R Rotation matrix [3, 3]
     * @param t Translation vector [3]
     * @return World-to-view matrix [1, 4, 4]
     */
    static Tensor world_to_view(const Tensor& R, const Tensor& t);

    // Camera parameters
    float FoVx_ = 0.0f;
    float FoVy_ = 0.0f;
    int uid_ = -1;
    float focal_x_ = 0.0f;
    float focal_y_ = 0.0f;
    float center_x_ = 0.0f;
    float center_y_ = 0.0f;

    // Extrinsics (redundancy with world_view_transform_ for convenience)
    Tensor R_;  // [3, 3] rotation matrix
    Tensor T_;  // [3] translation vector

    // Distortion parameters
    Tensor radial_distortion_;     // [4] or empty
    Tensor tangential_distortion_; // [2] or empty
    gsplat::CameraModelType camera_model_type_ = gsplat::CameraModelType::PINHOLE;

    // Image info
    std::string image_name_;
    std::filesystem::path image_path_;
    int camera_width_ = 0;   // Original camera dimensions
    int camera_height_ = 0;
    int image_width_ = 0;    // Actual loaded image dimensions
    int image_height_ = 0;

    // Computed tensors
    Tensor world_view_transform_;  // [1, 4, 4] world-to-view transform
    Tensor cam_position_;           // [3] camera position in world space

    // CUDA stream for async operations
    cudaStream_t stream_ = nullptr;
};

} // namespace gs