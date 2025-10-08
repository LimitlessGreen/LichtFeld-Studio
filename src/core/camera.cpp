/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <cuda_runtime.h>
#include <glm/gtc/matrix_inverse.hpp>

namespace gs {

    // Helper function to create world-to-view matrix
    static void create_world_to_view_matrix(const float* R_data, const float* T_data, float* out_w2c) {
        // Initialize as identity matrix
        std::memset(out_w2c, 0, 16 * sizeof(float));
        out_w2c[0] = out_w2c[5] = out_w2c[10] = out_w2c[15] = 1.0f;

        // Copy rotation matrix (3x3) into upper-left of 4x4
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                out_w2c[i * 4 + j] = R_data[i * 3 + j];
            }
        }

        // Copy translation vector into last column
        out_w2c[3] = T_data[0];
        out_w2c[7] = T_data[1];
        out_w2c[11] = T_data[2];
    }

    // Helper to compute matrix inverse
    static void invert_4x4_matrix(const float* in_matrix, float* out_matrix) {
        // Convert to GLM for inversion
        glm::mat4 mat;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                mat[j][i] = in_matrix[i * 4 + j]; // GLM is column-major
            }
        }

        glm::mat4 inv = glm::inverse(mat);

        // Convert back to row-major
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                out_matrix[i * 4 + j] = inv[j][i];
            }
        }
    }

    void Camera::compute_world_view_transform() {
        create_world_to_view_matrix(_R_cpu.data(), _T_cpu.data(), _world_view_transform_cpu.data());
    }

    void Camera::compute_camera_position() {
        // Compute inverse (c2w)
        float c2w[16];
        invert_4x4_matrix(_world_view_transform_cpu.data(), c2w);

        // Extract camera position (last column of c2w)
        _cam_position_cpu[0] = c2w[3];
        _cam_position_cpu[1] = c2w[7];
        _cam_position_cpu[2] = c2w[11];
    }

    void Camera::ensure_cuda_allocated() const {
        if (!_cuda_allocated) {
            // Allocate GPU memory
            cudaMalloc(&_world_view_transform_cuda, 16 * sizeof(float));
            cudaMalloc(&_cam_position_cuda, 3 * sizeof(float));

            // Copy data to GPU
            cudaMemcpy(_world_view_transform_cuda, _world_view_transform_cpu.data(),
                       16 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(_cam_position_cuda, _cam_position_cpu.data(),
                       3 * sizeof(float), cudaMemcpyHostToDevice);

            _cuda_allocated = true;
        }
    }

    void Camera::free_cuda_memory() {
        if (_cuda_allocated) {
            if (_world_view_transform_cuda) {
                cudaFree(_world_view_transform_cuda);
                _world_view_transform_cuda = nullptr;
            }
            if (_cam_position_cuda) {
                cudaFree(_cam_position_cuda);
                _cam_position_cuda = nullptr;
            }
            _cuda_allocated = false;
        }
    }

    Camera::Camera(const torch::Tensor& R,
                   const torch::Tensor& T,
                   float focal_x, float focal_y,
                   float center_x, float center_y,
                   const torch::Tensor radial_distortion,
                   const torch::Tensor tangential_distortion,
                   gsplat::CameraModelType camera_model_type,
                   const std::string& image_name,
                   const std::filesystem::path& image_path,
                   int camera_width, int camera_height,
                   int uid)
        : _uid(uid),
          _focal_x(focal_x),
          _focal_y(focal_y),
          _center_x(center_x),
          _center_y(center_y),
          _camera_model_type(camera_model_type),
          _image_name(image_name),
          _image_path(image_path),
          _camera_width(camera_width),
          _camera_height(camera_height),
          _image_width(camera_width),
          _image_height(camera_height) {

        // Copy R and T to CPU arrays
        auto R_cpu = R.to(torch::kCPU).contiguous();
        auto T_cpu = T.to(torch::kCPU).contiguous();

        const float* R_data = R_cpu.data_ptr<float>();
        const float* T_data = T_cpu.data_ptr<float>();

        std::memcpy(_R_cpu.data(), R_data, 9 * sizeof(float));
        std::memcpy(_T_cpu.data(), T_data, 3 * sizeof(float));

        // Copy distortion parameters if present
        if (radial_distortion.numel() > 0) {
            auto rad_cpu = radial_distortion.to(torch::kCPU).contiguous();
            _radial_distortion_cpu.resize(rad_cpu.numel());
            std::memcpy(_radial_distortion_cpu.data(), rad_cpu.data_ptr<float>(),
                        rad_cpu.numel() * sizeof(float));
        }

        if (tangential_distortion.numel() > 0) {
            auto tan_cpu = tangential_distortion.to(torch::kCPU).contiguous();
            _tangential_distortion_cpu.resize(tan_cpu.numel());
            std::memcpy(_tangential_distortion_cpu.data(), tan_cpu.data_ptr<float>(),
                        tan_cpu.numel() * sizeof(float));
        }

        // Compute world view transform
        compute_world_view_transform();

        // Compute camera position
        compute_camera_position();

        // Compute field of view
        _FoVx = focal2fov(_focal_x, _camera_width);
        _FoVy = focal2fov(_focal_y, _camera_height);
    }

    Camera::Camera(const Camera& other, const torch::Tensor& transform)
        : _uid(other._uid),
          _focal_x(other._focal_x),
          _focal_y(other._focal_y),
          _center_x(other._center_x),
          _center_y(other._center_y),
          _R_cpu(other._R_cpu),
          _T_cpu(other._T_cpu),
          _radial_distortion_cpu(other._radial_distortion_cpu),
          _tangential_distortion_cpu(other._tangential_distortion_cpu),
          _camera_model_type(other._camera_model_type),
          _image_name(other._image_name),
          _image_path(other._image_path),
          _camera_width(other._camera_width),
          _camera_height(other._camera_height),
          _image_width(other._image_width),
          _image_height(other._image_height),
          _cam_position_cpu(other._cam_position_cpu),
          _FoVx(other._FoVx),
          _FoVy(other._FoVy) {

        // Copy transform matrix to CPU array
        auto transform_cpu = transform.squeeze(0).to(torch::kCPU).contiguous();
        const float* transform_data = transform_cpu.data_ptr<float>();
        std::memcpy(_world_view_transform_cpu.data(), transform_data, 16 * sizeof(float));
    }

    // Move constructor
    Camera::Camera(Camera&& other) noexcept
        : _FoVx(other._FoVx),
          _FoVy(other._FoVy),
          _uid(other._uid),
          _focal_x(other._focal_x),
          _focal_y(other._focal_y),
          _center_x(other._center_x),
          _center_y(other._center_y),
          _R_cpu(other._R_cpu),
          _T_cpu(other._T_cpu),
          _radial_distortion_cpu(std::move(other._radial_distortion_cpu)),
          _tangential_distortion_cpu(std::move(other._tangential_distortion_cpu)),
          _camera_model_type(other._camera_model_type),
          _image_name(std::move(other._image_name)),
          _image_path(std::move(other._image_path)),
          _camera_width(other._camera_width),
          _camera_height(other._camera_height),
          _image_width(other._image_width),
          _image_height(other._image_height),
          _world_view_transform_cpu(other._world_view_transform_cpu),
          _cam_position_cpu(other._cam_position_cpu),
          _world_view_transform_cuda(other._world_view_transform_cuda),
          _cam_position_cuda(other._cam_position_cuda),
          _cuda_allocated(other._cuda_allocated),
          _stream(other._stream) {

        // Clear the source
        other._world_view_transform_cuda = nullptr;
        other._cam_position_cuda = nullptr;
        other._cuda_allocated = false;
    }

    // Move assignment
    Camera& Camera::operator=(Camera&& other) noexcept {
        if (this != &other) {
            // Free existing CUDA memory
            free_cuda_memory();

            // Move all members
            _FoVx = other._FoVx;
            _FoVy = other._FoVy;
            _uid = other._uid;
            _focal_x = other._focal_x;
            _focal_y = other._focal_y;
            _center_x = other._center_x;
            _center_y = other._center_y;
            _R_cpu = other._R_cpu;
            _T_cpu = other._T_cpu;
            _radial_distortion_cpu = std::move(other._radial_distortion_cpu);
            _tangential_distortion_cpu = std::move(other._tangential_distortion_cpu);
            _camera_model_type = other._camera_model_type;
            _image_name = std::move(other._image_name);
            _image_path = std::move(other._image_path);
            _camera_width = other._camera_width;
            _camera_height = other._camera_height;
            _image_width = other._image_width;
            _image_height = other._image_height;
            _world_view_transform_cpu = other._world_view_transform_cpu;
            _cam_position_cpu = other._cam_position_cpu;
            _world_view_transform_cuda = other._world_view_transform_cuda;
            _cam_position_cuda = other._cam_position_cuda;
            _cuda_allocated = other._cuda_allocated;
            _stream = other._stream;

            // Clear the source
            other._world_view_transform_cuda = nullptr;
            other._cam_position_cuda = nullptr;
            other._cuda_allocated = false;
        }
        return *this;
    }

    Camera::~Camera() {
        free_cuda_memory();
    }

    // ========== TORCH INTERFACE IMPLEMENTATION ==========

    torch::Tensor Camera::world_view_transform() const {
        ensure_cuda_allocated();
        return torch::from_blob(
            const_cast<float*>(_world_view_transform_cuda),
            {1, 4, 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }

    torch::Tensor Camera::cam_position() const {
        ensure_cuda_allocated();
        return torch::from_blob(
            const_cast<float*>(_cam_position_cuda),
            {3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }

    torch::Tensor Camera::R() const {
        return torch::from_blob(
                   const_cast<float*>(_R_cpu.data()),
                   {3, 3},
                   torch::TensorOptions().dtype(torch::kFloat32))
            .clone()
            .to(torch::kCUDA);
    }

    torch::Tensor Camera::T() const {
        return torch::from_blob(
                   const_cast<float*>(_T_cpu.data()),
                   {3},
                   torch::TensorOptions().dtype(torch::kFloat32))
            .clone()
            .to(torch::kCUDA);
    }

    torch::Tensor Camera::K() const {
        // Create K matrix on CPU first
        float K_data[9] = {0};
        auto [fx, fy, cx, cy] = get_intrinsics();

        K_data[0] = fx;   // K[0][0]
        K_data[4] = fy;   // K[1][1]
        K_data[2] = cx;   // K[0][2]
        K_data[5] = cy;   // K[1][2]
        K_data[8] = 1.0f; // K[2][2]

        // Create tensor and move to CUDA
        return torch::from_blob(
                   K_data,
                   {1, 3, 3},
                   torch::TensorOptions().dtype(torch::kFloat32))
            .clone()
            .to(torch::kCUDA);
    }

    torch::Tensor Camera::radial_distortion() const noexcept {
        if (_radial_distortion_cpu.empty()) {
            return torch::empty({0}, torch::kFloat32);
        }
        return torch::from_blob(
                   const_cast<float*>(_radial_distortion_cpu.data()),
                   {static_cast<long>(_radial_distortion_cpu.size())},
                   torch::TensorOptions().dtype(torch::kFloat32))
            .clone();
    }

    torch::Tensor Camera::tangential_distortion() const noexcept {
        if (_tangential_distortion_cpu.empty()) {
            return torch::empty({0}, torch::kFloat32);
        }
        return torch::from_blob(
                   const_cast<float*>(_tangential_distortion_cpu.data()),
                   {static_cast<long>(_tangential_distortion_cpu.size())},
                   torch::TensorOptions().dtype(torch::kFloat32))
            .clone();
    }

    std::tuple<float, float, float, float> Camera::get_intrinsics() const {
        float x_scale_factor = float(_image_width) / float(_camera_width);
        float y_scale_factor = float(_image_height) / float(_camera_height);
        float fx = _focal_x * x_scale_factor;
        float fy = _focal_y * y_scale_factor;
        float cx = _center_x * x_scale_factor;
        float cy = _center_y * y_scale_factor;
        return std::make_tuple(fx, fy, cx, cy);
    }

    void Camera::load_image_size(int resize_factor) {
        // Load image synchronously
        auto result = load_image(_image_path, resize_factor);
        int w = std::get<1>(result);
        int h = std::get<2>(result);

        _image_width = w;
        _image_height = h;
    }

    size_t Camera::get_num_bytes_from_file() const {
        auto [w, h, c] = get_image_info(_image_path);
        size_t num_bytes = w * h * c * sizeof(float);
        return num_bytes;
    }
} // namespace gs