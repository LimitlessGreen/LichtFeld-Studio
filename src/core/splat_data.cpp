/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "core/sogs.hpp"
#include "kernels/training_kernels.cuh"

#include "external/nanoflann.hpp"
#include "external/tinyply.hpp"
#include <algorithm>
#include <cmath>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <print>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>
#include <cuda_runtime.h>
#include <cstring>

namespace {
    std::string tensor_sizes_to_string(const c10::ArrayRef<int64_t>& sizes) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << sizes[i];
        }
        oss << "]";
        return oss.str();
    }

    // Point cloud adaptor for nanoflann
    struct PointCloudAdaptor {
        const float* points;
        size_t num_points;

        PointCloudAdaptor(const float* pts, size_t n) : points(pts),
                                                        num_points(n) {}

        inline size_t kdtree_get_point_count() const { return num_points; }

        inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
            return points[idx * 3 + dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
    };

    // Fixed: KDTree typedef on single line
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, PointCloudAdaptor, 3>;

    // Compute mean distance to 3 nearest neighbors for each point
    torch::Tensor compute_mean_neighbor_distances(const torch::Tensor& points) {
        auto cpu_points = points.to(torch::kCPU).contiguous();
        const int num_points = cpu_points.size(0);

        TORCH_CHECK(cpu_points.dim() == 2 && cpu_points.size(1) == 3,
                    "Input points must have shape [N, 3]");
        TORCH_CHECK(cpu_points.dtype() == torch::kFloat32,
                    "Input points must be float32");

        if (num_points <= 1) {
            return torch::full({num_points}, 0.01f, points.options());
        }

        const float* data = cpu_points.data_ptr<float>();

        PointCloudAdaptor cloud(data, num_points);
        KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        index.buildIndex();

        auto result = torch::zeros({num_points}, torch::kFloat32);
        float* result_data = result.data_ptr<float>();

#pragma omp parallel for if (num_points > 1000)
        for (int i = 0; i < num_points; i++) {
            const float query_pt[3] = {data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]};

            const size_t num_results = std::min(4, num_points);
            std::vector<size_t> ret_indices(num_results);
            std::vector<float> out_dists_sqr(num_results);

            nanoflann::KNNResultSet<float> resultSet(num_results);
            resultSet.init(&ret_indices[0], &out_dists_sqr[0]);
            index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParameters(10));

            float sum_dist = 0.0f;
            int valid_neighbors = 0;

            for (size_t j = 0; j < num_results && valid_neighbors < 3; j++) {
                if (out_dists_sqr[j] > 1e-8f) {
                    sum_dist += std::sqrt(out_dists_sqr[j]);
                    valid_neighbors++;
                }
            }

            result_data[i] = (valid_neighbors > 0) ? (sum_dist / valid_neighbors) : 0.01f;
        }

        return result.to(points.device());
    }

    void write_ply_impl(const gs::PointCloud& pc,
                        const std::filesystem::path& root,
                        int iteration) {
        namespace fs = std::filesystem;
        fs::create_directories(root);

        // Create host memory for PLY export
        std::vector<float> host_means(pc.num_points * 3);
        std::vector<float> host_normals(pc.num_points * 3);
        std::vector<float> host_sh0(pc.sh0_dims[0] * pc.sh0_dims[1] * pc.sh0_dims[2]);
        std::vector<float> host_shN(pc.shN_dims[0] * pc.shN_dims[1] * pc.shN_dims[2]);
        std::vector<float> host_opacity(pc.num_points);
        std::vector<float> host_scaling(pc.num_points * 3);
        std::vector<float> host_rotation(pc.num_points * 4);

        // Copy from CUDA to host
        cudaMemcpy(host_means.data(), pc.means_cuda, host_means.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (pc.normals_cuda)
            cudaMemcpy(host_normals.data(), pc.normals_cuda, host_normals.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (pc.sh0_cuda)
            cudaMemcpy(host_sh0.data(), pc.sh0_cuda, host_sh0.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (pc.shN_cuda)
            cudaMemcpy(host_shN.data(), pc.shN_cuda, host_shN.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (pc.opacity_cuda)
            cudaMemcpy(host_opacity.data(), pc.opacity_cuda, host_opacity.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (pc.scaling_cuda)
            cudaMemcpy(host_scaling.data(), pc.scaling_cuda, host_scaling.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (pc.rotation_cuda)
            cudaMemcpy(host_rotation.data(), pc.rotation_cuda, host_rotation.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // Create vector of pointers for tinyply
        std::vector<float*> data_ptrs;
        data_ptrs.push_back(host_means.data());
        if (pc.normals_cuda) data_ptrs.push_back(host_normals.data());
        if (pc.sh0_cuda) data_ptrs.push_back(host_sh0.data());
        if (pc.shN_cuda) data_ptrs.push_back(host_shN.data());
        if (pc.opacity_cuda) data_ptrs.push_back(host_opacity.data());
        if (pc.scaling_cuda) data_ptrs.push_back(host_scaling.data());
        if (pc.rotation_cuda) data_ptrs.push_back(host_rotation.data());

        auto write_output_ply =
            [](const fs::path& file_path,
               const std::vector<float*>& data,
               const std::vector<std::string>& attr_names,
               size_t num_points) {
                tinyply::PlyFile ply;
                size_t attr_off = 0;

                for (size_t i = 0; i < data.size(); ++i) {
                    // Determine number of attributes for this data block
                    size_t num_attrs = 3; // Default for positions, normals, scaling
                    if (i == data.size() - 1 && attr_names.back().find("rot_") != std::string::npos) {
                        num_attrs = 4; // Rotation has 4 components
                    } else if (attr_names[attr_off] == "opacity") {
                        num_attrs = 1; // Opacity is scalar
                    } else if (attr_names[attr_off].find("f_dc_") != std::string::npos ||
                               attr_names[attr_off].find("f_rest_") != std::string::npos) {
                        // Count how many consecutive f_dc_ or f_rest_ attributes
                        num_attrs = 0;
                        size_t j = attr_off;
                        while (j < attr_names.size() &&
                               (attr_names[j].find("f_dc_") != std::string::npos ||
                                attr_names[j].find("f_rest_") != std::string::npos)) {
                            num_attrs++;
                            j++;
                        }
                    }

                    std::vector<std::string> attrs(attr_names.begin() + attr_off,
                                                   attr_names.begin() + attr_off + num_attrs);

                    ply.add_properties_to_element(
                        "vertex",
                        attrs,
                        tinyply::Type::FLOAT32,
                        num_points,
                        reinterpret_cast<uint8_t*>(data[i]),
                        tinyply::Type::INVALID, 0);

                    attr_off += num_attrs;
                }

                std::filebuf fb;
                fb.open(file_path, std::ios::out | std::ios::binary);
                std::ostream out_stream(&fb);
                ply.write(out_stream, /*binary=*/true);
            };

        write_output_ply(root / ("splat_" + std::to_string(iteration) + ".ply"),
                        data_ptrs, pc.attribute_names, pc.num_points);
    }

    void write_sog_impl(const gs::SplatData& splat_data,
                        const std::filesystem::path& root,
                        int iteration,
                        int kmeans_iterations) {
        namespace fs = std::filesystem;

        // Create SOG subdirectory
        fs::path sog_dir = root / "sog";
        fs::create_directories(sog_dir);

        // Set up SOG write options - use .sog extension to create bundle
        gs::core::SogWriteOptions options{
            .iterations = kmeans_iterations,
            .output_path = sog_dir / ("splat_" + std::to_string(iteration) + ".sog")};

        // Write SOG format
        auto result = gs::core::write_sog(splat_data, options);
        if (!result) {
            LOG_ERROR("Failed to write SOG format: {}", result.error());
        } else {
            LOG_DEBUG("Successfully wrote SOG format for iteration {}", iteration);
        }
    }
} // namespace

namespace gs {

    void SplatData::allocate_cuda_memory_and_tensors(int64_t num_points,
                                                 int64_t sh0_dim1, int64_t sh0_dim2,
                                                 int64_t shN_dim1, int64_t shN_dim2) {
    _num_points = num_points;
    _sh0_dims[0] = num_points;
    _sh0_dims[1] = sh0_dim1;
    _sh0_dims[2] = sh0_dim2;
    _shN_dims[0] = num_points;
    _shN_dims[1] = shN_dim1;
    _shN_dims[2] = shN_dim2;

    // Allocate raw CUDA memory
    size_t means_size = num_points * 3 * sizeof(float);
    size_t opacity_size = num_points * 1 * sizeof(float);
    size_t rotation_size = num_points * 4 * sizeof(float);
    size_t scaling_size = num_points * 3 * sizeof(float);
    size_t sh0_size = _sh0_dims[0] * _sh0_dims[1] * _sh0_dims[2] * sizeof(float);
    size_t shN_size = _shN_dims[0] * _shN_dims[1] * _shN_dims[2] * sizeof(float);

    cudaMalloc(&_means_cuda, means_size);
    cudaMalloc(&_opacity_cuda, opacity_size);
    cudaMalloc(&_rotation_cuda, rotation_size);
    cudaMalloc(&_scaling_cuda, scaling_size);
    cudaMalloc(&_sh0_cuda, sh0_size);
    cudaMalloc(&_shN_cuda, shN_size);

    // Allocate gradient memory
    cudaMalloc(&_means_grad_cuda, means_size);
    cudaMalloc(&_opacity_grad_cuda, opacity_size);
    cudaMalloc(&_rotation_grad_cuda, rotation_size);
    cudaMalloc(&_scaling_grad_cuda, scaling_size);
    cudaMalloc(&_sh0_grad_cuda, sh0_size);
    cudaMalloc(&_shN_grad_cuda, shN_size);

    // Initialize gradients to zero
    cudaMemset(_means_grad_cuda, 0, means_size);
    cudaMemset(_opacity_grad_cuda, 0, opacity_size);
    cudaMemset(_rotation_grad_cuda, 0, rotation_size);
    cudaMemset(_scaling_grad_cuda, 0, scaling_size);
    cudaMemset(_sh0_grad_cuda, 0, sh0_size);
    cudaMemset(_shN_grad_cuda, 0, shN_size);

    // Create persistent tensor wrappers WITHOUT requires_grad
    const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Create data tensors - NO requires_grad!
    _means_tensor = torch::from_blob(_means_cuda, {num_points, 3}, tensor_opts);
    _opacity_tensor = torch::from_blob(_opacity_cuda, {num_points, 1}, tensor_opts);
    _rotation_tensor = torch::from_blob(_rotation_cuda, {num_points, 4}, tensor_opts);
    _scaling_tensor = torch::from_blob(_scaling_cuda, {num_points, 3}, tensor_opts);
    _sh0_tensor = torch::from_blob(_sh0_cuda, {_sh0_dims[0], _sh0_dims[1], _sh0_dims[2]}, tensor_opts);
    _shN_tensor = torch::from_blob(_shN_cuda, {_shN_dims[0], _shN_dims[1], _shN_dims[2]}, tensor_opts);

    // DON'T set requires_grad(true) - we compute gradients manually!
}

    void SplatData::ensure_grad_allocated() {
        // Since we allocate gradients in allocate_cuda_memory_and_tensors,
        // we just need to create tensor wrappers if they don't exist
        // No autograd needed since we compute gradients manually

        const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        // Only create gradient tensor wrappers if not already defined
        // These are just views into our raw gradient memory
        if (!_means_tensor.grad().defined() && _means_grad_cuda) {
            _means_tensor.mutable_grad() = torch::from_blob(_means_grad_cuda,
                {_num_points, 3}, tensor_opts);
        }
        if (!_opacity_tensor.grad().defined() && _opacity_grad_cuda) {
            _opacity_tensor.mutable_grad() = torch::from_blob(_opacity_grad_cuda,
                {_num_points, 1}, tensor_opts);
        }
        if (!_rotation_tensor.grad().defined() && _rotation_grad_cuda) {
            _rotation_tensor.mutable_grad() = torch::from_blob(_rotation_grad_cuda,
                {_num_points, 4}, tensor_opts);
        }
        if (!_scaling_tensor.grad().defined() && _scaling_grad_cuda) {
            _scaling_tensor.mutable_grad() = torch::from_blob(_scaling_grad_cuda,
                {_num_points, 3}, tensor_opts);
        }
        if (!_sh0_tensor.grad().defined() && _sh0_grad_cuda) {
            _sh0_tensor.mutable_grad() = torch::from_blob(_sh0_grad_cuda,
                {_sh0_dims[0], _sh0_dims[1], _sh0_dims[2]}, tensor_opts);
        }
        if (!_shN_tensor.grad().defined() && _shN_grad_cuda) {
            _shN_tensor.mutable_grad() = torch::from_blob(_shN_grad_cuda,
                {_shN_dims[0], _shN_dims[1], _shN_dims[2]}, tensor_opts);
        }
    }

    void SplatData::free_cuda_memory() {
        if (_means_cuda) cudaFree(_means_cuda);
        if (_opacity_cuda) cudaFree(_opacity_cuda);
        if (_rotation_cuda) cudaFree(_rotation_cuda);
        if (_scaling_cuda) cudaFree(_scaling_cuda);
        if (_sh0_cuda) cudaFree(_sh0_cuda);
        if (_shN_cuda) cudaFree(_shN_cuda);

        if (_means_grad_cuda) cudaFree(_means_grad_cuda);
        if (_opacity_grad_cuda) cudaFree(_opacity_grad_cuda);
        if (_rotation_grad_cuda) cudaFree(_rotation_grad_cuda);
        if (_scaling_grad_cuda) cudaFree(_scaling_grad_cuda);
        if (_sh0_grad_cuda) cudaFree(_sh0_grad_cuda);
        if (_shN_grad_cuda) cudaFree(_shN_grad_cuda);

        _means_cuda = nullptr;
        _opacity_cuda = nullptr;
        _rotation_cuda = nullptr;
        _scaling_cuda = nullptr;
        _sh0_cuda = nullptr;
        _shN_cuda = nullptr;

        _means_grad_cuda = nullptr;
        _opacity_grad_cuda = nullptr;
        _rotation_grad_cuda = nullptr;
        _scaling_grad_cuda = nullptr;
        _sh0_grad_cuda = nullptr;
        _shN_grad_cuda = nullptr;
    }

    // Constructor from tensors - copies to raw CUDA memory
    SplatData::SplatData(int sh_degree,
                         torch::Tensor means,
                         torch::Tensor sh0,
                         torch::Tensor shN,
                         torch::Tensor scaling,
                         torch::Tensor rotation,
                         torch::Tensor opacity,
                         float scene_scale)
        : _max_sh_degree{sh_degree},
          _active_sh_degree{0},
          _scene_scale{scene_scale} {

        // Allocate memory and create tensor wrappers
        allocate_cuda_memory_and_tensors(
            means.size(0),
            sh0.size(1), sh0.size(2),
            shN.size(1), shN.size(2)
        );

        // Copy data from input tensors to raw CUDA memory
        cudaMemcpy(_means_cuda, means.contiguous().data_ptr<float>(),
                  means.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(_opacity_cuda, opacity.contiguous().data_ptr<float>(),
                  opacity.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(_rotation_cuda, rotation.contiguous().data_ptr<float>(),
                  rotation.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(_scaling_cuda, scaling.contiguous().data_ptr<float>(),
                  scaling.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(_sh0_cuda, sh0.contiguous().data_ptr<float>(),
                  sh0.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(_shN_cuda, shN.contiguous().data_ptr<float>(),
                  shN.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Move constructor
    SplatData::SplatData(SplatData&& other) noexcept
        : _active_sh_degree(other._active_sh_degree),
          _max_sh_degree(other._max_sh_degree),
          _scene_scale(other._scene_scale),
          _num_points(other._num_points),
          _means_cuda(other._means_cuda),
          _sh0_cuda(other._sh0_cuda),
          _shN_cuda(other._shN_cuda),
          _scaling_cuda(other._scaling_cuda),
          _rotation_cuda(other._rotation_cuda),
          _opacity_cuda(other._opacity_cuda),
          _means_grad_cuda(other._means_grad_cuda),
          _sh0_grad_cuda(other._sh0_grad_cuda),
          _shN_grad_cuda(other._shN_grad_cuda),
          _scaling_grad_cuda(other._scaling_grad_cuda),
          _rotation_grad_cuda(other._rotation_grad_cuda),
          _opacity_grad_cuda(other._opacity_grad_cuda),
          _means_tensor(std::move(other._means_tensor)),
          _opacity_tensor(std::move(other._opacity_tensor)),
          _rotation_tensor(std::move(other._rotation_tensor)),
          _scaling_tensor(std::move(other._scaling_tensor)),
          _sh0_tensor(std::move(other._sh0_tensor)),
          _shN_tensor(std::move(other._shN_tensor)),
          _densification_info(std::move(other._densification_info)) {

        std::memcpy(_sh0_dims, other._sh0_dims, sizeof(_sh0_dims));
        std::memcpy(_shN_dims, other._shN_dims, sizeof(_shN_dims));

        // Clear source
        other._means_cuda = nullptr;
        other._sh0_cuda = nullptr;
        other._shN_cuda = nullptr;
        other._scaling_cuda = nullptr;
        other._rotation_cuda = nullptr;
        other._opacity_cuda = nullptr;
        other._means_grad_cuda = nullptr;
        other._sh0_grad_cuda = nullptr;
        other._shN_grad_cuda = nullptr;
        other._scaling_grad_cuda = nullptr;
        other._rotation_grad_cuda = nullptr;
        other._opacity_grad_cuda = nullptr;
    }

    // Move assignment operator
    SplatData& SplatData::operator=(SplatData&& other) noexcept {
        if (this != &other) {
            // Wait for any pending saves to complete
            wait_for_saves();

            // Free existing memory
            free_cuda_memory();

            // Move scalar members
            _active_sh_degree = other._active_sh_degree;
            _max_sh_degree = other._max_sh_degree;
            _scene_scale = other._scene_scale;
            _num_points = other._num_points;

            std::memcpy(_sh0_dims, other._sh0_dims, sizeof(_sh0_dims));
            std::memcpy(_shN_dims, other._shN_dims, sizeof(_shN_dims));

            // Move pointers
            _means_cuda = other._means_cuda;
            _sh0_cuda = other._sh0_cuda;
            _shN_cuda = other._shN_cuda;
            _scaling_cuda = other._scaling_cuda;
            _rotation_cuda = other._rotation_cuda;
            _opacity_cuda = other._opacity_cuda;
            _means_grad_cuda = other._means_grad_cuda;
            _sh0_grad_cuda = other._sh0_grad_cuda;
            _shN_grad_cuda = other._shN_grad_cuda;
            _scaling_grad_cuda = other._scaling_grad_cuda;
            _rotation_grad_cuda = other._rotation_grad_cuda;
            _opacity_grad_cuda = other._opacity_grad_cuda;

            // Move tensors
            _means_tensor = std::move(other._means_tensor);
            _opacity_tensor = std::move(other._opacity_tensor);
            _rotation_tensor = std::move(other._rotation_tensor);
            _scaling_tensor = std::move(other._scaling_tensor);
            _sh0_tensor = std::move(other._sh0_tensor);
            _shN_tensor = std::move(other._shN_tensor);

            _densification_info = std::move(other._densification_info);

            // Clear source
            other._means_cuda = nullptr;
            other._sh0_cuda = nullptr;
            other._shN_cuda = nullptr;
            other._scaling_cuda = nullptr;
            other._rotation_cuda = nullptr;
            other._opacity_cuda = nullptr;
            other._means_grad_cuda = nullptr;
            other._sh0_grad_cuda = nullptr;
            other._shN_grad_cuda = nullptr;
            other._scaling_grad_cuda = nullptr;
            other._rotation_grad_cuda = nullptr;
            other._opacity_grad_cuda = nullptr;
        }
        return *this;
    }

    SplatData::~SplatData() {
        wait_for_saves();
        free_cuda_memory();
    }

    // Computed getters
    torch::Tensor SplatData::get_means() const {
        return _means_tensor;
    }

    torch::Tensor SplatData::get_opacity() const {
        return torch::sigmoid(_opacity_tensor).squeeze(-1);
    }

    torch::Tensor SplatData::get_rotation() const {
        return torch::nn::functional::normalize(_rotation_tensor,
                                                torch::nn::functional::NormalizeFuncOptions().dim(-1));
    }

    torch::Tensor SplatData::get_scaling() const {
        return torch::exp(_scaling_tensor);
    }

    torch::Tensor SplatData::get_shs() const {
        return torch::cat({_sh0_tensor, _shN_tensor}, 1);
    }

    SplatData& SplatData::transform(const glm::mat4& transform_matrix) {
        LOG_TIMER("SplatData::transform");

        if (_num_points == 0) {
            return *this; // Nothing to transform
        }

        // Launch CUDA kernel to transform positions
        gs::training::launch_transform_positions(
            _means_cuda,
            &transform_matrix[0][0],  // Pass as float[16]
            _num_points,
            0  // stream
        );

        // Extract rotation from transform matrix
        glm::mat3 rot_mat(transform_matrix);

        // Normalize columns to remove scale
        glm::vec3 scale;
        for (int i = 0; i < 3; ++i) {
            scale[i] = glm::length(rot_mat[i]);
            if (scale[i] > 0.0f) {
                rot_mat[i] /= scale[i];
            }
        }

        // Convert rotation matrix to quaternion
        glm::quat rotation = glm::quat_cast(rot_mat);

        // Transform rotations (quaternions) if there's rotation
        if (std::abs(rotation.w - 1.0f) > 1e-6f) {
            float rot_quat[4] = {rotation.w, rotation.x, rotation.y, rotation.z};

            gs::training::launch_transform_quaternions(
                _rotation_cuda,
                rot_quat,
                _num_points,
                0  // stream
            );
        }

        // Transform scaling (if non-uniform scale is present)
        if (std::abs(scale.x - 1.0f) > 1e-6f ||
            std::abs(scale.y - 1.0f) > 1e-6f ||
            std::abs(scale.z - 1.0f) > 1e-6f) {

            // Average scale factor (for isotropic gaussian scaling)
            float avg_scale = (scale.x + scale.y + scale.z) / 3.0f;

            // Since _scaling is log(scale), we add log of the scale factor
            gs::training::launch_add_scalar_to_tensor(
                _scaling_cuda,
                std::log(avg_scale),
                _num_points * 3,
                0  // stream
            );
        }

        // Update scene scale using CUDA kernel
        float* scene_center = nullptr;
        cudaMalloc(&scene_center, 3 * sizeof(float));

        // Compute mean of positions
        gs::training::launch_compute_mean_3d(
            _means_cuda,
            scene_center,
            _num_points,
            0  // stream
        );

        // Compute median distance
        float* distances = nullptr;
        cudaMalloc(&distances, _num_points * sizeof(float));

        gs::training::launch_compute_distances_from_center(
            _means_cuda,
            scene_center,
            distances,
            _num_points,
            0  // stream
        );

        // Compute median (simplified - just use mean for now)
        float* median_dist = nullptr;
        cudaMalloc(&median_dist, sizeof(float));

        gs::training::launch_compute_mean(
            distances,
            median_dist,
            _num_points,
            0  // stream
        );

        cudaDeviceSynchronize();

        // Copy median back to host
        float new_scene_scale;
        cudaMemcpy(&new_scene_scale, median_dist, sizeof(float), cudaMemcpyDeviceToHost);

        // Update scene scale if significant change
        if (std::abs(new_scene_scale - _scene_scale) > _scene_scale * 0.1f) {
            _scene_scale = new_scene_scale;
        }

        // Clean up temporary buffers
        cudaFree(scene_center);
        cudaFree(distances);
        cudaFree(median_dist);

        LOG_DEBUG("Transformed {} gaussians", _num_points);
        return *this;
    }

    // Utility method
    void SplatData::increment_sh_degree() {
        if (_active_sh_degree < _max_sh_degree) {
            _active_sh_degree++;
        }
    }

    // Get attribute names for PLY format
    std::vector<std::string> SplatData::get_attribute_names() const {
        std::vector<std::string> a{"x", "y", "z", "nx", "ny", "nz"};

        for (int i = 0; i < _sh0_dims[1] * _sh0_dims[2]; ++i)
            a.emplace_back("f_dc_" + std::to_string(i));
        for (int i = 0; i < _shN_dims[1] * _shN_dims[2]; ++i)
            a.emplace_back("f_rest_" + std::to_string(i));

        a.emplace_back("opacity");

        for (int i = 0; i < 3; ++i)
            a.emplace_back("scale_" + std::to_string(i));
        for (int i = 0; i < 4; ++i)
            a.emplace_back("rot_" + std::to_string(i));

        return a;
    }

    void SplatData::wait_for_saves() const {
        std::lock_guard<std::mutex> lock(_save_mutex);

        // Wait for all pending saves
        for (auto& future : _save_futures) {
            if (future.valid()) {
                try {
                    future.wait();
                } catch (const std::exception& e) {
                    LOG_ERROR("Error waiting for save to complete: {}", e.what());
                }
            }
        }
        _save_futures.clear();
    }

    void SplatData::cleanup_finished_saves() const {
        std::lock_guard<std::mutex> lock(_save_mutex);

        // Remove completed futures
        _save_futures.erase(
            std::remove_if(_save_futures.begin(), _save_futures.end(),
                           [](const std::future<void>& f) {
                               return !f.valid() ||
                                      f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                           }),
            _save_futures.end());

        // Log if we have many pending saves (might indicate a problem)
        if (_save_futures.size() > 5) {
            LOG_WARN("Multiple saves pending: {} operations in queue", _save_futures.size());
        }
    }

    // Export to PLY
    void SplatData::save_ply(const std::filesystem::path& root, int iteration, bool join_threads) const {
        auto pc = to_point_cloud();

        if (join_threads) {
            // Synchronous save - wait for completion
            write_ply_impl(pc, root, iteration);
        } else {
            // Asynchronous save
            cleanup_finished_saves();

            std::lock_guard<std::mutex> lock(_save_mutex);
            _save_futures.emplace_back(
                std::async(std::launch::async, [pc = std::move(pc), root, iteration]() {
                    try {
                        write_ply_impl(pc, root, iteration);
                    } catch (const std::exception& e) {
                        LOG_ERROR("Failed to save PLY for iteration {}: {}", iteration, e.what());
                    }
                }));
        }
    }

    // Export to SOG
    void SplatData::save_sog(const std::filesystem::path& root, int iteration, int kmeans_iterations, bool join_threads) const {
        // SOG must always be synchronous - k-means clustering is too heavy for async
        // and the shared data access patterns don't work well with async execution
        write_sog_impl(*this, root, iteration, kmeans_iterations);
    }

    PointCloud SplatData::to_point_cloud() const {
        PointCloud pc;

        // Allocate Gaussian point cloud
        pc.allocate_gaussian(_num_points, _sh0_dims[1], _sh0_dims[2], _shN_dims[1], _shN_dims[2]);

        // Copy data from raw CUDA memory to PointCloud
        cudaMemcpy(pc.means_cuda, _means_cuda, _num_points * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemset(pc.normals_cuda, 0, _num_points * 3 * sizeof(float)); // Normals are zeros
        cudaMemcpy(pc.sh0_cuda, _sh0_cuda, _sh0_dims[0] * _sh0_dims[1] * _sh0_dims[2] * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(pc.shN_cuda, _shN_cuda, _shN_dims[0] * _shN_dims[1] * _shN_dims[2] * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(pc.opacity_cuda, _opacity_cuda, _num_points * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(pc.scaling_cuda, _scaling_cuda, _num_points * 3 * sizeof(float), cudaMemcpyDeviceToDevice);

        // Need to normalize quaternions before saving
        float* normalized_rotations = nullptr;
        cudaMalloc(&normalized_rotations, _num_points * 4 * sizeof(float));

        // Copy and normalize rotations using a simple kernel or on CPU
        cudaMemcpy(normalized_rotations, _rotation_cuda, _num_points * 4 * sizeof(float), cudaMemcpyDeviceToDevice);

        // For now, just copy as-is (normalization should be done by a kernel)
        cudaMemcpy(pc.rotation_cuda, normalized_rotations, _num_points * 4 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(normalized_rotations);

        // Set attribute names for PLY export
        pc.attribute_names = get_attribute_names();

        return pc;
    }

    std::expected<SplatData, std::string> SplatData::init_model_from_pointcloud(
        const param::TrainingParameters& params,
        torch::Tensor scene_center,
        const PointCloud& pcd) {

        try {
            // Create torch tensors from PointCloud raw memory
            torch::Tensor positions;
            torch::Tensor colors;

            if (pcd.means_cuda && pcd.colors_cuda) {
                // Wrap existing CUDA memory
                positions = torch::from_blob(
                    pcd.means_cuda,
                    {static_cast<long>(pcd.num_points), 3},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
                ).clone(); // Clone to ensure ownership

                colors = torch::from_blob(
                    pcd.colors_cuda,
                    {static_cast<long>(pcd.num_points), 3},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
                ).clone() * 255.0f; // Convert from [0,1] to [0,255]
            } else if (params.optimization.random) {
                // Random initialization
                const int num_points = params.optimization.init_num_pts;
                const float extent = params.optimization.init_extent;
                const auto f32_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

                positions = (torch::rand({num_points, 3}, f32_cuda) * 2.0f - 1.0f) * extent;
                colors = torch::rand({num_points, 3}, f32_cuda) * 255.0f;
            } else {
                return std::unexpected("No valid point cloud data and random initialization is disabled");
            }

            scene_center = scene_center.to(positions.device());
            const torch::Tensor dists = torch::norm(positions - scene_center, 2, 1);
            const auto scene_scale = dists.median().item<float>();

            auto rgb_to_sh = [](const torch::Tensor& rgb) {
                constexpr float kInvSH = 0.28209479177387814f;
                return (rgb - 0.5f) / kInvSH;
            };

            const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
            const auto f32_cuda = f32.device(torch::kCUDA);

            // 1. means
            torch::Tensor means;
            if (params.optimization.random) {
                // Scale positions
                means = (positions * scene_scale).to(torch::kCUDA);
            } else {
                means = positions.to(torch::kCUDA);
            }

            // 2. scaling (log(σ))
            auto nn_dist = torch::clamp_min(compute_mean_neighbor_distances(means), 1e-7);
            auto scaling = torch::log(torch::sqrt(nn_dist) * params.optimization.init_scaling)
                               .unsqueeze(-1)
                               .repeat({1, 3})
                               .to(f32_cuda);

            // 3. rotation (quaternion, identity)
            auto rotation = torch::zeros({means.size(0), 4}, f32_cuda);
            rotation.index_put_({torch::indexing::Slice(), 0}, 1);

            // 4. opacity (inverse sigmoid of init value)
            auto opacity = torch::logit(params.optimization.init_opacity * torch::ones({means.size(0), 1}, f32_cuda));

            // 5. shs (SH coefficients)
            auto colors_float = colors.to(torch::kCUDA) / 255.0f;
            auto fused_color = rgb_to_sh(colors_float);

            const int64_t feature_shape = static_cast<int64_t>(std::pow(params.optimization.sh_degree + 1, 2));
            auto shs = torch::zeros({fused_color.size(0), 3, feature_shape}, f32_cuda);

            // Set DC coefficients
            shs.index_put_({torch::indexing::Slice(),
                            torch::indexing::Slice(),
                            0},
                           fused_color);

            auto sh0 = shs.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::Slice(0, 1)})
                           .transpose(1, 2)
                           .contiguous();

            auto shN = shs.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::Slice(1, torch::indexing::None)})
                           .transpose(1, 2)
                           .contiguous();

            std::println("Scene scale: {}", scene_scale);
            std::println("Initialized SplatData with:");
            std::println("  - {} points", means.size(0));
            std::println("  - Max SH degree: {}", params.optimization.sh_degree);
            std::println("  - Total SH coefficients: {}", feature_shape);
            std::cout << std::format("  - sh0 shape: {}\n", tensor_sizes_to_string(sh0.sizes()));
            std::cout << std::format("  - shN shape: {}\n", tensor_sizes_to_string(shN.sizes()));

            return SplatData(
                params.optimization.sh_degree,
                means.contiguous(),
                sh0.contiguous(),
                shN.contiguous(),
                scaling.contiguous(),
                rotation.contiguous(),
                opacity.contiguous(),
                scene_scale);

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize SplatData: {}", e.what()));
        }
    }

    void SplatData::zero_grad_manual() {
        // Zero all gradients using the tensor interface
        // The tensor's grad() points to our raw gradient memory
        if (_means_tensor.grad().defined()) {
            _means_tensor.mutable_grad().zero_();
        }
        if (_opacity_tensor.grad().defined()) {
            _opacity_tensor.mutable_grad().zero_();
        }
        if (_rotation_tensor.grad().defined()) {
            _rotation_tensor.mutable_grad().zero_();
        }
        if (_scaling_tensor.grad().defined()) {
            _scaling_tensor.mutable_grad().zero_();
        }
        if (_sh0_tensor.grad().defined()) {
            _sh0_tensor.mutable_grad().zero_();
        }
        if (_shN_tensor.grad().defined()) {
            _shN_tensor.mutable_grad().zero_();
        }
    }
} // namespace gs