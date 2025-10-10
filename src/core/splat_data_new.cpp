/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data_new.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/sogs_new.hpp"
#include "external/tinyply.hpp"
#include "kernels/morton_encoding_new.cuh"
#include "kernels/training_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <sstream>

namespace gs {

    namespace {
        /**
         * @brief Compute mean distance to K nearest neighbors using Tensor operations
         *
         * This is a simplified version that computes pairwise distances and finds
         * the K smallest for each point. For very large point clouds, consider
         * using a KD-tree implementation.
         */
        Tensor compute_mean_neighbor_distances(const Tensor& points, int k = 3) {
            const size_t N = points.size(0);

            if (N <= 1) {
                return Tensor::full({N}, 0.01f, points.device(), DataType::Float32);
            }

            // For small point clouds, compute all pairwise distances
            if (N < 10000) {
                // Compute pairwise distances: ||points[i] - points[j]||^2
                // Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3) -> (N, N)
                auto p1 = points.unsqueeze(1);  // [N, 1, 3]
                auto p2 = points.unsqueeze(0);  // [1, N, 3]
                auto diffs = p1 - p2;            // [N, N, 3]
                auto dist_sq = (diffs * diffs).sum(2);  // [N, N]

                // Sort distances for each point and take mean of k smallest (excluding self)
                auto sorted_dists = dist_sq.sort(1);  // Returns (values, indices)
                auto k_nearest = sorted_dists.first.slice(1, 1, std::min(k + 1, static_cast<int>(N)));  // Skip self at index 0

                // Take mean and sqrt
                auto mean_dist_sq = k_nearest.mean(1);
                return mean_dist_sq.sqrt().clamp_min(1e-7f);
            } else {
                // For large point clouds, use a sampling-based approximation
                LOG_WARN("Using sampling-based neighbor distance for {} points", N);

                const size_t sample_size = 1000;
                auto indices = Tensor::randint({sample_size}, 0, static_cast<int>(N), points.device());
                auto sampled_points = points.index_select(0, indices);

                // Compute distances to sampled points for all points
                auto p1 = points.unsqueeze(1);  // [N, 1, 3]
                auto p2 = sampled_points.unsqueeze(0);  // [1, S, 3]
                auto diffs = p1 - p2;
                auto dist_sq = (diffs * diffs).sum(2);  // [N, S]

                // Sort and take k smallest
                auto sorted_dists = dist_sq.sort(1);
                auto k_nearest = sorted_dists.first.slice(1, 0, std::min(k, static_cast<int>(sample_size)));

                auto mean_dist_sq = k_nearest.mean(1);
                return mean_dist_sq.sqrt().clamp_min(1e-7f);
            }
        }

        /**
         * @brief Convert RGB to SH DC coefficient
         */
        Tensor rgb_to_sh(const Tensor& rgb) {
            constexpr float kInvSH = 0.28209479177387814f;
            return (rgb - 0.5f) / kInvSH;
        }

        /**
         * @brief Write PLY file using tinyply
         */
        void write_ply_impl(const PointCloudNew& pc,
                            const std::filesystem::path& root,
                            int iteration) {
            std::filesystem::create_directories(root);

            // Convert to CPU for file writing
            auto pc_cpu = pc.cpu();

            // Get raw pointers to data
            std::vector<float*> data_ptrs;
            data_ptrs.push_back(pc_cpu.means.ptr<float>());

            if (pc_cpu.normals.is_valid()) {
                data_ptrs.push_back(pc_cpu.normals.ptr<float>());
            }
            if (pc_cpu.sh0.is_valid()) {
                data_ptrs.push_back(pc_cpu.sh0.ptr<float>());
            }
            if (pc_cpu.shN.is_valid()) {
                data_ptrs.push_back(pc_cpu.shN.ptr<float>());
            }
            if (pc_cpu.opacity.is_valid()) {
                data_ptrs.push_back(pc_cpu.opacity.ptr<float>());
            }
            if (pc_cpu.scaling.is_valid()) {
                data_ptrs.push_back(pc_cpu.scaling.ptr<float>());
            }
            if (pc_cpu.rotation.is_valid()) {
                data_ptrs.push_back(pc_cpu.rotation.ptr<float>());
            }

            // Create PLY file
            tinyply::PlyFile ply;
            size_t attr_off = 0;

            for (size_t i = 0; i < data_ptrs.size(); ++i) {
                // Determine number of attributes for this data block
                size_t num_attrs = 3; // Default for positions, normals, scaling

                if (i == data_ptrs.size() - 1 && pc_cpu.attribute_names.back().find("rot_") != std::string::npos) {
                    num_attrs = 4; // Rotation has 4 components
                } else if (pc_cpu.attribute_names[attr_off] == "opacity") {
                    num_attrs = 1; // Opacity is scalar
                } else if (pc_cpu.attribute_names[attr_off].find("f_dc_") != std::string::npos ||
                           pc_cpu.attribute_names[attr_off].find("f_rest_") != std::string::npos) {
                    // Count how many consecutive f_dc_ or f_rest_ attributes
                    num_attrs = 0;
                    size_t j = attr_off;
                    while (j < pc_cpu.attribute_names.size() &&
                           (pc_cpu.attribute_names[j].find("f_dc_") != std::string::npos ||
                            pc_cpu.attribute_names[j].find("f_rest_") != std::string::npos)) {
                        num_attrs++;
                        j++;
                    }
                }

                std::vector<std::string> attrs(pc_cpu.attribute_names.begin() + attr_off,
                                               pc_cpu.attribute_names.begin() + attr_off + num_attrs);

                ply.add_properties_to_element(
                    "vertex",
                    attrs,
                    tinyply::Type::FLOAT32,
                    pc_cpu.size(),
                    reinterpret_cast<uint8_t*>(data_ptrs[i]),
                    tinyply::Type::INVALID, 0);

                attr_off += num_attrs;
            }

            // Write file
            auto filename = root / ("splat_" + std::to_string(iteration) + ".ply");
            std::filebuf fb;
            fb.open(filename, std::ios::out | std::ios::binary);
            std::ostream out_stream(&fb);
            ply.write(out_stream, /*binary=*/true);
        }
    } // anonymous namespace

    // Constructor
    SplatDataNew::SplatDataNew(int sh_degree,
                               Tensor means_in,
                               Tensor sh0_in,
                               Tensor shN_in,
                               Tensor scaling_in,
                               Tensor rotation_in,
                               Tensor opacity_in,
                               float scene_scale_in)
        : means(std::move(means_in)),
          opacity(std::move(opacity_in)),
          rotation(std::move(rotation_in)),
          scaling(std::move(scaling_in)),
          sh0(std::move(sh0_in)),
          shN(std::move(shN_in)),
          max_sh_degree(sh_degree),
          active_sh_degree(0),
          scene_scale(scene_scale_in) {

        // Initialize densification info
        if (means.is_valid()) {
            densification_info = Tensor::zeros({means.size(0)}, means.device(), DataType::Float32);
        }
    }

    // Get combined SH coefficients
    Tensor SplatDataNew::get_shs() const {
        if (!sh0.is_valid()) {
            return Tensor();
        }

        if (!shN.is_valid() || shN.numel() == 0) {
            return sh0;
        }

        // Concatenate along dimension 2 (coefficients): [N, 3, 1] + [N, 3, K-1] = [N, 3, K]
        return Tensor::cat({sh0, shN}, 2);
    }

    // Get opacity with sigmoid activation
    Tensor SplatDataNew::get_opacity() const {
        return opacity.sigmoid().squeeze(-1);
    }

    // Get rotation with normalization
    Tensor SplatDataNew::get_rotation() const {
        return rotation.normalize(-1);
    }

    // Get scaling with exp activation
    Tensor SplatDataNew::get_scaling() const {
        return scaling.exp();
    }

    // Increment SH degree
    void SplatDataNew::increment_sh_degree() {
        if (active_sh_degree < max_sh_degree) {
            active_sh_degree++;
        }
    }

    // Transform gaussians
    SplatDataNew& SplatDataNew::transform(const glm::mat4& transform_matrix) {
        LOG_TIMER("SplatDataNew::transform");

        if (!means.is_valid() || means.size(0) == 0) {
            return *this;
        }

        const size_t N = means.size(0);

        // Ensure data is on CUDA
        if (means.device() != Device::CUDA) {
            LOG_ERROR("Transform requires data on CUDA device");
            return *this;
        }

        // Convert GLM matrix to float array (GLM is column-major, kernel expects row-major)
        float transform_array[16];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                transform_array[i * 4 + j] = transform_matrix[j][i];
            }
        }

        // 1. Transform positions using kernel
        gs::training::launch_transform_positions(
            means.ptr<float>(),
            transform_array,
            static_cast<int>(N),
            nullptr  // default stream
        );

        // 2. Extract rotation from transform matrix
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
        glm::quat rot_quat = glm::quat_cast(rot_mat);

        // 3. Transform rotations if there's non-identity rotation
        if (std::abs(rot_quat.w - 1.0f) > 1e-6f ||
            std::abs(rot_quat.x) > 1e-6f ||
            std::abs(rot_quat.y) > 1e-6f ||
            std::abs(rot_quat.z) > 1e-6f) {

            float rot_quat_array[4] = {rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z};

            gs::training::launch_transform_quaternions(
                rotation.ptr<float>(),
                rot_quat_array,
                static_cast<int>(N),
                nullptr  // default stream
            );
        }

        // 4. Transform scaling if non-uniform scale is present
        if (std::abs(scale.x - 1.0f) > 1e-6f ||
            std::abs(scale.y - 1.0f) > 1e-6f ||
            std::abs(scale.z - 1.0f) > 1e-6f) {

            // Average scale factor (for isotropic gaussian scaling)
            float avg_scale = (scale.x + scale.y + scale.z) / 3.0f;

            // Since scaling is log(scale), we add log of the scale factor
            float log_scale = std::log(avg_scale);

            gs::training::launch_add_scalar_to_tensor(
                scaling.ptr<float>(),
                log_scale,
                static_cast<int>(N * 3),  // 3 components per gaussian
                nullptr  // default stream
            );
        }

        // 5. Update scene scale using kernels
        // Allocate temporary GPU memory for computations
        auto scene_center_tensor = Tensor::zeros({3}, Device::CUDA, DataType::Float32);
        auto distances_tensor = Tensor::zeros({N}, Device::CUDA, DataType::Float32);
        auto median_tensor = Tensor::zeros({1}, Device::CUDA, DataType::Float32);

        // Compute mean of positions
        gs::training::launch_compute_mean_3d(
            means.ptr<float>(),
            scene_center_tensor.ptr<float>(),
            static_cast<int>(N),
            nullptr
        );

        // Compute distances from center
        gs::training::launch_compute_distances_from_center(
            means.ptr<float>(),
            scene_center_tensor.ptr<float>(),
            distances_tensor.ptr<float>(),
            static_cast<int>(N),
            nullptr
        );

        // Compute median distance (using mean as approximation for now)
        gs::training::launch_compute_mean(
            distances_tensor.ptr<float>(),
            median_tensor.ptr<float>(),
            static_cast<int>(N),
            nullptr
        );

        // Synchronize to ensure all operations complete
        cudaDeviceSynchronize();

        // Copy median back to host
        float new_scene_scale = median_tensor.item();

        // Update scene scale if significant change
        if (std::abs(new_scene_scale - scene_scale) > scene_scale * 0.1f) {
            LOG_DEBUG("Updating scene scale: {} -> {}", scene_scale, new_scene_scale);
            scene_scale = new_scene_scale;
        }

        LOG_DEBUG("Transformed {} gaussians", N);
        return *this;
    }

    // Export to PLY
    void SplatDataNew::save_ply(const std::filesystem::path& root, int iteration) const {
        auto pc = to_point_cloud();
        write_ply_impl(pc, root, iteration);
    }

    // Export to SOG
    void SplatDataNew::save_sog(const std::filesystem::path& root, int iteration, int kmeans_iterations) const {
        namespace fs = std::filesystem;

        if (!is_valid()) {
            LOG_ERROR("Cannot save invalid SplatDataNew to SOG format");
            return;
        }

        // Create SOG subdirectory
        fs::path sog_dir = root / "sog";
        fs::create_directories(sog_dir);

        // Set up SOG write options - use .sog extension to create bundle
        core::SogWriteOptionsNew options{
            .iterations = kmeans_iterations,
            .output_path = sog_dir / ("splat_" + std::to_string(iteration) + ".sog")
        };

        // Write SOG format using the new interface
        auto result = core::write_sog_new(*this, options);
        if (!result) {
            LOG_ERROR("Failed to write SOG format: {}", result.error());
        } else {
            LOG_DEBUG("Successfully wrote SOG format for iteration {}", iteration);
        }
    }

    // Convert to point cloud
    PointCloudNew SplatDataNew::to_point_cloud() const {
        if (!means.is_valid() || means.size(0) == 0) {
            return PointCloudNew();
        }

        const size_t N = means.size(0);

        // Allocate Gaussian point cloud
        PointCloudNew pc;

        // Get SH dimensions
        size_t sh0_channels = sh0.is_valid() ? sh0.size(1) : 0;
        size_t sh0_coeffs = sh0.is_valid() ? sh0.size(2) : 0;
        size_t shN_channels = shN.is_valid() ? shN.size(1) : 0;
        size_t shN_coeffs = shN.is_valid() ? shN.size(2) : 0;

        pc.allocate_gaussian(N, sh0_channels, sh0_coeffs, shN_channels, shN_coeffs);

        // Copy data
        pc.means = means.clone();
        pc.opacity = opacity.clone();
        pc.rotation = rotation.clone(); // Store raw quaternions, not normalized
        pc.scaling = scaling.clone();

        if (sh0.is_valid()) {
            pc.sh0 = sh0.clone();
        }
        if (shN.is_valid()) {
            pc.shN = shN.clone();
        }

        // Normals are zeros for Gaussians
        if (pc.normals.is_valid()) {
            pc.normals.zero_();
        }

        // Set attribute names
        pc.attribute_names = get_attribute_names();

        return pc;
    }

    // Get attribute names for PLY format
    std::vector<std::string> SplatDataNew::get_attribute_names() const {
        std::vector<std::string> attrs{"x", "y", "z", "nx", "ny", "nz"};

        // SH0 attributes
        if (sh0.is_valid()) {
            size_t sh0_size = sh0.size(1) * sh0.size(2);
            for (size_t i = 0; i < sh0_size; ++i) {
                attrs.emplace_back("f_dc_" + std::to_string(i));
            }
        }

        // SHN attributes
        if (shN.is_valid()) {
            size_t shN_size = shN.size(1) * shN.size(2);
            for (size_t i = 0; i < shN_size; ++i) {
                attrs.emplace_back("f_rest_" + std::to_string(i));
            }
        }

        // Opacity
        attrs.emplace_back("opacity");

        // Scaling
        for (int i = 0; i < 3; ++i) {
            attrs.emplace_back("scale_" + std::to_string(i));
        }

        // Rotation
        for (int i = 0; i < 4; ++i) {
            attrs.emplace_back("rot_" + std::to_string(i));
        }

        return attrs;
    }

    // Validate structure
    bool SplatDataNew::is_valid() const {
        if (!means.is_valid()) {
            LOG_ERROR("SplatDataNew: means is invalid");
            return false;
        }

        const size_t N = means.size(0);

        if (means.ndim() != 2 || means.size(1) != 3) {
            LOG_ERROR("SplatDataNew: means must be [N, 3]");
            return false;
        }

        if (!opacity.is_valid() || opacity.size(0) != N || opacity.size(1) != 1) {
            LOG_ERROR("SplatDataNew: opacity must be [N, 1]");
            return false;
        }

        if (!rotation.is_valid() || rotation.size(0) != N || rotation.size(1) != 4) {
            LOG_ERROR("SplatDataNew: rotation must be [N, 4]");
            return false;
        }

        if (!scaling.is_valid() || scaling.size(0) != N || scaling.size(1) != 3) {
            LOG_ERROR("SplatDataNew: scaling must be [N, 3]");
            return false;
        }

        if (!sh0.is_valid() || sh0.size(0) != N) {
            LOG_ERROR("SplatDataNew: sh0 must have N points");
            return false;
        }

        return true;
    }

    // Diagnostic string
    std::string SplatDataNew::str() const {
        std::ostringstream oss;
        oss << "SplatDataNew(";
        oss << "points=" << size();
        oss << ", sh_degree=" << active_sh_degree << "/" << max_sh_degree;
        oss << ", scene_scale=" << scene_scale;

        if (sh0.is_valid()) {
            oss << ", sh0=" << sh0.shape().str();
        }
        if (shN.is_valid()) {
            oss << ", shN=" << shN.shape().str();
        }

        oss << ")";
        return oss.str();
    }

    // Initialize from point cloud
    std::expected<SplatDataNew, std::string> SplatDataNew::init_model_from_pointcloud(
        const param::TrainingParameters& params,
        const Tensor& scene_center,
        const PointCloudNew& pcd) {

        try {
            Tensor positions;
            Tensor colors;

            if (pcd.means.is_valid() && pcd.colors.is_valid()) {
                // Use provided point cloud
                positions = pcd.means.clone();
                colors = pcd.colors.clone() * 255.0f; // Convert from [0,1] to [0,255]
            } else if (params.optimization.random) {
                // Random initialization
                const int num_points = params.optimization.init_num_pts;
                const float extent = params.optimization.init_extent;

                positions = (Tensor::rand({static_cast<size_t>(num_points), 3}, Device::CUDA) * 2.0f - 1.0f) * extent;
                colors = Tensor::rand({static_cast<size_t>(num_points), 3}, Device::CUDA) * 255.0f;
            } else {
                return std::unexpected("No valid point cloud data and random initialization is disabled");
            }

            // Ensure on CUDA
            positions = positions.cuda();
            colors = colors.cuda();

            // Compute scene scale
            auto scene_center_cuda = scene_center.cuda();
            auto dists = (positions - scene_center_cuda.unsqueeze(0)).norm(2, 1);

            // Compute median distance
            auto sorted_dists = dists.sort();
            size_t median_idx = sorted_dists.first.size(0) / 2;
            float scene_scale = sorted_dists.first[median_idx].item();

            // 1. Means
            Tensor means;
            if (params.optimization.random) {
                means = positions * scene_scale;
            } else {
                means = positions;
            }

            // 2. Scaling (log(σ))
            auto nn_dist = compute_mean_neighbor_distances(means);  // [N]
            auto log_dist = (nn_dist.sqrt() * params.optimization.init_scaling).log();  // [N]

            // Stack 3 copies to create [N, 3]
            auto scaling = Tensor::stack({log_dist, log_dist, log_dist}, 1);

            // 3. Rotation (quaternion, identity: w=1, x=y=z=0)
            auto rotation = Tensor::zeros({means.size(0), 4}, Device::CUDA, DataType::Float32);
            // Set first column to 1 (w component)
            auto rotation_w = rotation.slice(1, 0, 1);
            rotation_w.fill_(1.0f);

            // 4. Opacity (inverse sigmoid of init value)
            auto opacity = (Tensor::ones({means.size(0), 1}, Device::CUDA) * params.optimization.init_opacity).logit();

            // 5. SH coefficients
            auto colors_float = colors / 255.0f;
            auto fused_color = rgb_to_sh(colors_float);  // [N, 3]

            // Calculate number of SH coefficients
            const int64_t feature_shape = static_cast<int64_t>(std::pow(params.optimization.sh_degree + 1, 2));

            // sh0 is just the DC coefficient: [N, 3] -> [N, 3, 1]
            auto sh0 = fused_color.unsqueeze(-1);  // [N, 3, 1]

            // shN contains the rest of the coefficients (all zeros initially)
            auto shN = Tensor::zeros({fused_color.size(0), 3, static_cast<size_t>(feature_shape - 1)},
                                     Device::CUDA, DataType::Float32);  // [N, 3, K-1]

            LOG_INFO("Scene scale: {}", scene_scale);
            LOG_INFO("Initialized SplatDataNew with:");
            LOG_INFO("  - {} points", means.size(0));
            LOG_INFO("  - Max SH degree: {}", params.optimization.sh_degree);
            LOG_INFO("  - Total SH coefficients: {}", feature_shape);
            LOG_INFO("  - sh0 shape: {}", sh0.shape().str());
            LOG_INFO("  - shN shape: {}", shN.shape().str());

            return SplatDataNew(
                params.optimization.sh_degree,
                means.contiguous(),
                sh0.contiguous(),
                shN.contiguous(),
                scaling.contiguous(),
                rotation.contiguous(),
                opacity.contiguous(),
                scene_scale
            );

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize SplatDataNew: {}", e.what()));
        }
    }

} // namespace gs