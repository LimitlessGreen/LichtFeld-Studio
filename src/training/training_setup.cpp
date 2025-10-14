/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_setup.hpp"
#include "core/logger.hpp"
#include "core/point_cloud_new.hpp"
#include "core/splat_data_new.hpp"
#include "loader/loader.hpp"
#include "strategies/default_strategy.hpp"
#include "strategies/mcmc.hpp"
#include <format>
#include <random>

namespace gs::training {

    // Helper function to generate random point cloud (torch-free)
    static PointCloudNew generate_random_initialization() {
        constexpr int NUM_INIT_GAUSSIAN = 10000;
        constexpr uint64_t SEED = 8128;

        LOG_DEBUG("Generating random point cloud with {} points", NUM_INIT_GAUSSIAN);

        std::mt19937 gen(SEED);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        // Generate random positions in [-1, 1]^3
        std::vector<float> positions(NUM_INIT_GAUSSIAN * 3);
        for (int i = 0; i < NUM_INIT_GAUSSIAN * 3; ++i) {
            positions[i] = dist(gen) * 2.0f - 1.0f;
        }

        // Generate random colors in [0, 1]
        std::vector<float> colors(NUM_INIT_GAUSSIAN * 3);
        for (int i = 0; i < NUM_INIT_GAUSSIAN * 3; ++i) {
            colors[i] = dist(gen);
        }

        Tensor means = Tensor::from_vector(positions, TensorShape{NUM_INIT_GAUSSIAN, 3}, Device::CUDA);
        Tensor colors_tensor = Tensor::from_vector(colors, TensorShape{NUM_INIT_GAUSSIAN, 3}, Device::CUDA);

        return PointCloudNew(std::move(means), std::move(colors_tensor));
    }

    std::expected<TrainingSetup, std::string> setupTraining(const param::TrainingParameters& params) {
        // 1. Create loader
        auto loader = loader::Loader::create();

        // 2. Set up load options
        loader::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .images_folder = params.dataset.images,
            .validate_only = false,
            .progress = [](float percentage, const std::string& message) {
                LOG_DEBUG("[{:5.1f}%] {}", percentage, message);
            }};

        // 3. Load the dataset
        LOG_INFO("Loading dataset from: {}", params.dataset.data_path.string());
        auto load_result = loader->load(params.dataset.data_path, load_options);
        if (!load_result) {
            return std::unexpected(std::format("Failed to load dataset: {}", load_result.error()));
        }

        LOG_INFO("Dataset loaded successfully using {} loader", load_result->loader_used);

        // 4. Handle the loaded data based on type
        return std::visit([&params, &load_result](auto&& data) -> std::expected<TrainingSetup, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<gs::SplatDataNew>>) {
                // Direct PLY load - not supported for training
                return std::unexpected(
                    "Direct PLY loading is not supported for training. Please use a dataset format (COLMAP or Blender).");
            } else if constexpr (std::is_same_v<T, loader::LoadedScene>) {
                // Full scene data - set up training

                // Initialize model directly with point cloud
                std::expected<SplatDataNew, std::string> splat_result;
                if (params.init_ply.has_value()) {
                    // Load initialization PLY file
                    auto loader_ply = loader::Loader::create();
                    auto ply_load_result = loader_ply->load(params.init_ply.value());

                    if (!ply_load_result) {
                        splat_result = std::unexpected(std::format(
                            "Failed to load initialization PLY file '{}': {}",
                            params.init_ply.value(),
                            ply_load_result.error()));
                    } else {
                        try {
                            auto splat_ptr = std::get<std::shared_ptr<SplatDataNew>>(ply_load_result->data);
                            if (!splat_ptr) {
                                splat_result = std::unexpected(std::format(
                                    "Initialization PLY file '{}' returned null SplatData",
                                    params.init_ply.value()));
                            } else {
                                splat_result = std::move(*splat_ptr);
                            }
                        } catch (const std::bad_variant_access&) {
                            splat_result = std::unexpected(std::format(
                                "Initialization PLY file '{}' did not contain valid SplatData",
                                params.init_ply.value()));
                        }
                    }

                } else {
                    // Get point cloud or generate random one
                    PointCloudNew point_cloud_to_use;
                    if (data.point_cloud && data.point_cloud->size() > 0) {
                        // Use move semantics to transfer ownership
                        point_cloud_to_use = std::move(*data.point_cloud);
                        LOG_INFO("Using point cloud with {} points", point_cloud_to_use.size());
                    } else {
                        // Generate random point cloud if needed
                        LOG_INFO("No point cloud provided, using random initialization");
                        point_cloud_to_use = generate_random_initialization();
                    }

                    splat_result = SplatDataNew::init_model_from_pointcloud(
                        params,
                        load_result->scene_center,
                        point_cloud_to_use);
                }

                if (!splat_result) {
                    return std::unexpected(
                        std::format("Failed to initialize model: {}", splat_result.error()));
                }

                // 5. Create strategy
                std::unique_ptr<IStrategy> strategy;
                if (params.optimization.strategy == "mcmc") {
                    //strategy = std::make_unique<MCMC>(std::move(*splat_result));
                    LOG_DEBUG("Created MCMC strategy");
                } else {
                    //strategy = std::make_unique<DefaultStrategy>(std::move(*splat_result));
                    LOG_DEBUG("Created default strategy");
                }

                // Create trainer (without parameters)
                auto trainer = std::make_unique<Trainer>(
                    data.cameras,
                    std::move(strategy));

                return TrainingSetup{
                    .trainer = std::move(trainer),
                    .dataset = data.cameras,
                    .scene_center = load_result->scene_center};
            } else {
                return std::unexpected("Unknown data type returned from loader");
            }
        },
                          load_result->data);
    }

} // namespace gs::training