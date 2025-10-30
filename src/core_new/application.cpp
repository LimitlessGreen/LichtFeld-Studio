/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/application.hpp"
#include "core_new/argument_parser.hpp"
#include "core_new/logger.hpp"
#include "project/project.hpp"
#include "training_new/training_setup.hpp"
#include "visualizer_new/visualizer.hpp"
#include <cstring>
#ifdef WIN32
#include <windows.h>
#endif

namespace lfs::core {

    // Helper functions to convert between old gs::param and new lfs::core::param types
    // Cannot use memcpy because structures contain non-trivial types (std::filesystem::path, std::vector, std::string)
    static gs::param::DatasetConfig convertDatasetConfig(const lfs::core::param::DatasetConfig& src) {
        gs::param::DatasetConfig dst;
        dst.data_path = src.data_path;
        dst.output_path = src.output_path;
        dst.project_path = src.project_path;
        dst.images = src.images;
        dst.resize_factor = src.resize_factor;
        dst.test_every = src.test_every;
        dst.timelapse_images = src.timelapse_images;
        dst.timelapse_every = src.timelapse_every;
        dst.max_width = src.max_width;
        // Copy LoadingParams field-by-field (cannot use direct assignment due to different types)
        dst.loading_params.use_cpu_memory = src.loading_params.use_cpu_memory;
        dst.loading_params.min_cpu_free_memory_ratio = src.loading_params.min_cpu_free_memory_ratio;
        dst.loading_params.min_cpu_free_GB = src.loading_params.min_cpu_free_GB;
        dst.loading_params.use_fs_cache = src.loading_params.use_fs_cache;
        dst.loading_params.print_cache_status = src.loading_params.print_cache_status;
        dst.loading_params.print_status_freq_num = src.loading_params.print_status_freq_num;
        return dst;
    }

    static gs::param::OptimizationParameters convertOptimizationParams(const lfs::core::param::OptimizationParameters& src) {
        gs::param::OptimizationParameters dst;
        // Copy all scalar and simple types
        dst.iterations = src.iterations;
        dst.sh_degree_interval = src.sh_degree_interval;
        dst.means_lr = src.means_lr;
        dst.shs_lr = src.shs_lr;
        dst.opacity_lr = src.opacity_lr;
        dst.scaling_lr = src.scaling_lr;
        dst.rotation_lr = src.rotation_lr;
        dst.lambda_dssim = src.lambda_dssim;
        dst.min_opacity = src.min_opacity;
        dst.refine_every = src.refine_every;
        dst.start_refine = src.start_refine;
        dst.stop_refine = src.stop_refine;
        dst.grad_threshold = src.grad_threshold;
        dst.sh_degree = src.sh_degree;
        dst.opacity_reg = src.opacity_reg;
        dst.scale_reg = src.scale_reg;
        dst.init_opacity = src.init_opacity;
        dst.init_scaling = src.init_scaling;
        dst.num_workers = src.num_workers;
        dst.max_cap = src.max_cap;
        dst.eval_steps = src.eval_steps;
        dst.save_steps = src.save_steps;
        dst.skip_intermediate_saving = src.skip_intermediate_saving;
        dst.bg_modulation = src.bg_modulation;
        dst.enable_eval = src.enable_eval;
        dst.rc = src.rc;
        dst.enable_save_eval_images = src.enable_save_eval_images;
        dst.headless = src.headless;
        dst.render_mode = src.render_mode;
        dst.strategy = src.strategy;
        dst.preload_to_ram = src.preload_to_ram;
        dst.pose_optimization = src.pose_optimization;
        dst.use_bilateral_grid = src.use_bilateral_grid;
        dst.bilateral_grid_X = src.bilateral_grid_X;
        dst.bilateral_grid_Y = src.bilateral_grid_Y;
        dst.bilateral_grid_W = src.bilateral_grid_W;
        dst.bilateral_grid_lr = src.bilateral_grid_lr;
        dst.tv_loss_weight = src.tv_loss_weight;
        dst.prune_opacity = src.prune_opacity;
        dst.grow_scale3d = src.grow_scale3d;
        dst.grow_scale2d = src.grow_scale2d;
        dst.prune_scale3d = src.prune_scale3d;
        dst.prune_scale2d = src.prune_scale2d;
        dst.reset_every = src.reset_every;
        dst.pause_refine_after_reset = src.pause_refine_after_reset;
        dst.revised_opacity = src.revised_opacity;
        dst.gut = src.gut;
        dst.steps_scaler = src.steps_scaler;
        dst.antialiasing = src.antialiasing;
        dst.random = src.random;
        dst.init_num_pts = src.init_num_pts;
        dst.init_extent = src.init_extent;
        dst.save_sog = src.save_sog;
        dst.sog_iterations = src.sog_iterations;
        dst.enable_sparsity = src.enable_sparsity;
        dst.sparsify_steps = src.sparsify_steps;
        dst.init_rho = src.init_rho;
        dst.prune_ratio = src.prune_ratio;
        dst.config_file = src.config_file;
        return dst;
    }

    int run_headless_app(std::unique_ptr<param::TrainingParameters> params) {
        if (params->dataset.data_path.empty()) {
            LOG_ERROR("Headless mode requires --data-path");
            return -1;
        }

        LOG_INFO("Starting headless training...");

        auto project = gs::management::CreateNewProject(
            convertDatasetConfig(params->dataset),
            convertOptimizationParams(params->optimization));
        if (!project) {
            LOG_ERROR("Project creation failed");
            return -1;
        }

        auto setup_result = lfs::training::setupTraining(*params);
        if (!setup_result) {
            LOG_ERROR("Training setup failed: {}", setup_result.error());
            return -1;
        }

        setup_result->trainer->setProject(project);

        // Initialize trainer in headless mode with parameters
        auto init_result = setup_result->trainer->initialize(*params);
        if (!init_result) {
            LOG_ERROR("Failed to initialize trainer: {}", init_result.error());
            return -1;
        }

        auto train_result = setup_result->trainer->train();
        if (!train_result) {
            LOG_ERROR("Training error: {}", train_result.error());
            return -1;
        }

        LOG_INFO("Headless training completed successfully");
        return 0;
    }

    int run_gui_app(std::unique_ptr<param::TrainingParameters> params) {
        LOG_INFO("Starting viewer mode...");

        LOG_DEBUG("removing temporary projects");
        gs::management::RemoveTempUnlockedProjects();

        // Create visualizer with options
        auto viewer = lfs::vis::Visualizer::create({.title = "LichtFeld Studio",
                                                    .width = 1280,
                                                    .height = 720,
                                                    .antialiasing = params->optimization.antialiasing,
                                                    .enable_cuda_interop = true,
                                                    .gut = params->optimization.gut});

        if (!params->dataset.project_path.empty() &&
            !std::filesystem::exists(params->dataset.project_path)) {
            LOG_ERROR("Project file does not exist: {}", params->dataset.project_path.string());
            return -1;
        }

        if (std::filesystem::exists(params->dataset.project_path)) {
            bool success = viewer->openProject(params->dataset.project_path);
            if (!success) {
                LOG_ERROR("Error opening existing project");
                return -1;
            }
            if (!params->ply_path.empty()) {
                LOG_ERROR("Cannot open PLY and project from command line simultaneously");
                return -1;
            }
            if (!params->dataset.data_path.empty()) {
                LOG_ERROR("Cannot open new data_path and project from command line simultaneously");
                return -1;
            }
        } else { // create temporary project until user will save it in desired location
            std::shared_ptr<gs::management::Project> project = nullptr;
            if (params->dataset.output_path.empty()) {
                project = gs::management::CreateTempNewProject(
                    convertDatasetConfig(params->dataset),
                    convertOptimizationParams(params->optimization));
                if (!project) {
                    LOG_ERROR("Temporary project creation failed");
                    return -1;
                }
                params->dataset.output_path = project->getProjectOutputFolder();
                LOG_DEBUG("Created temporary project at: {}", params->dataset.output_path.string());
            } else {
                project = gs::management::CreateNewProject(
                    convertDatasetConfig(params->dataset),
                    convertOptimizationParams(params->optimization));
                if (!project) {
                    LOG_ERROR("Project creation failed");
                    return -1;
                }
                LOG_DEBUG("Created project at: {}", params->dataset.output_path.string());
            }
            viewer->attachProject(project);
        }

        // Set parameters
        viewer->setParameters(*params);

        // Load data if specified
        if (!params->ply_path.empty()) {
            LOG_INFO("Loading PLY file: {}", params->ply_path.string());
            auto result = viewer->loadPLY(params->ply_path);
            if (!result) {
                LOG_ERROR("Failed to load PLY: {}", result.error());
                return -1;
            }
        } else if (!params->dataset.data_path.empty()) {
            LOG_INFO("Loading dataset: {}", params->dataset.data_path.string());
            auto result = viewer->loadDataset(params->dataset.data_path);
            if (!result) {
                LOG_ERROR("Failed to load dataset: {}", result.error());
                return -1;
            }
        }

        LOG_INFO("Anti-aliasing: {}", params->optimization.antialiasing ? "enabled" : "disabled");

        // Run the viewer
        viewer->run();

        LOG_INFO("Viewer closed");
        return 0;
    }

    int Application::run(std::unique_ptr<param::TrainingParameters> params) {
        // no gui
        if (params->optimization.headless) {
            return run_headless_app(std::move(params));
        }

#ifdef WIN32
        // hide console window on windows
        HWND hwnd = GetConsoleWindow();
        Sleep(1);
        HWND owner = GetWindow(hwnd, GW_OWNER);
        DWORD dwProcessId;
        GetWindowThreadProcessId(hwnd, &dwProcessId);

        // Only hide if did not start from console
        if (GetCurrentProcessId() == dwProcessId) {
            if (owner == NULL) {
                ShowWindow(hwnd, SW_HIDE); // Windows 10
            } else {
                ShowWindow(owner, SW_HIDE); // Windows 11
            }
        }
#endif
        // gui app
        return run_gui_app(std::move(params));
    }
} // namespace lfs::core