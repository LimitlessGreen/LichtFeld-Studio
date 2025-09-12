/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "trainer.hpp"
#include "components/bilateral_grid.hpp"
#include "training_kernels.cuh"
#include "memory_tracker.hpp"
#include "components/poseopt.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include "components/sparsity_optimizer.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "kernels/ssim.cuh"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/rasterizer.hpp"
#include <ATen/cuda/CUDAEvent.h>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <expected>
#include <memory>

namespace gs::training {

    void Trainer::cleanup() {
        LOG_DEBUG("Cleaning up trainer for re-initialization");

        // Stop any ongoing operations
        stop_requested_ = true;

        // Wait for callback to finish if busy
        if (callback_busy_.load()) {
            callback_stream_.synchronize();
            callback_busy_.store(false);
        }

        // Reset all components
        progress_.reset();
        bilateral_grid_.reset();
        bilateral_grid_optimizer_.reset();
        bilateral_grid_scheduler_.reset();
        poseopt_module_.reset();
        poseopt_optimizer_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();

        // Clear datasets (will be recreated)
        train_dataset_.reset();
        val_dataset_.reset();

        // Clear camera cache
        m_cam_id_to_cam.clear();

        // Reset flags
        pause_requested_ = false;
        save_requested_ = false;
        stop_requested_ = false;
        is_paused_ = false;
        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false;
        current_iteration_ = 0;
        current_loss_ = 0.0f;

        LOG_DEBUG("Trainer cleanup complete");
    }

    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return {};
        }

        try {
            bilateral_grid_ = std::make_unique<BilateralGrid>(
                train_dataset_size_,
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W);

            bilateral_grid_optimizer_ = std::make_unique<torch::optim::Adam>(
                std::vector<torch::Tensor>{bilateral_grid_->parameters()},
                torch::optim::AdamOptions(params_.optimization.bilateral_grid_lr)
                    .eps(1e-15));

            // Create scheduler with warmup
            const double gamma = std::pow(0.01, 1.0 / params_.optimization.iterations);
            bilateral_grid_scheduler_ = std::make_unique<WarmupExponentialLR>(
                *bilateral_grid_optimizer_,
                gamma,
                1000, // warmup steps
                0.01, // start at 1% of initial LR
                -1    // all param groups
            );

            LOG_DEBUG("Bilateral grid initialized with size {}x{}x{} and warmup scheduler",
                      params_.optimization.bilateral_grid_X,
                      params_.optimization.bilateral_grid_Y,
                      params_.optimization.bilateral_grid_W);
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize bilateral grid: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_scale_reg_loss(
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {
        try {
            if (opt_params.scale_reg > 0.0f) {
                auto scale_l1 = splatData.get_scaling().mean();
                return opt_params.scale_reg * scale_l1;
            }
            // Return zero scalar without requires_grad
            return torch::zeros({}, torch::kFloat32);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing scale regularization loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_opacity_reg_loss(
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {
        try {
            if (opt_params.opacity_reg > 0.0f) {
                auto opacity_l1 = splatData.get_opacity().mean();
                return opt_params.opacity_reg * opacity_l1;
            }
            // Return zero scalar without requires_grad
            return torch::zeros({}, torch::kFloat32);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing opacity regularization loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_bilateral_grid_tv_loss(
        const std::unique_ptr<BilateralGrid>& bilateral_grid,
        const param::OptimizationParameters& opt_params) {
        try {
            if (opt_params.use_bilateral_grid && opt_params.tv_loss_weight > 0.0f) {
                return opt_params.tv_loss_weight * bilateral_grid->tv_loss();
            }
            // Return zero scalar without requires_grad
            return torch::zeros({}, torch::kFloat32);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing bilateral grid TV loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_sparsity_loss(
        int iter,
        const SplatData& splatData) {
        try {
            if (sparsity_optimizer_ && sparsity_optimizer_->should_apply_loss(iter)) {
                // Initialize on first use (lazy initialization)
                if (!sparsity_optimizer_->is_initialized()) {
                    auto init_result = sparsity_optimizer_->initialize(splatData.opacity_raw());
                    if (!init_result) {
                        return std::unexpected(init_result.error());
                    }
                    LOG_INFO("Sparsity optimizer initialized at iteration {}", iter);
                }

                auto loss_result = sparsity_optimizer_->compute_loss(splatData.opacity_raw());
                if (!loss_result) {
                    return std::unexpected(loss_result.error());
                }
                return *loss_result;
            }
            // Return zero scalar on CPU, then move to CUDA
            return torch::zeros({}, torch::kFloat32).to(torch::kCUDA);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing sparsity loss: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::handle_sparsity_update(
        int iter,
        SplatData& splatData) {
        try {
            if (sparsity_optimizer_ && sparsity_optimizer_->should_update(iter)) {
                LOG_TRACE("Updating sparsity state at iteration {}", iter);
                auto result = sparsity_optimizer_->update_state(splatData.opacity_raw());
                if (!result) {
                    return std::unexpected(result.error());
                }
            }
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error updating sparsity state: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::apply_sparsity_pruning(
        int iter,
        SplatData& splatData) {
        try {
            if (sparsity_optimizer_ && sparsity_optimizer_->should_prune(iter)) {
                LOG_INFO("Applying sparsity-based pruning at iteration {}", iter);

                auto mask_result = sparsity_optimizer_->get_prune_mask(splatData.opacity_raw());
                if (!mask_result) {
                    return std::unexpected(mask_result.error());
                }

                auto prune_mask = *mask_result;
                int n_prune = prune_mask.sum().item<int>();
                int n_before = splatData.size();

                // Use strategy's remove functionality
                strategy_->remove_gaussians(prune_mask);

                int n_after = splatData.size();
                std::println("Sparsity pruning complete: {} -> {} Gaussians (removed {})",
                             n_before, n_after, n_prune);

                // Clear sparsity optimizer after pruning
                sparsity_optimizer_.reset();
                LOG_DEBUG("Sparsity optimizer cleared after pruning");
            }
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error applying sparsity pruning: {}", e.what()));
        }
    }

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy)
        : base_dataset_(std::move(dataset)),
          strategy_(std::move(strategy)) {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }
        LOG_DEBUG("Trainer constructed with {} cameras", base_dataset_->get_cameras().size());
    }

    void Trainer::load_cameras_info() {
        m_cam_id_to_cam.clear();
        // Setup camera cache
        for (const auto& cam : base_dataset_->get_cameras()) {
            m_cam_id_to_cam[cam->uid()] = cam;
        }
    }

    std::expected<void, std::string> Trainer::initialize(const param::TrainingParameters& params) {
        // Thread-safe initialization using mutex
        std::lock_guard<std::mutex> lock(init_mutex_);

        // Check again after acquiring lock (double-checked locking pattern)
        if (initialized_.load()) {
            LOG_INFO("Re-initializing trainer with new parameters");
            // Clean up existing state for re-initialization
            cleanup();
        }

        LOG_INFO("Initializing trainer with {} iterations", params.optimization.iterations);

        try {
            params_ = params;

            // Handle dataset split based on evaluation flag
            if (params.optimization.enable_eval) {
                // Create train/val split
                train_dataset_ = std::make_shared<CameraDataset>(
                    base_dataset_->get_cameras(), params.dataset, CameraDataset::Split::TRAIN);
                val_dataset_ = std::make_shared<CameraDataset>(
                    base_dataset_->get_cameras(), params.dataset, CameraDataset::Split::VAL);

                LOG_INFO("Created train/val split: {} train, {} val images",
                         train_dataset_->size().value(),
                         val_dataset_->size().value());
            } else {
                // Use all images for training
                train_dataset_ = base_dataset_;
                val_dataset_ = nullptr;

                LOG_INFO("Using all {} images for training (no evaluation)",
                         train_dataset_->size().value());
            }

            // chage resize factor (change may comes from gui)
            if (train_dataset_) {
                train_dataset_->set_resize_factor(params.dataset.resize_factor);
            }
            if (val_dataset_) {
                val_dataset_->set_resize_factor(params.dataset.resize_factor);
            }

            train_dataset_size_ = train_dataset_->size().value();

            m_cam_id_to_cam.clear();
            // Setup camera cache
            for (const auto& cam : base_dataset_->get_cameras()) {
                m_cam_id_to_cam[cam->uid()] = cam;
            }
            LOG_DEBUG("Camera cache initialized with {} cameras", m_cam_id_to_cam.size());

            // Re-initialize strategy with new parameters
            strategy_->initialize(params.optimization);
            LOG_DEBUG("Strategy initialized");

            // Initialize bilateral grid if enabled
            if (auto result = initialize_bilateral_grid(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize sparsity optimizer if enabled
            if (params.optimization.enable_sparsity) {
                // Calculate when sparsity should start
                int base_iterations = params.optimization.iterations;
                int sparsity_start = base_iterations; // Start after base training
                int total_iterations = base_iterations + params.optimization.sparsify_steps;

                // Extend the total training iterations
                params_.optimization.iterations = total_iterations;

                ADMMSparsityOptimizer::Config sparsity_config{
                    .sparsify_steps = params.optimization.sparsify_steps,
                    .init_rho = params.optimization.init_rho,
                    .prune_ratio = params.optimization.prune_ratio,
                    .update_every = 50,
                    .start_iteration = sparsity_start // Start after base training completes
                };

                sparsity_optimizer_ = SparsityOptimizerFactory::create("admm", sparsity_config);

                if (sparsity_optimizer_) {
                    // Don't initialize yet - will initialize when we reach start_iteration
                    LOG_INFO("=== Sparsity Optimization Configuration ===");
                    LOG_INFO("Base training iterations: {}", base_iterations);
                    LOG_INFO("Sparsification starts at: iteration {}", sparsity_start);
                    LOG_INFO("Sparsification duration: {} iterations", params.optimization.sparsify_steps);
                    LOG_INFO("Total training iterations: {}", total_iterations);
                    LOG_INFO("Pruning ratio: {}%", params.optimization.prune_ratio * 100);
                    LOG_INFO("ADMM penalty (rho): {}", params.optimization.init_rho);
                }
            }

            if (!cuda_memory_) {
                cuda_memory_ = std::make_unique<TrainingMemory>();
                // We'll initialize with actual dimensions on first use
            }

            // Set default background
            if (cuda_memory_->allocated()) {
                cuda_memory_->set_background(0.0f, 0.0f, 0.0f);
            }

            if (params.optimization.pose_optimization != "none") {
                if (params.optimization.enable_eval) {
                    return std::unexpected("Evaluating with pose optimization is not supported yet. "
                                           "Please disable pose optimization or evaluation.");
                }
                if (params.optimization.gut) {
                    return std::unexpected("The 3DGUT rasterizer doesn't have camera gradients yet. "
                                           "Please disable pose optimization or disable gut.");
                }
                if (params.optimization.pose_optimization == "direct") {
                    poseopt_module_ = std::make_unique<DirectPoseOptimizationModule>(train_dataset_->get_cameras().size());
                    LOG_DEBUG("Direct pose optimization module created");
                } else if (params.optimization.pose_optimization == "mlp") {
                    poseopt_module_ = std::make_unique<MLPPoseOptimizationModule>(train_dataset_->get_cameras().size());
                    LOG_DEBUG("MLP pose optimization module created");
                } else {
                    return std::unexpected("Invalid pose optimization type: " + params.optimization.pose_optimization);
                }
                poseopt_optimizer_ = std::make_unique<torch::optim::Adam>(
                    std::vector<torch::Tensor>{poseopt_module_->parameters()},
                    torch::optim::AdamOptions(1e-5));
            } else {
                poseopt_module_ = std::make_unique<PoseOptimizationModule>();
            }

            // Create progress bar based on headless flag
            if (params.optimization.headless) {
                progress_ = std::make_unique<TrainingProgress>(
                    params_.optimization.iterations, // This now includes sparsity steps if enabled
                    /*update_frequency=*/100);
                LOG_DEBUG("Progress bar initialized for {} total iterations", params_.optimization.iterations);
            }

            // Initialize the evaluator - it handles all metrics internally
            evaluator_ = std::make_unique<MetricsEvaluator>(params_);
            LOG_DEBUG("Metrics evaluator initialized");

            // Print configuration
            LOG_INFO("Render mode: {}", params.optimization.render_mode);
            LOG_INFO("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
            LOG_INFO("Strategy: {}", params.optimization.strategy);

            initialized_ = true;
            LOG_INFO("Trainer initialization complete");
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize trainer: {}", e.what()));
        }
    }

    Trainer::~Trainer() {
        // Ensure training is stopped
        stop_requested_ = true;

        // Wait for callback to finish if busy
        if (callback_busy_.load()) {
            callback_stream_.synchronize();
        }
        LOG_DEBUG("Trainer destroyed");
    }

    void Trainer::handle_control_requests(int iter, std::stop_token stop_token) {
        // Check stop token first
        if (stop_token.stop_requested()) {
            stop_requested_ = true;
            return;
        }

        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            if (progress_) {
                progress_->pause();
            }
            LOG_INFO("Training paused at iteration {}", iter);
            LOG_DEBUG("Click 'Resume Training' to continue.");
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            if (progress_) {
                progress_->resume(iter, current_loss_.load(), static_cast<int>(strategy_->get_model().size()));
            }
            LOG_INFO("Training resumed at iteration {}", iter);
        }

        // Handle save request
        if (save_requested_.exchange(false)) {
            LOG_INFO("Saving checkpoint at iteration {}...", iter);
            auto checkpoint_path = params_.dataset.output_path / "checkpoints";
            save_ply(checkpoint_path, iter, /*join=*/true);

            LOG_INFO("Checkpoint saved to {}", checkpoint_path.string());

            // Emit checkpoint saved event
            events::state::CheckpointSaved{
                .iteration = iter,
                .path = checkpoint_path}
                .emit();
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            LOG_INFO("Stopping training permanently at iteration {}...", iter);
            LOG_DEBUG("Saving final model...");
            save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    // Replace the sine_background_for_step and background_for_step functions in trainer.cpp

    inline float inv_weight_piecewise(int step, int max_steps) {
        // Keep this function as-is, it's just CPU computation
        const float phase = std::max(0.f, std::min(1.f, step / float(std::max(1, max_steps))));
        const float limit_hi = 1.0f / 4.0f;
        const float limit_mid = 2.0f / 4.0f;
        const float limit_lo = 3.0f / 4.0f;
        const float weight_hi = 1.0f;
        const float weight_mid = 0.5f;
        const float weight_lo = 0.0f;

        if (phase < limit_hi) {
            return weight_hi;
        } else if (phase < limit_mid) {
            const float t = (phase - limit_hi) / (limit_mid - limit_hi);
            return weight_hi + (weight_mid - weight_hi) * t;
        } else {
            const float t = (phase - limit_mid) / (limit_lo - limit_mid);
            return weight_mid + (weight_lo - weight_mid) * t;
        }
    }


    float* Trainer::background_for_step_raw(int iter) {
        const auto& opt = params_.optimization;

        if (!opt.bg_modulation) {
            return cuda_memory_->background();
        }

        const float w_mix = inv_weight_piecewise(iter, opt.iterations);
        if (w_mix <= 0.0f) {
            return cuda_memory_->background();
        }

        // Generate sine background in a temp buffer first
        launch_compute_sine_background(
            cuda_memory_->bg_mix(),  // Use bg_mix as temp storage
            iter,
            37, 41, 43,
            0.03f,
            0
        );

        // Now blend base with sine
        launch_background_blend(
            cuda_memory_->bg_mix(),
            cuda_memory_->background(),
            cuda_memory_->bg_mix(),  // sine is already here
            w_mix,
            0
        );

        cudaDeviceSynchronize();  // Ensure kernel completes
        return cuda_memory_->bg_mix();
    }

    torch::Tensor& Trainer::background_for_step(int iter) {
        // For now, create a tensor wrapper around the raw background
        float* bg_raw = background_for_step_raw(iter);

        // Create a tensor that wraps the raw memory (no copy)
        background_wrapper_ = torch::from_blob(
            bg_raw,
            {3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
        );

        return background_wrapper_;
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
    int iter,
    Camera* cam,
    torch::Tensor gt_image,
    RenderMode render_mode,
    std::stop_token stop_token) {
    try {
        // Memory tracking for critical iterations
        bool track_memory = (iter <= 10) || (iter % 100 == 0) || (iter % 1000 == 0);

        if (track_memory) {
            auto step_start = MemoryTracker::get().capture(iter, "step_start");
            if (iter <= 5) {
                MemoryTracker::get().log_snapshot(step_start);
            }
        }

        // Camera model checks
        if (params_.optimization.gut) {
            if (cam->camera_model_type() == gsplat::CameraModelType::ORTHO) {
                return std::unexpected("Training on cameras with ortho model is not supported yet.");
            }
        } else {
            if (!params_.optimization.rc) {
                if (cam->radial_distortion().numel() != 0 ||
                    cam->tangential_distortion().numel() != 0) {
                    return std::unexpected("You must use --gut option to train on cameras with distortion.");
                }
                if (cam->camera_model_type() != gsplat::CameraModelType::PINHOLE) {
                    return std::unexpected("You must use --gut option to train on cameras with non-pinhole model.");
                }
            }
        }

        current_iteration_ = iter;

        // Check control requests at the beginning
        handle_control_requests(iter, stop_token);

        if (stop_requested_.load() || stop_token.stop_requested()) {
            return StepResult::Stop;
        }

        // Handle pause
        while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            handle_control_requests(iter, stop_token);
        }

        if (stop_requested_.load() || stop_token.stop_requested()) {
            return StepResult::Stop;
        }

        // Sparsity phase logging
        if (params_.optimization.enable_sparsity) {
            int base_iterations = params_.optimization.iterations - params_.optimization.sparsify_steps;
            if (iter == base_iterations + 1) {
                LOG_INFO("=== Entering Sparsification Phase ===");
                LOG_INFO("Base training complete at iteration {}", base_iterations);
                LOG_INFO("Starting ADMM sparsification for {} iterations",
                         params_.optimization.sparsify_steps);
                LOG_INFO("Current model size: {} Gaussians", strategy_->get_model().size());
                LOG_INFO("Target pruning: {}% of Gaussians", params_.optimization.prune_ratio * 100);
            }
            if (iter == params_.optimization.iterations - 100) {
                LOG_INFO("Approaching final pruning in 100 iterations (at iteration {})",
                         params_.optimization.iterations);
            }
            if (iter == params_.optimization.iterations - 1) {
                LOG_INFO("Final pruning will occur next iteration");
            }
        }

        auto adjusted_cam_pos = poseopt_module_->forward(cam->world_view_transform(), torch::tensor({cam->uid()}));
        auto adjusted_cam = Camera(*cam, adjusted_cam_pos);

        // Initialize CUDA memory on first use or when dimensions change
        const int width = adjusted_cam.image_width();
        const int height = adjusted_cam.image_height();

        if (!cuda_memory_) {
            cuda_memory_ = std::make_unique<TrainingMemory>();
            cuda_memory_->initialize(width, height, 3, 1); // batch=1
            cuda_memory_->set_background(0.0f, 0.0f, 0.0f);
            LOG_DEBUG("Initialized CUDA memory for {}x{} images", width, height);
        } else {
            cuda_memory_->ensure_size(width, height, 3, 1); // batch=1
        }

        // Get background using our new system
        torch::Tensor& bg = background_for_step(iter);

        // Track memory before forward pass
        if (track_memory) {
            auto pre_forward = MemoryTracker::get().capture(iter, "pre_forward");
            if (iter <= 5) {
                LOG_DEBUG("[Memory] Pre-forward - Allocated: {:.2f}MB",
                         pre_forward.cuda_allocated_bytes / (1024.0 * 1024.0));
            }
        }

        // Forward pass
        RenderOutput r_output;
        {
            ScopedMemoryTracker forward_tracker(iter, "forward_pass", track_memory);
            if (!params_.optimization.gut) {
                r_output = fast_rasterize(adjusted_cam, strategy_->get_model(), bg);
            } else {
                r_output = rasterize(adjusted_cam, strategy_->get_model(), bg, 1.0f, false, false, render_mode,
                                    nullptr);
            }
        }

        // Apply bilateral grid if enabled
        torch::Tensor processed_image = r_output.image;
        if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
            ScopedMemoryTracker bilateral_tracker(iter, "bilateral_grid", track_memory);
            processed_image = bilateral_grid_->apply(r_output.image, cam->uid());
        }

        // Track memory after forward pass
        if (track_memory) {
            auto post_forward = MemoryTracker::get().capture(iter, "post_forward");
            if (iter <= 5 || iter % 1000 == 0) {
                LOG_INFO("[Memory] Iter {} post-forward - Allocated: {:.2f}MB",
                        iter,
                        post_forward.cuda_allocated_bytes / (1024.0 * 1024.0));
            }
        }

        // ============= LOSS COMPUTATION - Using CUDA memory =============
        float total_loss_value = 0.0f;

        // Variables declared here for scope throughout function
        const int batch = 1;
        const int channels = 3;
        const int h = height;
        const int w = width;
        const size_t image_elements = batch * channels * h * w;

        {
            ScopedMemoryTracker loss_tracker(iter, "loss_computation", track_memory);

            // Ensure CUDA memory is sized for batch dimension
            cuda_memory_->ensure_size(w, h, channels, batch);

            // Prepare tensors in 4D format [B, C, H, W]
            torch::Tensor rendered_4d = processed_image;
            torch::Tensor gt_4d = gt_image;

            if (processed_image.dim() == 3) {
                rendered_4d = processed_image.unsqueeze(0);  // [C, H, W] -> [1, C, H, W]
            }
            if (gt_image.dim() == 3) {
                gt_4d = gt_image.unsqueeze(0);  // [C, H, W] -> [1, C, H, W]
            }

            rendered_4d = rendered_4d.contiguous();
            gt_4d = gt_4d.contiguous();

            // Get raw pointers from 4D tensors
            const float* rendered_ptr = rendered_4d.data_ptr<float>();
            const float* gt_ptr = gt_4d.data_ptr<float>();

            // Compute L1 loss using CUDA kernels
            launch_compute_l1_loss_forward(
                rendered_ptr,
                gt_ptr,
                cuda_memory_->diff_buffer(),
                cuda_memory_->abs_diff_buffer(),
                cuda_memory_->l1_grad(),
                image_elements,
                0
            );

            // Compute mean for L1 loss value
            launch_compute_mean(
                cuda_memory_->abs_diff_buffer(),
                cuda_memory_->loss_value(),
                image_elements,
                0
            );
            cudaDeviceSynchronize();
            float l1_loss_value = cuda_memory_->get_loss_value();

            // SSIM loss - now using raw pointers
            constexpr float kC1 = 0.01f * 0.01f;
            constexpr float kC2 = 0.03f * 0.03f;
            constexpr int crop_border = 5;  // Crop 5 pixels from each side

            // Call raw SSIM forward
            launch_ssim_forward(
                rendered_ptr,
                gt_ptr,
                cuda_memory_->ssim_map(),
                cuda_memory_->ssim_dm_dmu1(),
                cuda_memory_->ssim_dm_dsigma1_sq(),
                cuda_memory_->ssim_dm_dsigma12(),
                batch, channels, h, w,
                kC1, kC2,
                0  // stream
            );

            // Compute mean of SSIM map with cropping
            launch_ssim_reduce_mean(
                cuda_memory_->ssim_map(),
                cuda_memory_->ssim_loss_value(),
                batch, channels, h, w,
                (h > 10 && w > 10) ? crop_border : 0,  // Only crop if image is large enough
                0  // stream
            );
            cudaDeviceSynchronize();
            float ssim_val = cuda_memory_->get_ssim_loss_value();
            float ssim_loss_value = 1.0f - ssim_val;

            // Prepare gradient for SSIM backward
            int valid_elements = batch * channels * h * w;
            if (h > 10 && w > 10) {
                // With cropping
                int cropped_h = h - 2 * crop_border;
                int cropped_w = w - 2 * crop_border;
                valid_elements = batch * channels * cropped_h * cropped_w;
            }
            float grad_val = -1.0f / static_cast<float>(valid_elements);

            // Fill gradient map with appropriate values
            launch_ssim_fill_gradient(
                cuda_memory_->ssim_dL_dmap(),
                grad_val,
                batch, channels, h, w,
                (h > 10 && w > 10) ? crop_border : 0,
                0  // stream
            );

            // Call raw SSIM backward
            launch_ssim_backward(
                rendered_ptr,
                gt_ptr,
                cuda_memory_->ssim_dL_dmap(),
                cuda_memory_->ssim_dm_dmu1(),
                cuda_memory_->ssim_dm_dsigma1_sq(),
                cuda_memory_->ssim_dm_dsigma12(),
                cuda_memory_->ssim_grad(),
                batch, channels, h, w,
                kC1, kC2,
                0  // stream
            );

            // Combine losses
            float l1_weight = 1.0f - params_.optimization.lambda_dssim;
            float ssim_weight = params_.optimization.lambda_dssim;

            total_loss_value = l1_weight * l1_loss_value + ssim_weight * ssim_loss_value;

            // Combine gradients using CUDA kernel
            launch_combine_gradients(
                cuda_memory_->grad_image(),
                cuda_memory_->l1_grad(),
                l1_weight,
                cuda_memory_->ssim_grad(),
                ssim_weight,
                image_elements,
                0
            );
        }

        current_loss_ = total_loss_value;

        // Handle bilateral grid gradient if needed
        float* final_grad_image_ptr = cuda_memory_->grad_image();

        if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
            ScopedMemoryTracker bilateral_backward_tracker(iter, "bilateral_backward", track_memory);

            // Create minimal tensor wrappers for bilateral grid
            torch::Tensor grad_image_tensor = torch::from_blob(
                cuda_memory_->grad_image(),
                {batch, channels, h, w},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            );

            // Match dimensions with processed_image
            if (processed_image.dim() == 3) {
                grad_image_tensor = grad_image_tensor.squeeze(0);
            }

            auto bilateral_input = r_output.image.detach().requires_grad_(true);
            auto bilateral_output = bilateral_grid_->apply(bilateral_input, cam->uid());

            if (params_.optimization.tv_loss_weight > 0.0f) {
                auto tv_loss = params_.optimization.tv_loss_weight * bilateral_grid_->tv_loss();
                total_loss_value += tv_loss.item<float>();
                tv_loss.backward(torch::ones_like(tv_loss), true);
            }

            auto bilateral_grads = torch::autograd::grad(
                {bilateral_output},
                {bilateral_input},
                {grad_image_tensor},
                /*retain_graph=*/false,
                /*create_graph=*/false,
                /*allow_unused=*/false);

            // Copy result back to our buffer if needed
            if (bilateral_grads[0].data_ptr<float>() != final_grad_image_ptr) {
                // Adjust for potential dimension mismatch
                size_t copy_elements = bilateral_grads[0].numel();
                cudaMemcpy(final_grad_image_ptr,
                          bilateral_grads[0].data_ptr<float>(),
                          copy_elements * sizeof(float),
                          cudaMemcpyDeviceToDevice);
            }
        }

        current_loss_ = total_loss_value;

        // Create zero gradient for alpha using CUDA memory
        launch_zero_tensor(cuda_memory_->grad_alpha(), batch * height * width, 0);

        // Track memory before backward pass
        if (track_memory) {
            auto pre_backward = MemoryTracker::get().capture(iter, "pre_backward");
            if (iter <= 5 || iter % 1000 == 0) {
                LOG_INFO("[Memory] Iter {} pre-backward - Allocated: {:.2f}MB",
                        iter,
                        pre_backward.cuda_allocated_bytes / (1024.0 * 1024.0));
            }
        }

        // ============= BACKWARD PASS =============
        {
            ScopedMemoryTracker backward_tracker(iter, "backward_pass", track_memory);
            torch::NoGradGuard no_grad;

            strategy_->get_model().ensure_grad_allocated();

            // Create minimal tensor wrappers just for the backward call
            torch::Tensor grad_image_tensor = torch::from_blob(
                final_grad_image_ptr,
                processed_image.sizes(),  // Use the actual processed_image dimensions
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            );

            torch::Tensor grad_alpha_tensor = torch::from_blob(
                cuda_memory_->grad_alpha(),
                r_output.alpha.sizes(),
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
            );

            // Call backward with direct gradient writing
            if (!params_.optimization.gut) {
                fast_rasterize_backward(grad_image_tensor, grad_alpha_tensor, r_output,
                                      strategy_->get_model(), adjusted_cam);
            }

            // Debug gradient statistics
            if (iter % 100 == 0) {
                auto means_grad = strategy_->get_model().means().grad();
                if (means_grad.defined()) {
                    LOG_DEBUG("Iter {} - Means grad: min={}, max={}, mean={}",
                             iter,
                             means_grad.min().item<float>(),
                             means_grad.max().item<float>(),
                             means_grad.mean().item<float>());
                }
            }

            // Use CUDA kernels for regularization
            if (params_.optimization.scale_reg > 0.0f) {
                try {
                    auto scale_grad = strategy_->get_model().scaling_raw().mutable_grad();
                    auto scaling_raw = strategy_->get_model().scaling_raw();

                    launch_add_scale_regularization(
                        scale_grad.data_ptr<float>(),
                        scaling_raw.data_ptr<float>(),
                        params_.optimization.scale_reg,
                        scaling_raw.numel(),
                        0
                    );
                } catch (const std::exception& e) {
                    LOG_ERROR("Error computing scale regularization: {}", e.what());
                }
            }

            if (params_.optimization.opacity_reg > 0.0f) {
                try {
                    auto opacity_grad = strategy_->get_model().opacity_raw().mutable_grad();
                    auto opacity_raw = strategy_->get_model().opacity_raw();

                    launch_add_opacity_regularization(
                        opacity_grad.data_ptr<float>(),
                        opacity_raw.data_ptr<float>(),
                        params_.optimization.opacity_reg,
                        opacity_raw.size(0),
                        0
                    );
                } catch (const std::exception& e) {
                    LOG_ERROR("Error computing opacity regularization: {}", e.what());
                }
            }

            // Add sparsity gradients if applicable
            if (sparsity_optimizer_ && sparsity_optimizer_->should_apply_loss(iter)) {
                if (!sparsity_optimizer_->is_initialized()) {
                    auto init_result = sparsity_optimizer_->initialize(strategy_->get_model().opacity_raw());
                    if (!init_result) {
                        return std::unexpected(init_result.error());
                    }
                }

                auto sparsity_loss_result = sparsity_optimizer_->compute_loss(strategy_->get_model().opacity_raw());
                if (sparsity_loss_result) {
                    total_loss_value += sparsity_loss_result->item<float>();
                }
            }
        }

        // Track memory after backward pass
        if (track_memory) {
            auto post_backward = MemoryTracker::get().capture(iter, "post_backward");
            if (iter <= 5 || iter % 1000 == 0) {
                LOG_INFO("[Memory] Iter {} post-backward - Allocated: {:.2f}MB, Reserved: {:.2f}MB",
                        iter,
                        post_backward.cuda_allocated_bytes / (1024.0 * 1024.0),
                        post_backward.cuda_reserved_bytes / (1024.0 * 1024.0));
            }
        }

        current_loss_ = total_loss_value;

        // Update progress
        if (progress_) {
            progress_->update(iter, current_loss_.load(),
                              static_cast<int>(strategy_->get_model().size()),
                              strategy_->is_refining(iter));
        }

        // Emit training progress event
        if (iter % 10 == 0 || iter == 1) {
            events::state::TrainingProgress{
                .iteration = iter,
                .loss = current_loss_.load(),
                .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                .is_refining = strategy_->is_refining(iter)}
                .emit();
        }

        // Optimizer step and model updates
        {
            ScopedMemoryTracker optimizer_tracker(iter, "optimizer_step", track_memory);
            torch::NoGradGuard no_grad;

            DeferredEvents deferred;
            {
                std::unique_lock<std::shared_mutex> lock(render_mutex_);

                if (params_.optimization.enable_sparsity) {
                    int base_iterations = params_.optimization.iterations - params_.optimization.sparsify_steps;
                    if (iter <= base_iterations) {
                        strategy_->post_backward(iter, r_output);
                    }
                } else {
                    strategy_->post_backward(iter, r_output);
                    if (iter % 100 == 0)
                        c10::cuda::CUDACachingAllocator::emptyCache();
                }

                strategy_->step(iter);

                if (params_.optimization.use_bilateral_grid) {
                    bilateral_grid_optimizer_->step();
                    bilateral_grid_optimizer_->zero_grad(true);
                    bilateral_grid_scheduler_->step();
                }
                if (params_.optimization.pose_optimization != "none") {
                    poseopt_optimizer_->step();
                    poseopt_optimizer_->zero_grad(true);
                }

                deferred.add(events::state::ModelUpdated{
                    .iteration = iter,
                    .num_gaussians = static_cast<size_t>(strategy_->get_model().size())});
            }

            if (auto result = handle_sparsity_update(iter, strategy_->get_model()); !result) {
                LOG_ERROR("Sparsity update failed: {}", result.error());
            }

            if (auto result = apply_sparsity_pruning(iter, strategy_->get_model()); !result) {
                LOG_ERROR("Sparsity pruning failed: {}", result.error());
            }

            if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                ScopedMemoryTracker eval_tracker(iter, "evaluation", true);
                evaluator_->print_evaluation_header(iter);

                // Create tensor wrapper for evaluator
                torch::Tensor bg_eval = torch::from_blob(
                    cuda_memory_->background(),
                    {3},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
                );

                auto metrics = evaluator_->evaluate(iter,
                                                    strategy_->get_model(),
                                                    val_dataset_,
                                                    bg_eval);
                LOG_INFO("{}", metrics.to_string());
            }

            if (!params_.optimization.skip_intermediate_saving) {
                for (size_t save_step : params_.optimization.save_steps) {
                    if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                        ScopedMemoryTracker save_tracker(iter, "checkpoint_save", true);
                        const bool join_threads = (iter == params_.optimization.save_steps.back());
                        auto save_path = params_.dataset.output_path;
                        save_ply(save_path, iter, join_threads);
                        events::state::CheckpointSaved{
                            .iteration = iter,
                            .path = save_path}
                            .emit();
                    }
                }
            }

            if (!params_.dataset.timelapse_images.empty() && iter % params_.dataset.timelapse_every == 0) {
                ScopedMemoryTracker timelapse_tracker(iter, "timelapse", true);
                for (const auto& img_name : params_.dataset.timelapse_images) {
                    auto train_cam = train_dataset_->get_camera_by_filename(img_name);
                    auto val_cam = val_dataset_ ? val_dataset_->get_camera_by_filename(img_name) : std::nullopt;
                    if (train_cam.has_value() || val_cam.has_value()) {
                        Camera* cam_to_use = train_cam.has_value() ? train_cam.value() : val_cam.value();

                        if (cam_to_use->camera_height() == cam_to_use->image_height() && params_.dataset.resize_factor != 1) {
                            cam_to_use->load_image_size(params_.dataset.resize_factor);
                        }

                        // Create background tensor for timelapse
                        torch::Tensor bg_timelapse = torch::from_blob(
                            cuda_memory_->background(),
                            {3},
                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
                        );

                        RenderOutput rendered_timelapse_output = fast_rasterize(
                            *cam_to_use, strategy_->get_model(), bg_timelapse);

                        std::string folder_name = img_name;
                        auto last_dot = folder_name.find_last_of('.');
                        if (last_dot != std::string::npos) {
                            folder_name = folder_name.substr(0, last_dot);
                        }

                        auto output_path = params_.dataset.output_path / "timelapse" / folder_name;
                        std::filesystem::create_directories(output_path);

                        image_io::save_image_async(output_path / std::format("{:06d}.jpg", iter),
                                                   rendered_timelapse_output.image);
                    } else {
                        LOG_WARN("Timelapse image '{}' not found in dataset.", img_name);
                    }
                }
            }
        }

        // Final memory tracking
        if (track_memory) {
            auto step_end = MemoryTracker::get().capture(iter, "step_end");
            if (iter <= 5 || iter % 1000 == 0) {
                LOG_INFO("[Memory] Iter {} complete - Allocated: {:.2f}MB, Reserved: {:.2f}MB",
                        iter,
                        step_end.cuda_allocated_bytes / (1024.0 * 1024.0),
                        step_end.cuda_reserved_bytes / (1024.0 * 1024.0));
            }
        }

        if (iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()) {
            return StepResult::Continue;
        } else {
            return StepResult::Stop;
        }
    } catch (const std::exception& e) {
        return std::unexpected(std::format("Training step failed: {}", e.what()));
    }
}


    std::expected<void, std::string> Trainer::train(std::stop_token stop_token) {
    // Check if initialized
    if (!initialized_.load()) {
        return std::unexpected("Trainer not initialized. Call initialize() before train()");
    }

    is_running_ = false;
    training_complete_ = false;
    ready_to_start_ = false; // Reset the flag

    // Event-based ready signaling
    if (!params_.optimization.headless) {
        // Subscribe to start signal (no need to store handle)
        events::internal::TrainingReadyToStart::when([this](const auto&) {
            ready_to_start_ = true;
        });

        // Signal we're ready
        events::internal::TrainerReady{}.emit();

        // Wait for start signal
        LOG_DEBUG("Waiting for start signal from GUI...");
        while (!ready_to_start_.load() && !stop_token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    is_running_ = true; // Now we can start
    LOG_INFO("Starting training loop with {} workers", params_.optimization.num_workers);

    try {
        int iter = 1;
        const int num_workers = params_.optimization.num_workers;
        const RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);

        if (progress_) {
            progress_->update(iter, current_loss_.load(),
                              static_cast<int>(strategy_->get_model().size()),
                              strategy_->is_refining(iter));
        }

        // Use infinite dataloader to avoid epoch restarts - WORKS EXACTLY THE SAME!
        auto train_dataloader = create_infinite_dataloader_from_dataset(train_dataset_, num_workers);
        auto loader = train_dataloader->begin();

        LOG_DEBUG("Starting training iterations");
        // Single loop without epochs
        while (iter <= params_.optimization.iterations) {
            if (stop_token.stop_requested() || stop_requested_.load()) {
                break;
            }

            // Wait for previous callback if still running
            if (callback_busy_.load()) {
                callback_stream_.synchronize();
            }

            // Get batch - EXACTLY THE SAME!
            auto& batch = *loader;
            auto camera_with_image = batch[0].data;

            // Camera is now torch-free internally, but interface is the same
            Camera* cam = camera_with_image.camera;
            torch::Tensor gt_image = std::move(camera_with_image.image).to(torch::kCUDA, /*non_blocking=*/true);

            // train_step will use cam's raw pointers internally via fast_rasterize
            // cam->world_view_transform_cuda_ptr() and cam->cam_position_cuda_ptr()
            auto step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
            if (!step_result) {
                return std::unexpected(step_result.error());
            }

            if (*step_result == StepResult::Stop) {
                break;
            }

            // Launch callback for async progress update (except first iteration)
            if (iter > 1 && callback_) {
                callback_busy_ = true;
                auto err = cudaLaunchHostFunc(
                    callback_stream_.stream(),
                    [](void* self) {
                        auto* trainer = static_cast<Trainer*>(self);
                        if (trainer->callback_) {
                            trainer->callback_();
                        }
                        trainer->callback_busy_ = false;
                    },
                    this);
                if (err != cudaSuccess) {
                    LOG_WARN("Failed to launch callback: {}", cudaGetErrorString(err));
                    callback_busy_ = false;
                }
            }

            ++iter;
            ++loader;
        }

        // Ensure callback is finished before final save
        if (callback_busy_.load()) {
            callback_stream_.synchronize();
        }

        // Final save if not already saved by stop request
        if (!stop_requested_.load() && !stop_token.stop_requested()) {
            auto final_path = params_.dataset.output_path;
            save_ply(final_path, params_.optimization.iterations, /*join=*/true);
            // Emit final checkpoint saved event
            events::state::CheckpointSaved{
                static_cast<int>(params_.optimization.iterations),
                final_path}
                .emit();
        }

        if (progress_) {
            progress_->complete();
        }
        evaluator_->save_report();
        if (progress_) {
            progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
        }

        is_running_ = false;
        training_complete_ = true;

        LOG_INFO("Training completed successfully");
        return {};
    } catch (const std::exception& e) {
        is_running_ = false;
        return std::unexpected(std::format("Training failed: {}", e.what()));
    }
}

    std::shared_ptr<const Camera> Trainer::getCamById(int camId) const {
        const auto it = m_cam_id_to_cam.find(camId);
        if (it == m_cam_id_to_cam.end()) {
            LOG_ERROR("getCamById - could not find cam with cam id {}", camId);
            return nullptr;
        }
        return it->second;
    }

    std::vector<std::shared_ptr<const Camera>> Trainer::getCamList() const {
        std::vector<std::shared_ptr<const Camera>> cams;
        cams.reserve(m_cam_id_to_cam.size());
        for (auto& [key, value] : m_cam_id_to_cam) {
            cams.push_back(value);
        }

        return cams;
    }

    void Trainer::save_ply(const std::filesystem::path& save_path, int iter_num, bool join_threads) {
        // Save PLY format - join_threads controls sync vs async
        strategy_->get_model().save_ply(save_path, iter_num, join_threads);

        // Save SOG format if requested - ALWAYS synchronous
        if (params_.optimization.save_sog) {
            strategy_->get_model().save_sog(save_path, iter_num,
                                            params_.optimization.sog_iterations,
                                            true); // Always synchronous
        }

        // Update project with PLY info
        if (lf_project_) {
            const std::string ply_name = "splat_" + std::to_string(iter_num);
            const std::filesystem::path ply_path = save_path / (ply_name + ".ply");
            lf_project_->addPly(gs::management::PlyData(false, ply_path, iter_num, ply_name));
        }

        LOG_DEBUG("PLY save initiated: {} (sync={}), SOG always sync", save_path.string(), join_threads);
    }
} // namespace gs::training