/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/training_manager.hpp"
#include "core_new/logger.hpp"
#include "training_new/training_setup.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace lfs::vis {

    using namespace lfs::core::events;

    // Helper function to convert gs::param types to lfs::core::param types
    // Cannot use memcpy because structures contain non-trivial types (std::string, std::vector)
    static lfs::core::param::OptimizationParameters convertOptimizationParams(const gs::param::OptimizationParameters& src) {
        lfs::core::param::OptimizationParameters dst;
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

    TrainerManager::TrainerManager() {
        setupEventHandlers();
        LOG_DEBUG("TrainerManager created");
    }

    TrainerManager::~TrainerManager() {
        // Ensure training is stopped before destruction
        if (training_thread_ && training_thread_->joinable()) {
            LOG_INFO("Stopping training thread during destruction...");
            stopTraining();
            waitForCompletion();
        }
    }

    void TrainerManager::setTrainer(std::unique_ptr<lfs::training::Trainer> trainer) {
        LOG_TIMER_TRACE("TrainerManager::setTrainer");

        // Clear any existing trainer first
        clearTrainer();

        if (trainer) {
            LOG_DEBUG("Setting new trainer");
            trainer_ = std::move(trainer);
            trainer_->setProject(project_);

            if (project_) {
                trainer_->load_cameras_info();
            }

            setState(State::Ready);

            // Trainer is ready
            lfs::core::events::internal::TrainerReady{}.emit();
            LOG_INFO("Trainer ready for training");
        }
    }

    bool TrainerManager::hasTrainer() const {
        return trainer_ != nullptr;
    }

    void TrainerManager::clearTrainer() {
        LOG_DEBUG("Clearing trainer");

        lfs::core::events::cmd::StopTraining{}.emit();
        // Stop any ongoing training first
        if (isTrainingActive()) {
            LOG_INFO("Stopping active training before clearing trainer");
            stopTraining();
            waitForCompletion();
        }

        // Additional safety: ensure thread is properly stopped even if not "active"
        if (training_thread_ && training_thread_->joinable()) {
            LOG_WARN("Force stopping training thread that wasn't marked as active");
            training_thread_->request_stop();

            // Try to wait for completion with a short timeout
            auto timeout = std::chrono::milliseconds(500);
            {
                std::unique_lock<std::mutex> lock(completion_mutex_);
                if (completion_cv_.wait_for(lock, timeout, [this] { return training_complete_; })) {
                    lock.unlock();
                    LOG_DEBUG("Thread completed gracefully, joining...");
                    training_thread_->join();
                } else {
                    lock.unlock();
                    LOG_WARN("Thread didn't respond to stop request within timeout, detaching...");
                    training_thread_->detach();
                }
            }
            training_thread_.reset();
        }

        // Now safe to clear the trainer
        trainer_.reset();
        last_error_.clear();
        setState(State::Idle);

        // Reset loss buffer
        loss_buffer_.clear();
        LOG_INFO("Trainer cleared");
    }

    std::expected<bool, std::string> TrainerManager::initializeTrainerFromProject() {
        if (!trainer_) {
            return std::unexpected("No trainer available");
        }

        if (!project_) {
            return std::unexpected("No project available");
        }

        // Create training parameters from project
        lfs::core::param::TrainingParameters params;

        // Convert gs::management::DataSetInfo to lfs::core::param::DatasetConfig
        // DataSetInfo inherits from DatasetConfig, so we can slice-copy the base
        const auto& old_dataset = project_->getProjectData().data_set_info;
        params.dataset.data_path = old_dataset.data_path;
        params.dataset.output_path = project_->getProjectOutputFolder();
        params.dataset.project_path = old_dataset.project_path;
        params.dataset.images = old_dataset.images;
        params.dataset.resize_factor = old_dataset.resize_factor;
        params.dataset.test_every = old_dataset.test_every;
        params.dataset.timelapse_images = old_dataset.timelapse_images;
        params.dataset.timelapse_every = old_dataset.timelapse_every;
        params.dataset.max_width = old_dataset.max_width;

        // Convert gs::param::OptimizationParameters to lfs::core::param::OptimizationParameters
        params.optimization = convertOptimizationParams(project_->getOptimizationParams());

        // Initialize trainer
        auto init_result = trainer_->initialize(params);
        if (!init_result) {
            return std::unexpected(init_result.error());
        }

        return true;
    }

    bool TrainerManager::startTraining() {
        LOG_TIMER("TrainerManager::startTraining");

        if (!canStart()) {
            LOG_WARN("Cannot start training in current state: {}", static_cast<int>(state_.load()));
            return false;
        }

        if (!trainer_) {
            LOG_ERROR("Cannot start training - no trainer available");
            return false;
        }

        // ALWAYS reinitialize trainer to pick up any parameter changes from the project
        // This ensures that any UI changes are applied
        LOG_INFO("Initializing trainer with current project parameters");
        auto init_result = initializeTrainerFromProject();
        if (!init_result) {
            LOG_ERROR("Failed to initialize trainer: {}", init_result.error());
            last_error_ = init_result.error();
            setState(State::Error);
            return false;
        }

        // Reset completion state
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            training_complete_ = false;
        }

        setState(State::Running);

        // Emit training started event
        state::TrainingStarted{
            .total_iterations = getTotalIterations()}
            .emit();

        // Start training thread
        training_thread_ = std::make_unique<std::jthread>(
            [this](std::stop_token stop_token) {
                trainingThreadFunc(stop_token);
            });

        LOG_INFO("Training started - {} iterations planned", getTotalIterations());
        return true;
    }

    void TrainerManager::pauseTraining() {
        if (!canPause()) {
            LOG_TRACE("Cannot pause training in current state");
            return;
        }

        if (trainer_) {
            trainer_->request_pause();
            setState(State::Paused);

            state::TrainingPaused{
                .iteration = getCurrentIteration()}
                .emit();

            LOG_INFO("Training paused at iteration {}", getCurrentIteration());
        }
    }

    void TrainerManager::resumeTraining() {
        if (!canResume()) {
            LOG_TRACE("Cannot resume training in current state");
            return;
        }

        if (trainer_) {
            trainer_->request_resume();
            setState(State::Running);

            state::TrainingResumed{
                .iteration = getCurrentIteration()}
                .emit();

            LOG_INFO("Training resumed from iteration {}", getCurrentIteration());
        }
    }

    void TrainerManager::stopTraining() {
        if (!isTrainingActive()) {
            LOG_TRACE("Training not active, nothing to stop");
            return;
        }

        LOG_DEBUG("Requesting training stop");
        setState(State::Stopping);

        if (trainer_) {
            trainer_->request_stop();
        }

        if (training_thread_ && training_thread_->joinable()) {
            LOG_DEBUG("Requesting training thread to stop...");
            training_thread_->request_stop();
        }

        state::TrainingStopped{
            .iteration = getCurrentIteration(),
            .user_requested = true}
            .emit();

        LOG_INFO("Training stop requested at iteration {}", getCurrentIteration());
    }

    void TrainerManager::requestSaveCheckpoint() {
        if (trainer_ && isTrainingActive()) {
            trainer_->request_save();
            LOG_INFO("Checkpoint save requested at iteration {}", getCurrentIteration());
        } else {
            LOG_WARN("Cannot save checkpoint - training not active");
        }
    }

    bool TrainerManager::resetTraining() {
        LOG_INFO("Resetting training to initial state");

        if (!trainer_) {
            LOG_WARN("No trainer to reset");
            return false;
        }

        // Stop if active
        if (isTrainingActive()) {
            stopTraining();
            waitForCompletion();
        }

        if (trainer_->isInitialized()) {
            LOG_DEBUG("Clearing GPU memory from previous training");

            // Save params before destroying
            auto params = trainer_->getParams();

            // Destroy the trainer to release all tensors
            trainer_.reset();

            // Synchronize to ensure all GPU operations are complete
            cudaDeviceSynchronize();

            LOG_DEBUG("GPU memory released");

            // Recreate trainer
            auto setup_result = lfs::training::setupTraining(params);
            if (setup_result) {
                trainer_ = std::move(setup_result->trainer);
                trainer_->setProject(project_);
                if (project_) {
                    trainer_->load_cameras_info();
                }
            } else {
                LOG_ERROR("Failed to recreate trainer after reset: {}", setup_result.error());
                setState(State::Error);
                return false;
            }
        }

        // Clear loss buffer
        loss_buffer_.clear();

        // Set to Ready state
        setState(State::Ready);

        LOG_INFO("Training reset complete - GPU memory freed, ready to start with current parameters");
        return true;
    }

    void TrainerManager::waitForCompletion() {
        if (!training_thread_ || !training_thread_->joinable()) {
            return;
        }

        LOG_DEBUG("Waiting for training thread to complete...");

        std::unique_lock<std::mutex> lock(completion_mutex_);
        completion_cv_.wait(lock, [this] { return training_complete_; });

        training_thread_->join();
        training_thread_.reset();

        LOG_DEBUG("Training thread joined successfully");
    }

    int TrainerManager::getCurrentIteration() const {
        return trainer_ ? trainer_->get_current_iteration() : 0;
    }

    float TrainerManager::getCurrentLoss() const {
        return trainer_ ? trainer_->get_current_loss() : 0.0f;
    }

    int TrainerManager::getTotalIterations() const {
        if (!trainer_)
            return 0;
        return trainer_->getParams().optimization.iterations;
    }

    int TrainerManager::getNumSplats() const {
        if (!trainer_)
            return 0;
        return static_cast<int>(trainer_->get_strategy().get_model().size());
    }

    void TrainerManager::updateLoss(float loss) {
        std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
        loss_buffer_.push_back(loss);
        while (loss_buffer_.size() > static_cast<size_t>(max_loss_points_)) {
            loss_buffer_.pop_front();
        }
        LOG_TRACE("Loss updated: {:.6f} (buffer size: {})", loss, loss_buffer_.size());
    }

    std::deque<float> TrainerManager::getLossBuffer() const {
        std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
        return loss_buffer_;
    }

    void TrainerManager::trainingThreadFunc(std::stop_token stop_token) {
        LOG_INFO("Training thread started");
        LOG_TIMER("Training execution");

        try {
            LOG_DEBUG("Starting trainer->train() with stop token");
            auto train_result = trainer_->train(stop_token);

            if (!train_result) {
                LOG_ERROR("Training failed: {}", train_result.error());
                handleTrainingComplete(false, train_result.error());
            } else {
                LOG_INFO("Training completed successfully");
                handleTrainingComplete(true);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Exception in training thread: {}", e.what());
            handleTrainingComplete(false, std::format("Exception in training: {}", e.what()));
        } catch (...) {
            LOG_CRITICAL("Unknown exception in training thread");
            handleTrainingComplete(false, "Unknown exception in training");
        }

        LOG_INFO("Training thread finished");
    }

    void TrainerManager::setState(State new_state) {
        std::lock_guard<std::mutex> lock(state_mutex_);

        State old_state = state_.load();
        state_ = new_state;

        const char* state_str = "";
        switch (new_state) {
        case State::Idle: state_str = "Idle"; break;
        case State::Ready: state_str = "Ready"; break;
        case State::Running: state_str = "Running"; break;
        case State::Paused: state_str = "Paused"; break;
        case State::Stopping: state_str = "Stopping"; break;
        case State::Completed: state_str = "Completed"; break;
        case State::Error: state_str = "Error"; break;
        }

        LOG_DEBUG("Training state changed from {} to {}",
                  static_cast<int>(old_state), state_str);
    }

    void TrainerManager::handleTrainingComplete(bool success, const std::string& error) {
        if (!error.empty()) {
            last_error_ = error;
            LOG_ERROR("Training error: {}", error);
        }

        setState(success ? State::Completed : State::Error);

        int final_iteration = getCurrentIteration();
        float final_loss = getCurrentLoss();

        LOG_INFO("Training finished - Success: {}, Final iteration: {}, Final loss: {:.6f}",
                 success, final_iteration, final_loss);

        state::TrainingCompleted{
            .iteration = final_iteration,
            .final_loss = final_loss,
            .success = success,
            .error = error.empty() ? std::nullopt : std::optional(error)}
            .emit();

        // Notify completion
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            training_complete_ = true;
        }
        completion_cv_.notify_all();
    }

    void TrainerManager::setupEventHandlers() {
        using namespace lfs::core::events;

        // Listen for training progress events - only update loss buffer
        state::TrainingProgress::when([this](const auto& event) {
            updateLoss(event.loss);
        });
    }

    std::shared_ptr<const lfs::core::Camera> TrainerManager::getCamById(int camId) const {
        if (trainer_) {
            LOG_TRACE("Retrieving camera with ID: {}", camId);
            return trainer_->getCamById(camId);
        }
        LOG_ERROR("getCamById called but trainer is not initialized");
        return nullptr;
    }

    std::vector<std::shared_ptr<const lfs::core::Camera>> TrainerManager::getCamList() const {
        if (trainer_) {
            auto cams = trainer_->getCamList();
            LOG_TRACE("Retrieved {} cameras from trainer", cams.size());
            return cams;
        }
        LOG_ERROR("getCamList called but trainer is not initialized");
        return {};
    }

    void TrainerManager::setProject(std::shared_ptr<gs::management::Project> project) {
        project_ = project;
        if (trainer_) {
            trainer_->setProject(project);
        }
    }

} // namespace lfs::vis