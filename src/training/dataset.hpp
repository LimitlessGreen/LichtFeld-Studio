/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "loader/loader.hpp"
#include <expected>
#include <format>
#include <memory>
#include <random>
#include <torch/torch.h>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>

namespace gs::training {
    struct CameraWithImage {
        Camera* camera;
        torch::Tensor image;

        // Add raw pointer accessors for fast access
        const float* world_view_transform_ptr() const {
            return camera->world_view_transform_cuda_ptr();
        }

        const float* cam_position_ptr() const {
            return camera->cam_position_cuda_ptr();
        }
    };

    // Forward declarations
    class InfiniteRandomSampler;
    class RandomSampler;

    class CameraDataset {
    public:
        enum class Split {
            TRAIN,
            VAL,
            ALL
        };

        CameraDataset(std::vector<std::shared_ptr<Camera>> cameras,
                      const gs::param::DatasetConfig& params,
                      Split split = Split::ALL)
            : _cameras(std::move(cameras)),
              _datasetConfig(params),
              _split(split) {
            // Create indices based on split
            _indices.clear();
            for (size_t i = 0; i < _cameras.size(); ++i) {
                const bool is_test = (i % params.test_every) == 0;

                if (_split == Split::ALL ||
                    (_split == Split::TRAIN && !is_test) ||
                    (_split == Split::VAL && is_test)) {
                    _indices.push_back(i);
                }
            }

            std::cout << "Dataset created with " << _indices.size()
                      << " images (split: " << static_cast<int>(_split) << ")" << std::endl;
        }

        // Default copy constructor works with shared_ptr
        CameraDataset(const CameraDataset&) = default;
        CameraDataset(CameraDataset&&) noexcept = default;
        CameraDataset& operator=(CameraDataset&&) noexcept = default;
        CameraDataset& operator=(const CameraDataset&) = default;

        CameraWithImage get(size_t index) {
            if (index >= _indices.size()) {
                throw std::out_of_range("Dataset index out of range");
            }

            size_t camera_idx = _indices[index];
            auto& cam = _cameras[camera_idx];

            // Load image - returns a torch tensor but Camera internally doesn't use torch
            torch::Tensor image = cam->load_and_get_image(_datasetConfig.resize_factor);

            // Return camera pointer and image
            return {cam.get(), std::move(image)};
        }

        size_t size() const {
            return _indices.size();
        }

        const std::vector<std::shared_ptr<Camera>>& get_cameras() const {
            return _cameras;
        }

        Split get_split() const { return _split; }

        size_t get_num_bytes() const {
            if (_cameras.empty()) {
                return 0;
            }
            size_t total_bytes = 0;
            for (const auto& cam : _cameras) {
                total_bytes += cam->get_num_bytes_from_file();
            }
            // Adjust for resolution factor if specified
            if (_datasetConfig.resize_factor > 0) {
                total_bytes /= _datasetConfig.resize_factor * _datasetConfig.resize_factor;
            }
            return total_bytes;
        }

        [[nodiscard]] std::optional<Camera*> get_camera_by_filename(const std::string& filename) const {
            for (const auto& cam : _cameras) {
                if (cam->image_name() == filename) {
                    return cam.get();
                }
            }
            return std::nullopt;
        }

        void set_resize_factor(int resize_factor) {
            _datasetConfig.resize_factor = resize_factor;
        }

    private:
        std::vector<std::shared_ptr<Camera>> _cameras;
        gs::param::DatasetConfig _datasetConfig;
        Split _split;
        std::vector<size_t> _indices;
    };

    // Regular random sampler (non-infinite)
    class RandomSampler {
    public:
        explicit RandomSampler(size_t dataset_size)
            : size_(dataset_size),
              current_position_(0),
              gen_(std::random_device{}()) {

            if (size_ == 0) {
                throw std::invalid_argument("Dataset size must be positive");
            }

            // Generate initial permutation
            generate_permutation();
        }

        // Get next batch of indices (returns nullopt when epoch is done)
        std::optional<std::vector<size_t>> next(size_t batch_size) {
            if (current_position_ >= indices_.size()) {
                return std::nullopt;  // End of epoch
            }

            std::vector<size_t> batch;
            batch.reserve(batch_size);

            for (size_t i = 0; i < batch_size && current_position_ < indices_.size(); ++i) {
                batch.push_back(indices_[current_position_]);
                current_position_++;
            }

            return batch;
        }

        void reset() {
            generate_permutation();
            current_position_ = 0;
        }

        size_t size() const { return size_; }

    private:
        void generate_permutation() {
            indices_.resize(size_);
            for (size_t i = 0; i < size_; ++i) {
                indices_[i] = i;
            }
            std::shuffle(indices_.begin(), indices_.end(), gen_);
        }

        size_t size_;
        size_t current_position_;
        std::vector<size_t> indices_;
        std::mt19937 gen_;
    };

    // Torch-free infinite random sampler
    class InfiniteRandomSampler {
    public:
        explicit InfiniteRandomSampler(size_t dataset_size)
            : size_(dataset_size),
              current_position_(0),
              epoch_(0),
              gen_(std::random_device{}()) {

            if (size_ == 0) {
                throw std::invalid_argument("Dataset size must be positive");
            }

            // Generate initial permutation
            generate_permutation();
        }

        // Get next batch of indices
        std::vector<size_t> next(size_t batch_size) {
            std::vector<size_t> batch;
            batch.reserve(batch_size);

            for (size_t i = 0; i < batch_size; ++i) {
                // Check if we need to generate new permutation
                if (current_position_ >= indices_.size()) {
                    generate_permutation();
                    current_position_ = 0;
                    epoch_++;
                }

                batch.push_back(indices_[current_position_]);
                current_position_++;
            }

            return batch;
        }

        // Get single index
        size_t next_single() {
            std::lock_guard<std::mutex> lock(mutex_);
            if (current_position_ >= indices_.size()) {
                generate_permutation();
                current_position_ = 0;
                epoch_++;
            }

            return indices_[current_position_++];
        }

        void reset() {
            std::lock_guard<std::mutex> lock(mutex_);
            generate_permutation();
            current_position_ = 0;
            epoch_++;
        }

        size_t size() const { return size_; }
        size_t epoch() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return epoch_;
        }
        size_t position() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return current_position_;
        }

    private:
        void generate_permutation() {
            indices_.resize(size_);
            for (size_t i = 0; i < size_; ++i) {
                indices_[i] = i;
            }
            std::shuffle(indices_.begin(), indices_.end(), gen_);
        }

        size_t size_;
        size_t current_position_;
        size_t epoch_;
        std::vector<size_t> indices_;
        std::mt19937 gen_;
        mutable std::mutex mutex_;
    };

    // Simple batch structure
    struct CameraBatch {
        std::vector<CameraWithImage> data;

        // For compatibility with existing code that expects [0].data
        CameraWithImage& operator[](size_t idx) {
            if (idx >= data.size()) {
                throw std::out_of_range("Batch index out of range");
            }
            return data[idx];
        }

        const CameraWithImage& operator[](size_t idx) const {
            if (idx >= data.size()) {
                throw std::out_of_range("Batch index out of range");
            }
            return data[idx];
        }
    };

    // Regular DataLoader (non-infinite) - for evaluation, single-threaded
    class DataLoader {
    public:
        DataLoader(std::shared_ptr<CameraDataset> dataset,
                   size_t batch_size = 1,
                   [[maybe_unused]] int num_workers = 0)  // num_workers ignored for evaluation
            : dataset_(dataset),
              batch_size_(batch_size),
              sampler_(dataset->size()) {
        }

        // Iterator for range-based for loops
        class Iterator {
        public:
            Iterator(DataLoader* loader, bool is_end = false)
                : loader_(loader), is_end_(is_end) {
                if (!is_end_) {
                    advance();
                }
            }

            CameraBatch operator*() {
                return current_batch_;
            }

            Iterator& operator++() {
                advance();
                return *this;
            }

            bool operator!=(const Iterator& other) const {
                return is_end_ != other.is_end_;
            }

        private:
            void advance() {
                auto indices = loader_->sampler_.next(loader_->batch_size_);
                if (!indices) {
                    is_end_ = true;
                    return;
                }

                current_batch_.data.clear();
                for (size_t idx : *indices) {
                    current_batch_.data.push_back(loader_->dataset_->get(idx));
                }
            }

            DataLoader* loader_;
            CameraBatch current_batch_;
            bool is_end_;
        };

        Iterator begin() {
            sampler_.reset();  // Reset sampler for new epoch
            return Iterator(this, false);
        }

        Iterator end() {
            return Iterator(this, true);
        }

    private:
        std::shared_ptr<CameraDataset> dataset_;
        size_t batch_size_;
        RandomSampler sampler_;
    };

    // Torch-free Infinite DataLoader with worker threads for training
    class InfiniteDataLoader {
    public:
        InfiniteDataLoader(std::shared_ptr<CameraDataset> dataset,
                          size_t batch_size = 1,
                          int num_workers = 4)
            : dataset_(dataset),
              batch_size_(batch_size),
              num_workers_(std::max(0, num_workers)),
              sampler_(dataset->size()),
              stop_workers_(false),
              prefetch_factor_(2) {

            if (num_workers_ > 0) {
                // Start worker threads
                for (int i = 0; i < num_workers_; ++i) {
                    workers_.emplace_back(&InfiniteDataLoader::worker_thread, this, i);
                }
                LOG_DEBUG("Started {} worker threads for training dataloader", num_workers_);
            } else {
                LOG_WARN("Running training dataloader without workers - expect memory issues!");
            }
        }

        ~InfiniteDataLoader() {
            stop_workers_ = true;
            cv_workers_.notify_all();

            for (auto& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }

        // Iterator for range-based for loops
        class Iterator {
        public:
            Iterator(InfiniteDataLoader* loader, bool is_end = false)
                : loader_(loader), is_end_(is_end) {
                if (!is_end_) {
                    advance();
                }
            }

            CameraBatch operator*() {
                return current_batch_;
            }

            Iterator& operator++() {
                advance();
                return *this;
            }

            bool operator!=(const Iterator& other) const {
                // Infinite iterator never equals end
                return !is_end_ || !other.is_end_;
            }

        private:
            void advance() {
                if (loader_->num_workers_ > 0) {
                    // Get from queue (workers are loading in background)
                    current_batch_ = loader_->get_next_batch();
                } else {
                    // Single-threaded fallback
                    size_t idx = loader_->sampler_.next_single();
                    current_batch_.data.clear();
                    current_batch_.data.push_back(loader_->dataset_->get(idx));
                }
            }

            InfiniteDataLoader* loader_;
            CameraBatch current_batch_;
            bool is_end_;
        };

        Iterator begin() {
            return Iterator(this, false);
        }

        Iterator end() {
            return Iterator(this, true);
        }

    private:
        void worker_thread(int worker_id) {
            // Set CPU affinity if possible
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(worker_id % std::thread::hardware_concurrency(), &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

            while (!stop_workers_) {
                // Check queue size
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    cv_workers_.wait(lock, [this] {
                        return stop_workers_ ||
                               loaded_batches_.size() < prefetch_factor_ * num_workers_;
                    });
                }

                if (stop_workers_) break;

                // Get next index
                size_t idx = sampler_.next_single();

                // Load data (this happens outside the lock)
                CameraBatch batch;
                batch.data.push_back(dataset_->get(idx));

                // Move image to GPU immediately to avoid keeping CPU copy
                batch.data[0].image = batch.data[0].image.to(torch::kCUDA, /*non_blocking=*/true);

                // Add to queue
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    loaded_batches_.push(std::move(batch));
                }
                cv_main_.notify_one();
            }
        }

        CameraBatch get_next_batch() {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // Notify workers if queue is getting low
            if (loaded_batches_.size() < prefetch_factor_) {
                cv_workers_.notify_all();
            }

            // Wait for batch
            cv_main_.wait(lock, [this] {
                return !loaded_batches_.empty() || stop_workers_;
            });

            if (stop_workers_ && loaded_batches_.empty()) {
                return CameraBatch();
            }

            CameraBatch batch = std::move(loaded_batches_.front());
            loaded_batches_.pop();

            // Notify workers to produce more
            cv_workers_.notify_one();

            return batch;
        }

        std::shared_ptr<CameraDataset> dataset_;
        size_t batch_size_;
        int num_workers_;
        InfiniteRandomSampler sampler_;

        // Worker management
        std::vector<std::thread> workers_;
        std::queue<CameraBatch> loaded_batches_;
        std::mutex queue_mutex_;
        std::condition_variable cv_workers_;
        std::condition_variable cv_main_;
        std::atomic<bool> stop_workers_;
        size_t prefetch_factor_;
    };

    // Factory functions to match existing interface
    inline std::unique_ptr<DataLoader> create_dataloader_from_dataset(
        std::shared_ptr<CameraDataset> dataset,
        int num_workers = 4) {

        return std::make_unique<DataLoader>(dataset, 1, num_workers);
    }

    inline std::unique_ptr<InfiniteDataLoader> create_infinite_dataloader_from_dataset(
        std::shared_ptr<CameraDataset> dataset,
        int num_workers = 4) {

        return std::make_unique<InfiniteDataLoader>(dataset, 1, num_workers);
    }

    // Keep the existing create_dataset functions unchanged as they return CameraDataset
    inline std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
    create_dataset_from_colmap(const gs::param::DatasetConfig& datasetConfig) {
        try {
            if (!std::filesystem::exists(datasetConfig.data_path)) {
                return std::unexpected(std::format("Data path does not exist: {}",
                                                   datasetConfig.data_path.string()));
            }

            // Create loader
            auto loader = gs::loader::Loader::create();

            // Set up load options
            gs::loader::LoadOptions options{
                .resize_factor = datasetConfig.resize_factor,
                .images_folder = datasetConfig.images,
                .validate_only = false};

            // Load the data
            auto result = loader->load(datasetConfig.data_path, options);
            if (!result) {
                return std::unexpected(std::format("Failed to load COLMAP dataset: {}", result.error()));
            }

            // Handle the result
            return std::visit(
                [&result, &datasetConfig](
                    auto&& data) -> std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string> {
                    using T = std::decay_t<decltype(data)>;

                    if constexpr (std::is_same_v<T, std::shared_ptr<gs::SplatData>>) {
                        return std::unexpected("Expected COLMAP dataset but got PLY file");
                    } else if constexpr (std::is_same_v<T, gs::loader::LoadedScene>) {
                        if (!data.cameras) {
                            return std::unexpected("Loaded scene has no cameras");
                        }

                        // Create dataset with cameras that are internally torch-free
                        auto dataset = std::make_shared<CameraDataset>(
                            data.cameras->get_cameras(),
                            datasetConfig,
                            CameraDataset::Split::ALL
                        );

                        return std::make_tuple(dataset, result->scene_center);
                    } else {
                        return std::unexpected("Unknown data type returned from loader");
                    }
                },
                result->data);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to create dataset from COLMAP: {}", e.what()));
        }
    }

    inline std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string>
    create_dataset_from_transforms(const gs::param::DatasetConfig& datasetConfig) {
        try {
            if (!std::filesystem::exists(datasetConfig.data_path)) {
                return std::unexpected(std::format("Data path does not exist: {}",
                                                   datasetConfig.data_path.string()));
            }

            // Create loader
            auto loader = gs::loader::Loader::create();

            // Set up load options
            gs::loader::LoadOptions options{
                .resize_factor = datasetConfig.resize_factor,
                .images_folder = datasetConfig.images,
                .validate_only = false};

            // Load the data
            auto result = loader->load(datasetConfig.data_path, options);
            if (!result) {
                return std::unexpected(std::format("Failed to load transforms dataset: {}", result.error()));
            }

            // Handle the result
            return std::visit(
                [&datasetConfig, &result](
                    auto&& data) -> std::expected<std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor>, std::string> {
                    using T = std::decay_t<decltype(data)>;

                    if constexpr (std::is_same_v<T, std::shared_ptr<gs::SplatData>>) {
                        return std::unexpected("Expected transforms.json dataset but got PLY file");
                    } else if constexpr (std::is_same_v<T, gs::loader::LoadedScene>) {
                        if (!data.cameras) {
                            return std::unexpected("Loaded scene has no cameras");
                        }

                        // Create dataset with cameras that are internally torch-free
                        auto dataset = std::make_shared<CameraDataset>(
                            data.cameras->get_cameras(),
                            datasetConfig,
                            CameraDataset::Split::ALL
                        );

                        return std::make_tuple(dataset, result->scene_center);
                    } else {
                        return std::unexpected("Unknown data type returned from loader");
                    }
                },
                result->data);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to create dataset from transforms: {}", e.what()));
        }
    }
} // namespace gs::training