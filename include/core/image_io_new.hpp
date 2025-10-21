/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <condition_variable>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <vector>

namespace gs {
    namespace image_io {

        /**
         * @brief Get image information (width, height, channels) without loading the full image
         * @param path Path to the image file
         * @return Tuple of (width, height, channels)
         */
        std::tuple<int, int, int> get_image_info(const std::filesystem::path& path);

        /**
         * @brief Load image from disk
         * @param path Path to the image file
         * @param resize_factor Downsampling factor (1, 2, 4, 8, or -1 for no resize)
         * @return Tuple of (data pointer, width, height, channels)
         *
         * The returned data pointer must be freed with free_image().
         * Data is always RGB (3 channels), uint8, row-major layout.
         */
        std::tuple<unsigned char*, int, int, int>
        load_image(const std::filesystem::path& path, int resize_factor = -1);

        /**
         * @brief Free image data allocated by load_image
         * @param data Image data pointer
         */
        void free_image(unsigned char* data);

        /**
         * @brief Save a tensor as an image
         * @param path Output file path
         * @param image Tensor to save. Can be:
         *   - [H, W, C] layout (C = 1, 2, 3, or 4)
         *   - [C, H, W] layout (C = 1, 2, 3, or 4)
         *   - [1, C, H, W] layout (batch dimension will be squeezed)
         *
         * The tensor will be normalized to [0, 1] and converted to uint8.
         * Supports formats: PNG, JPG, BMP, TGA, etc. (based on extension)
         */
        void save_image(const std::filesystem::path& path, const Tensor& image);

        /**
         * @brief Save multiple tensors side-by-side as a single image
         * @param path Output file path
         * @param images Vector of tensors to concatenate
         * @param horizontal If true, concatenate horizontally; otherwise vertically
         * @param separator_width Width of white separator between images (in pixels)
         */
        void save_image(const std::filesystem::path& path,
                        const std::vector<Tensor>& images,
                        bool horizontal = true,
                        int separator_width = 2);

        /**
         * @brief Batch image saver for asynchronous image saving
         *
         * This class manages a thread pool for saving images in the background,
         * preventing I/O operations from blocking the main computation thread.
         */
        class BatchImageSaver {
        public:
            /**
             * @brief Get singleton instance
             */
            static BatchImageSaver& instance() {
                static BatchImageSaver instance;
                return instance;
            }

            // Delete copy/move constructors
            BatchImageSaver(const BatchImageSaver&) = delete;
            BatchImageSaver& operator=(const BatchImageSaver&) = delete;
            BatchImageSaver(BatchImageSaver&&) = delete;
            BatchImageSaver& operator=(BatchImageSaver&&) = delete;

            /**
             * @brief Queue a single image for asynchronous saving
             * @param path Output file path
             * @param image Image tensor to save
             */
            void queue_save(const std::filesystem::path& path, Tensor image);

            /**
             * @brief Queue multiple images for side-by-side saving
             * @param path Output file path
             * @param images Vector of image tensors
             * @param horizontal Concatenation direction
             * @param separator_width Width of separator between images
             */
            void queue_save_multiple(const std::filesystem::path& path,
                                     const std::vector<Tensor>& images,
                                     bool horizontal = true,
                                     int separator_width = 2);

            /**
             * @brief Wait for all pending saves to complete
             */
            void wait_all();

            /**
             * @brief Flush all pending saves and stop worker threads
             *
             * Called automatically on destruction
             */
            void shutdown();

            /**
             * @brief Get number of pending save operations
             */
            size_t pending_count() const;

            /**
             * @brief Enable or disable batch saving
             *
             * When disabled, all operations are performed synchronously.
             * Useful for debugging.
             */
            void set_enabled(bool enabled) { enabled_ = enabled; }

            /**
             * @brief Check if batch saving is enabled
             */
            bool is_enabled() const { return enabled_; }

        private:
            BatchImageSaver(size_t num_workers = 4);
            ~BatchImageSaver();

            struct SaveTask {
                std::filesystem::path path;
                Tensor image;
                std::vector<Tensor> images;
                bool is_multi;
                bool horizontal;
                int separator_width;
            };

            void worker_thread();
            void process_task(const SaveTask& task);

            std::vector<std::thread> workers_;
            std::queue<SaveTask> task_queue_;
            mutable std::mutex queue_mutex_;
            std::condition_variable cv_;
            std::condition_variable cv_finished_;
            std::atomic<bool> stop_{false};
            std::atomic<size_t> active_tasks_{0};
            std::atomic<bool> enabled_{true};
            size_t num_workers_;
        };

        // ============================================================================
        // Convenience Functions
        // ============================================================================

        /**
         * @brief Save image asynchronously using the singleton BatchImageSaver
         */
        inline void save_image_async(const std::filesystem::path& path, Tensor image) {
            BatchImageSaver::instance().queue_save(path, std::move(image));
        }

        /**
         * @brief Save multiple images asynchronously using the singleton BatchImageSaver
         */
        inline void save_images_async(const std::filesystem::path& path,
                                      const std::vector<Tensor>& images,
                                      bool horizontal = true,
                                      int separator_width = 2) {
            BatchImageSaver::instance().queue_save_multiple(path, images, horizontal, separator_width);
        }

        /**
         * @brief Wait for all pending asynchronous saves
         */
        inline void wait_for_pending_saves() {
            BatchImageSaver::instance().wait_all();
        }

    } // namespace image_io
} // namespace gs
