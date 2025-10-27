/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/logger.hpp"
#include <array>
#include <atomic>
#include <cuda_runtime.h>

namespace lfs::core {

    /**
     * @brief CUDA stream pool for concurrent kernel execution
     *
     * Manages multiple non-blocking CUDA streams to enable concurrent execution
     * of independent tensor operations. Uses round-robin allocation for load
     * balancing across streams.
     *
     * Performance impact:
     * - Multiple operations can execute concurrently on GPU
     * - CPU doesn't block waiting for GPU completion (async execution)
     * - Kernel overlap when operations are independent
     *
     * Expected speedup: 2-4× for typical training workloads with many operations
     *
     * Usage:
     *   auto stream = StreamPool::instance().get_stream();
     *   launch_kernel<<<grid, block, 0, stream>>>(args...);
     *   // At API boundaries (CPU←→GPU transfers):
     *   StreamPool::instance().sync_all();
     */
    class StreamPool {
    public:
        static constexpr size_t NUM_STREAMS = 4;

        static StreamPool& instance() {
            static StreamPool pool;
            return pool;
        }

        /**
         * @brief Get next stream in round-robin fashion
         * @return Non-blocking CUDA stream for kernel launch
         *
         * Thread-safe: Uses atomic fetch_add for lock-free round-robin
         */
        cudaStream_t get_stream() {
            size_t index = counter_.fetch_add(1, std::memory_order_relaxed) % NUM_STREAMS;
            return streams_[index];
        }

        /**
         * @brief Synchronize all streams
         *
         * Call this at API boundaries where CPU needs to wait for GPU results:
         * - Tensor::to(Device::CPU) - GPU→CPU transfer
         * - Tensor::item() - Read single value from GPU
         * - Tensor::to_host() - Copy GPU tensor to host memory
         */
        void sync_all() {
            for (auto stream : streams_) {
                cudaError_t err = cudaStreamSynchronize(stream);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaStreamSynchronize failed: {}", cudaGetErrorString(err));
                }
            }
        }

        /**
         * @brief Synchronize a specific stream
         * @param stream Stream to synchronize
         */
        void sync_stream(cudaStream_t stream) {
            cudaError_t err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaStreamSynchronize failed: {}", cudaGetErrorString(err));
            }
        }

        /**
         * @brief Get the default stream (nullptr)
         * @return Default CUDA stream
         *
         * Use this for operations that must complete before subsequent operations
         */
        cudaStream_t get_default_stream() const {
            return nullptr;
        }

        /**
         * @brief Get total number of streams in pool
         */
        size_t num_streams() const {
            return NUM_STREAMS;
        }

        // Disable copy and move
        StreamPool(const StreamPool&) = delete;
        StreamPool& operator=(const StreamPool&) = delete;
        StreamPool(StreamPool&&) = delete;
        StreamPool& operator=(StreamPool&&) = delete;

    private:
        StreamPool() : counter_(0) {
            // Create non-blocking streams
            for (size_t i = 0; i < NUM_STREAMS; ++i) {
                cudaError_t err = cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaStreamCreateWithFlags failed for stream {}: {}",
                              i, cudaGetErrorString(err));
                    streams_[i] = nullptr;
                }
            }
            LOG_INFO("StreamPool initialized with {} non-blocking CUDA streams", NUM_STREAMS);
        }

        ~StreamPool() {
            // Synchronize and destroy all streams
            for (size_t i = 0; i < NUM_STREAMS; ++i) {
                if (streams_[i]) {
                    cudaStreamSynchronize(streams_[i]);
                    cudaError_t err = cudaStreamDestroy(streams_[i]);
                    if (err != cudaSuccess) {
                        LOG_ERROR("cudaStreamDestroy failed for stream {}: {}",
                                  i, cudaGetErrorString(err));
                    }
                }
            }
        }

        std::array<cudaStream_t, NUM_STREAMS> streams_;
        std::atomic<size_t> counter_; // Thread-safe round-robin counter
    };

} // namespace lfs::core
