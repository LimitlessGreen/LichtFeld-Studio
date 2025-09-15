/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <torch/data/samplers/base.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace gs::training {

/**
 * @brief High-performance infinite random sampler for continuous training
 *
 * This sampler provides infinite random sampling by automatically resetting
 * when exhausted. It includes several performance optimizations:
 * - Buffer reuse to minimize allocations
 * - Direct memory access for common data types
 * - Optional CUDA acceleration for permutation generation
 * - Memory-efficient int32 indices for smaller datasets
 */
class InfiniteRandomSampler : public torch::data::samplers::Sampler<> {
public:
    /**
     * @brief Construct an infinite random sampler
     * @param size Dataset size
     * @param index_dtype Data type for indices (kInt32 saves memory for smaller datasets)
     */
    explicit InfiniteRandomSampler(int64_t size, torch::Dtype index_dtype = torch::kInt64);

    ~InfiniteRandomSampler() override = default;

    // Delete copy constructor and assignment
    InfiniteRandomSampler(const InfiniteRandomSampler&) = delete;
    InfiniteRandomSampler& operator=(const InfiniteRandomSampler&) = delete;

    // Enable move constructor and assignment
    InfiniteRandomSampler(InfiniteRandomSampler&& other) noexcept;
    InfiniteRandomSampler& operator=(InfiniteRandomSampler&& other) noexcept;

    /**
     * @brief Reset the sampler, optionally with a new size
     * @param new_size Optional new dataset size
     */
    void reset(std::optional<size_t> new_size = std::nullopt) override;

    /**
     * @brief Get next batch of indices
     * @param batch_size Number of indices to return
     * @return Vector of indices, never nullopt (infinite behavior)
     */
    std::optional<std::vector<size_t>> next(size_t batch_size) override;

    /**
     * @brief Serialize sampler state
     */
    void save(torch::serialize::OutputArchive& archive) const override;

    /**
     * @brief Deserialize sampler state
     */
    void load(torch::serialize::InputArchive& archive) override;

    /**
     * @brief Get current position in the current permutation
     */
    size_t index() const noexcept { return static_cast<size_t>(index_); }

    /**
     * @brief Get current epoch count (number of complete passes)
     */
    size_t epoch() const noexcept { return epoch_; }

    /**
     * @brief Get dataset size
     */
    size_t size() const noexcept { return static_cast<size_t>(size_); }

private:
    /**
     * @brief Generate new random permutation
     */
    void generate_indices();

    /**
     * @brief Optimized copy from tensor to vector based on dtype
     */
    void copy_indices_to_buffer(size_t count);

private:
    torch::Tensor indices_;           // Current permutation
    int64_t index_ = 0;              // Current position in permutation
    int64_t size_;                   // Dataset size
    torch::Dtype dtype_;             // Data type of indices
    size_t epoch_ = 0;               // Epoch counter

    // Performance optimization: reuse buffer
    mutable std::vector<size_t> batch_buffer_;

    // Thread safety for generation - using shared_ptr for movability
    mutable std::shared_ptr<std::mutex> generation_mutex_;
};

} // namespace gs::training