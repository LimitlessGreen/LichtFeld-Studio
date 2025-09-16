/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "infinite_random_sampler.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <torch/cuda.h>

namespace gs::training {

    InfiniteRandomSampler::InfiniteRandomSampler(int64_t size, torch::Dtype index_dtype)
        : size_(size),
          dtype_(index_dtype),
          generation_mutex_(std::make_shared<std::mutex>()) {

        // Validate inputs
        TORCH_CHECK(size > 0, "Dataset size must be positive, got ", size);
        TORCH_CHECK(dtype_ == torch::kInt32 || dtype_ == torch::kInt64,
                    "Index dtype must be Int32 or Int64, got ", dtype_);

        // Reserve reasonable default capacity for batch buffer
        batch_buffer_.reserve(std::min(static_cast<size_t>(64), static_cast<size_t>(size_)));

        // Generate initial permutation
        generate_indices();

        // Convert dtype to string for logging
        const char* dtype_str = (dtype_ == torch::kInt32) ? "int32" : "int64";
        LOG_TRACE("InfiniteRandomSampler created with size {} and dtype {}",
                  size_, dtype_str);
    }

    // Move constructor
    InfiniteRandomSampler::InfiniteRandomSampler(InfiniteRandomSampler&& other) noexcept
        : indices_(std::move(other.indices_)),
          index_(other.index_),
          size_(other.size_),
          dtype_(other.dtype_),
          epoch_(other.epoch_),
          batch_buffer_(std::move(other.batch_buffer_)),
          generation_mutex_(std::move(other.generation_mutex_)) {

        // Reset the moved-from object to a valid state
        other.index_ = 0;
        other.size_ = 0;
        other.epoch_ = 0;
        other.generation_mutex_ = std::make_shared<std::mutex>();
    }

    // Move assignment operator
    InfiniteRandomSampler& InfiniteRandomSampler::operator=(InfiniteRandomSampler&& other) noexcept {
        if (this != &other) {
            indices_ = std::move(other.indices_);
            index_ = other.index_;
            size_ = other.size_;
            dtype_ = other.dtype_;
            epoch_ = other.epoch_;
            batch_buffer_ = std::move(other.batch_buffer_);
            generation_mutex_ = std::move(other.generation_mutex_);

            // Reset the moved-from object
            other.index_ = 0;
            other.size_ = 0;
            other.epoch_ = 0;
            other.generation_mutex_ = std::make_shared<std::mutex>();
        }
        return *this;
    }

    void InfiniteRandomSampler::generate_indices() {
        // Thread-safe generation
        std::lock_guard<std::mutex> lock(*generation_mutex_);

        auto options = torch::TensorOptions().dtype(dtype_);

        // Use CUDA for faster generation if available and dataset is large enough
        constexpr int64_t CUDA_THRESHOLD = 10000;
        if (torch::cuda::is_available() && size_ > CUDA_THRESHOLD) {
            try {
                // Generate on GPU then transfer to CPU
                options = options.device(torch::kCUDA);
                indices_ = torch::randperm(size_, options).to(torch::kCPU, /*non_blocking=*/true);
                torch::cuda::synchronize(); // Ensure transfer completes
            } catch (const std::exception& e) {
                // Fallback to CPU if CUDA fails
                LOG_TRACE("CUDA permutation generation failed, using CPU: {}", e.what());
                indices_ = torch::randperm(size_, torch::TensorOptions().dtype(dtype_));
            }
        } else {
            // Direct CPU generation for small datasets
            indices_ = torch::randperm(size_, options);
        }

        // Reset index to beginning
        index_ = 0;
    }

    void InfiniteRandomSampler::reset(std::optional<size_t> new_size) {
        if (new_size.has_value()) {
            int64_t new_size_val = static_cast<int64_t>(*new_size);
            TORCH_CHECK(new_size_val > 0, "New size must be positive, got ", new_size_val);
            size_ = new_size_val;
        }

        generate_indices();
        epoch_++;

        LOG_TRACE("InfiniteRandomSampler reset: size={}, epoch={}", size_, epoch_);
    }

    void InfiniteRandomSampler::copy_indices_to_buffer(size_t count) {
        // Ensure buffer has correct size
        batch_buffer_.resize(count);

        // Optimized copy based on dtype
        if (dtype_ == torch::kInt64) {
            // Fast path: direct copy for int64
            const int64_t* data = indices_.data_ptr<int64_t>() + index_;
            std::copy(data, data + count, batch_buffer_.begin());
        } else if (dtype_ == torch::kInt32) {
            // Common case: int32 to size_t conversion
            const int32_t* data = indices_.data_ptr<int32_t>() + index_;
            std::transform(data, data + count, batch_buffer_.begin(),
                           [](int32_t idx) { return static_cast<size_t>(idx); });
        } else {
            // Fallback path (should not happen with validation in constructor)
            auto slice = indices_.slice(0, index_, index_ + count);
            slice = slice.to(torch::kInt64);
            const int64_t* data = slice.const_data_ptr<int64_t>();
            std::copy(data, data + count, batch_buffer_.begin());
        }
    }

    std::optional<std::vector<size_t>> InfiniteRandomSampler::next(size_t batch_size) {
        TORCH_CHECK(batch_size > 0, "Batch size must be positive");

        const int64_t remaining = indices_.numel() - index_;

        // Check if we need to wrap around for infinite behavior
        if (remaining <= 0) {
            // Automatically generate new permutation
            generate_indices();
            epoch_++;

            LOG_TRACE("InfiniteRandomSampler wrapped to new epoch: {}", epoch_);
        }

        // Calculate actual batch size (may be smaller at epoch boundary)
        const size_t actual_batch_size = std::min(
            batch_size,
            static_cast<size_t>(indices_.numel() - index_));

        // Reserve extra capacity if needed (with geometric growth)
        if (batch_buffer_.capacity() < actual_batch_size) {
            batch_buffer_.reserve(actual_batch_size * 2);
        }

        // Copy indices to buffer
        copy_indices_to_buffer(actual_batch_size);

        // Advance index
        index_ += static_cast<int64_t>(actual_batch_size);

        // Return copy of buffer (required by interface)
        return batch_buffer_;
    }

    void InfiniteRandomSampler::save(torch::serialize::OutputArchive& archive) const {
        std::lock_guard<std::mutex> lock(*generation_mutex_);

        archive.write("version", torch::tensor(1, torch::kInt32), /*is_buffer=*/true);
        archive.write("index", torch::tensor(index_, torch::kInt64), /*is_buffer=*/true);
        archive.write("indices", indices_, /*is_buffer=*/true);
        archive.write("size", torch::tensor(size_, torch::kInt64), /*is_buffer=*/true);

        // Save dtype as int32
        int32_t dtype_val = static_cast<int32_t>(dtype_);
        archive.write("dtype", torch::tensor(dtype_val, torch::kInt32), /*is_buffer=*/true);

        archive.write("epoch", torch::tensor(static_cast<int64_t>(epoch_), torch::kInt64), /*is_buffer=*/true);
    }

    void InfiniteRandomSampler::load(torch::serialize::InputArchive& archive) {
        std::lock_guard<std::mutex> lock(*generation_mutex_);

        // Check version for future compatibility
        auto version_tensor = torch::empty(1, torch::kInt32);
        archive.read("version", version_tensor, /*is_buffer=*/true);
        int32_t version = version_tensor.item<int32_t>();
        TORCH_CHECK(version == 1, "Unsupported sampler version: ", version);

        auto index_tensor = torch::empty(1, torch::kInt64);
        archive.read("index", index_tensor, /*is_buffer=*/true);
        index_ = index_tensor.item<int64_t>();

        archive.read("indices", indices_, /*is_buffer=*/true);

        auto size_tensor = torch::empty(1, torch::kInt64);
        archive.read("size", size_tensor, /*is_buffer=*/true);
        size_ = size_tensor.item<int64_t>();

        // Load dtype as int32 and cast
        auto dtype_tensor = torch::empty(1, torch::kInt32);
        archive.read("dtype", dtype_tensor, /*is_buffer=*/true);
        dtype_ = static_cast<torch::ScalarType>(dtype_tensor.item<int32_t>());

        auto epoch_tensor = torch::empty(1, torch::kInt64);
        archive.read("epoch", epoch_tensor, /*is_buffer=*/true);
        epoch_ = static_cast<size_t>(epoch_tensor.item<int64_t>());

        // Ensure buffer has appropriate capacity
        batch_buffer_.reserve(std::min(static_cast<size_t>(64), static_cast<size_t>(size_)));
    }

} // namespace gs::training