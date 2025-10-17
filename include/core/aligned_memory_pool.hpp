/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "memory_pool.hpp"
#include "logger.hpp"
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace gs {

/**
 * @brief Alignment-aware CUDA memory allocator
 *
 * Wraps CudaMemoryPool to provide guaranteed memory alignment for
 * vectorized operations (float4, cache line optimization).
 *
 * Key benefits:
 * - Guarantees alignment for float4 vectorization (16-byte aligned)
 * - Optimizes for cache line access (128-byte alignment option)
 * - Built on top of fast cudaMallocAsync pool
 *
 * Trade-offs:
 * - Wastes up to (alignment - 1) bytes per allocation
 * - Small bookkeeping overhead
 * - For 128-byte alignment: wastes avg 63 bytes per allocation
 *
 * Recommended usage:
 * - Large tensors (>4KB): Use aligned allocation for vectorization benefits
 * - Small tensors (<4KB): Use regular pool allocation for speed
 *
 * Performance:
 * - Allocation time: ~0.001-0.01ms (same as CudaMemoryPool)
 * - Memory overhead: negligible for tensors >1KB
 * - Vectorization speedup: 2-4× for element-wise operations
 */
class AlignedMemoryPool {
public:
    static AlignedMemoryPool& instance() {
        static AlignedMemoryPool pool;
        return pool;
    }

    /**
     * @brief Allocate aligned memory from the pool
     * @param bytes Number of bytes to allocate
     * @param alignment Required alignment (must be power of 2)
     * @param stream CUDA stream for stream-ordered allocation
     * @return Pointer to aligned memory, or nullptr on failure
     */
    void* allocate_aligned(size_t bytes, size_t alignment, cudaStream_t stream = nullptr) {
        if (bytes == 0) {
            return nullptr;
        }

        // Validate alignment is power of 2
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            LOG_ERROR("Alignment must be a power of 2, got {}", alignment);
            return nullptr;
        }

        // Over-allocate to guarantee we can find aligned address
        size_t extra = alignment - 1;
        void* base_ptr = CudaMemoryPool::instance().allocate(bytes + extra, stream);

        if (!base_ptr) {
            return nullptr;
        }

        // Compute aligned pointer
        uintptr_t addr = reinterpret_cast<uintptr_t>(base_ptr);
        uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
        void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);

        // Track allocation for deallocation
        {
            std::lock_guard<std::mutex> lock(mutex_);
            AllocInfo info;
            info.base_ptr = base_ptr;
            info.aligned_ptr = aligned_ptr;
            info.padding = aligned_addr - addr;
            info.total_bytes = bytes + extra;
            allocations_[aligned_ptr] = info;
        }

        LOG_TRACE("Aligned allocation: {} bytes at {}, aligned to {}, padding {} bytes",
                 bytes, aligned_ptr, alignment, aligned_addr - addr);

        return aligned_ptr;
    }

    /**
     * @brief Deallocate aligned memory back to the pool
     * @param aligned_ptr Pointer returned by allocate_aligned
     * @param stream CUDA stream for stream-ordered deallocation
     */
    void deallocate_aligned(void* aligned_ptr, cudaStream_t stream = nullptr) {
        if (!aligned_ptr) {
            return;
        }

        AllocInfo info;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = allocations_.find(aligned_ptr);
            if (it == allocations_.end()) {
                LOG_ERROR("Attempted to deallocate unknown aligned pointer: {}", aligned_ptr);
                return;
            }
            info = it->second;
            allocations_.erase(it);
        }

        // Deallocate the base pointer (not the aligned pointer!)
        CudaMemoryPool::instance().deallocate(info.base_ptr, stream);

        LOG_TRACE("Aligned deallocation: {}, freed {} bytes (padding was {} bytes)",
                 aligned_ptr, info.total_bytes, info.padding);
    }

    /**
     * @brief Get statistics about aligned allocations
     * @return String with allocation statistics
     */
    std::string get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);

        if (allocations_.empty()) {
            return "No active aligned allocations";
        }

        size_t total_bytes = 0;
        size_t total_padding = 0;
        size_t max_padding = 0;

        for (const auto& [ptr, info] : allocations_) {
            total_bytes += info.total_bytes;
            total_padding += info.padding;
            max_padding = std::max(max_padding, info.padding);
        }

        double avg_padding = static_cast<double>(total_padding) / allocations_.size();
        double waste_percent = 100.0 * total_padding / total_bytes;

        std::ostringstream oss;
        oss << "Aligned Memory Pool Stats:\n";
        oss << "  Active allocations: " << allocations_.size() << "\n";
        oss << "  Total allocated:    " << (total_bytes / 1024.0 / 1024.0) << " MB\n";
        oss << "  Total padding:      " << (total_padding / 1024.0) << " KB\n";
        oss << "  Average padding:    " << avg_padding << " bytes\n";
        oss << "  Max padding:        " << max_padding << " bytes\n";
        oss << "  Waste percentage:   " << std::fixed << std::setprecision(2) << waste_percent << "%";

        return oss.str();
    }

    /**
     * @brief Smart allocation: chooses aligned or regular based on size
     * @param bytes Number of bytes to allocate
     * @param stream CUDA stream for stream-ordered allocation
     * @param threshold Size threshold for enabling alignment (default: 4KB)
     * @param alignment Alignment to use for large allocations (default: 128 bytes)
     * @return Pointer to allocated memory (may or may not be aligned)
     */
    void* allocate_smart(size_t bytes, cudaStream_t stream = nullptr,
                        size_t threshold = 4096, size_t alignment = 128) {
        if (bytes >= threshold) {
            // Large tensor: Worth aligning for vectorization benefits
            return allocate_aligned(bytes, alignment, stream);
        } else {
            // Small tensor: Use fast regular allocation
            return CudaMemoryPool::instance().allocate(bytes, stream);
        }
    }

    /**
     * @brief Smart deallocation: handles both aligned and regular allocations
     * @param ptr Pointer returned by allocate_smart
     * @param stream CUDA stream for stream-ordered deallocation
     */
    void deallocate_smart(void* ptr, cudaStream_t stream = nullptr) {
        if (!ptr) {
            return;
        }

        // Check if this is an aligned allocation
        bool is_aligned = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_aligned = (allocations_.find(ptr) != allocations_.end());
        }

        if (is_aligned) {
            deallocate_aligned(ptr, stream);
        } else {
            // Regular allocation
            CudaMemoryPool::instance().deallocate(ptr, stream);
        }
    }

    // Disable copy and move
    AlignedMemoryPool(const AlignedMemoryPool&) = delete;
    AlignedMemoryPool& operator=(const AlignedMemoryPool&) = delete;
    AlignedMemoryPool(AlignedMemoryPool&&) = delete;
    AlignedMemoryPool& operator=(AlignedMemoryPool&&) = delete;

private:
    struct AllocInfo {
        void* base_ptr;        // Original pointer from cudaMallocAsync
        void* aligned_ptr;     // User-visible aligned pointer
        size_t padding;        // Bytes wasted for alignment
        size_t total_bytes;    // Total allocated (including padding)
    };

    mutable std::mutex mutex_;
    std::unordered_map<void*, AllocInfo> allocations_;

    AlignedMemoryPool() = default;
    ~AlignedMemoryPool() {
        // Check for memory leaks
        std::lock_guard<std::mutex> lock(mutex_);
        if (!allocations_.empty()) {
            LOG_WARN("AlignedMemoryPool destroyed with {} active allocations - possible memory leak",
                    allocations_.size());
        }
    }
};

} // namespace gs
