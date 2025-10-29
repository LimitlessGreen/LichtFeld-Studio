// vmm_allocator.hpp
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include "core/logger.hpp"
#include "offset_allocator.hpp"

namespace lfs::core {

/**
 * @brief Virtual Memory Management (VMM) based GPU memory allocator
 *
 * Uses CUDA Virtual Memory Management to separate virtual address space from physical memory.
 * Provides fast reuse (10μs commit vs 100μs cudaMalloc) while returning freed memory to OS.
 *
 * Key features:
 * - 64 GB virtual address space (costs no RAM, just address mapping)
 * - Commit/decommit physical memory on demand
 * - Instant reuse by re-committing to same virtual address
 * - No fragmentation (contiguous virtual space)
 * - Memory returned to OS on decommit (available to other processes)
 *
 * Requirements:
 * - CUDA 11.2+ (for Virtual Memory Management API)
 * - Ampere/Ada/Hopper GPU (or any VMM-capable device)
 */
class VMM_Allocator {
public:
    struct Config {
        size_t virtual_size = (4ULL << 30) - (16 << 20);  // ~4 GB virtual (OffsetAllocator uint32 limit)
        size_t initial_commit = 512 << 20;         // 512 MB initial physical memory
        size_t max_physical = 20ULL << 30;         // 20 GB max physical memory
        size_t commit_granularity = 16 << 20;      // 16 MB chunks (must match GPU)
        size_t alignment = 256;                    // Alignment for allocations
        bool auto_decommit = true;                 // Decommit on free (vs lazy)
        float decommit_threshold = 0.7f;           // Decommit if memory pressure > 70%
        bool enable_statistics = true;             // Track detailed stats
    };

    struct Statistics {
        size_t virtual_reserved = 0;           // Virtual space reserved (cheap)
        size_t physical_committed = 0;         // Physical memory committed (expensive)
        size_t currently_allocated = 0;        // Memory in use by allocations
        size_t peak_allocated = 0;             // Peak memory in use
        size_t num_allocations = 0;            // Total allocations
        size_t num_deallocations = 0;          // Total deallocations
        size_t num_commits = 0;                // Physical memory commits
        size_t num_decommits = 0;              // Physical memory decommits
        size_t num_chunk_reuses = 0;           // Reused uncommitted chunks
        size_t fragmentation_events = 0;       // Failed to find contiguous range
    };

    struct MemoryInfo {
        size_t virtual_reserved = 0;
        size_t physical_committed = 0;
        size_t currently_allocated = 0;
        size_t peak_allocated = 0;
        size_t gpu_free = 0;
        size_t gpu_total = 0;
        float utilization_percent = 0.0f;          // currently_allocated / physical_committed
        float fragmentation_ratio = 0.0f;          // (committed - allocated) / committed
    };

    VMM_Allocator();
    explicit VMM_Allocator(const Config& cfg);
    ~VMM_Allocator();

    // Delete copy operations
    VMM_Allocator(const VMM_Allocator&) = delete;
    VMM_Allocator& operator=(const VMM_Allocator&) = delete;

    // Allow move operations
    VMM_Allocator(VMM_Allocator&&) noexcept;
    VMM_Allocator& operator=(VMM_Allocator&&) noexcept;

    /**
     * @brief Allocate GPU memory using VMM
     *
     * Fast path (if virtual range already committed): ~10μs
     * Slow path (if need to commit new physical memory): ~110μs
     *
     * @param bytes Number of bytes to allocate
     * @param stream CUDA stream for async operations (optional)
     * @return Pointer to allocated memory, or nullptr on failure
     */
    void* allocate(size_t bytes, cudaStream_t stream = nullptr);

    /**
     * @brief Deallocate GPU memory
     *
     * Decommits physical memory (returns to OS) but keeps virtual mapping.
     * Future allocations at the same address can re-commit instantly.
     *
     * @param ptr Pointer to free
     * @param stream CUDA stream for async operations (optional)
     */
    void deallocate(void* ptr, cudaStream_t stream = nullptr);

    /**
     * @brief Check if VMM is supported on this device
     */
    static bool is_vmm_supported(int device = 0);

    /**
     * @brief Get memory statistics
     */
    Statistics get_statistics() const;
    MemoryInfo get_memory_info() const;

    /**
     * @brief Get current memory pressure (0.0 = plenty, 1.0 = critical)
     */
    float get_memory_pressure() const;

    /**
     * @brief Manually trigger decommit of unused physical memory
     */
    void trim_unused_memory();

    /**
     * @brief Print detailed statistics
     */
    void dump_statistics() const;

private:
    struct PhysicalChunk {
        CUmemGenericAllocationHandle handle = 0;
        size_t size = 0;
        size_t granularity = 0;
        bool is_uncommitted = false;  // True if decommitted (available for reuse)
        int device = -1;
    };

    struct AllocationMetadata {
        PhysicalChunk* chunk = nullptr;     // Physical chunk backing this allocation
        size_t allocated_size = 0;          // User-requested size
        OffsetAllocator::Allocation alloc;  // OffsetAllocator metadata
    };

    // Virtual memory management
    CUdeviceptr virtual_base_ = 0;
    size_t virtual_size_ = 0;
    size_t granularity_ = 0;

    // Physical chunk management
    std::vector<std::unique_ptr<PhysicalChunk>> all_chunks_;
    std::set<PhysicalChunk*> uncommitted_chunks_;  // Available for reuse

    // Virtual address space management using OffsetAllocator (works in granularity units)
    std::unique_ptr<OffsetAllocator::Allocator> offset_allocator_;

    // Track allocation metadata: byte_offset → metadata
    std::unordered_map<size_t, AllocationMetadata> allocations_;
    mutable std::mutex allocations_mutex_;

    // Configuration and state
    Config config_;
    int device_ = -1;
    bool is_initialized_ = false;

    // Internal statistics (atomic for thread-safety)
    std::atomic<size_t> stats_virtual_reserved_{0};
    std::atomic<size_t> stats_physical_committed_{0};
    std::atomic<size_t> stats_currently_allocated_{0};
    std::atomic<size_t> stats_peak_allocated_{0};
    std::atomic<size_t> stats_num_allocations_{0};
    std::atomic<size_t> stats_num_deallocations_{0};
    std::atomic<size_t> stats_num_commits_{0};
    std::atomic<size_t> stats_num_decommits_{0};
    std::atomic<size_t> stats_num_chunk_reuses_{0};
    std::atomic<size_t> stats_fragmentation_events_{0};

    // Initialization
    bool initialize();
    void cleanup();

    // Physical memory management
    PhysicalChunk* create_physical_chunk(size_t size);
    PhysicalChunk* find_uncommitted_chunk(size_t size);
    bool commit_physical_to_virtual(size_t offset, size_t size, PhysicalChunk* chunk);
    void decommit_physical_from_virtual(size_t offset, size_t size, PhysicalChunk* chunk);

    // Utilities
    size_t align_size(size_t size) const;
    size_t round_up_to_granularity(size_t size) const;
    void update_statistics(size_t size, bool is_alloc);
};

} // namespace lfs::core
