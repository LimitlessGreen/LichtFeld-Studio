/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/logger.hpp"
#include "gpu_arena_allocator.hpp"
#include "vmm_allocator.hpp"
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>
#include <memory>

namespace lfs::core {

    // Threshold for using arena allocator vs cudaMallocAsync
    // Tensors < 100MB use arena (O(1) allocation)
    // Tensors ≥ 100MB use VMM or cudaMallocAsync (avoid arena fragmentation)
    static constexpr size_t ARENA_ALLOCATOR_THRESHOLD = 100 * 1024 * 1024; // 100MB
    static constexpr size_t VMM_ALLOCATOR_THRESHOLD = 1024 * 1024;        // 1MB - use VMM for ≥1MB
    static constexpr size_t DIRECT_ALLOC_THRESHOLD = 1024 * 1024 * 1024;  // 1GB - bypass pool for very large allocations (fallback)

    /**
     * @brief Allocation method used for a pointer
     *
     * CRITICAL: We MUST track allocation method because deallocation differs:
     * - Arena: Return to arena pool
     * - VMM: Use VMM deallocate (decommits physical, keeps virtual mapping)
     * - Async: Use cudaFreeAsync (returns to CUDA cache for reuse)
     * - Direct: Use cudaFree (immediately returns to OS/driver)
     *
     * Mixing these causes memory leaks!
     */
    enum class AllocMethod : uint8_t {
        Arena,      // From GPUArenaAllocator
        VMM,        // From VMM_Allocator (CUDA 11.2+, VMM-capable GPU)
        Async,      // From cudaMallocAsync (CUDA 12.8+)
        Direct      // From cudaMalloc (large allocations, fallback)
    };

    /**
     * @brief CUDA memory pool for fast allocation/deallocation
     *
     * Uses cudaMallocAsync with memory pools (CUDA 12.8+) for near-instant
     * allocation from cached memory. Falls back to regular cudaMalloc on older
     * CUDA versions.
     *
     * Performance impact:
     * - cudaMallocAsync from pool: ~0.001-0.01ms (50-600× faster!)
     * - Regular cudaMalloc: ~0.15-0.6ms
     *
     * Expected speedup: 2-10× for typical tensor operations
     */
    class CudaMemoryPool {
    public:
        static CudaMemoryPool& instance() {
            static CudaMemoryPool pool;
            return pool;
        }

        /**
         * @brief Allocate memory from the pool
         * @param bytes Number of bytes to allocate
         * @param stream CUDA stream for stream-ordered allocation
         * @return Pointer to allocated memory, or nullptr on failure
         *
         * Hybrid allocation strategy:
         * - Small/medium (<100MB): Use GPU arena (O(1), 150-600× faster)
         * - Large (≥100MB): Use cudaMallocAsync pool (avoid fragmentation)
         */
        void* allocate(size_t bytes, cudaStream_t stream = nullptr) {
            if (bytes == 0) {
                return nullptr;
            }

            void* ptr = nullptr;

            // DISABLED: Arena and VMM allocators have stream-ordering bugs
            // TODO: Implement proper stream-ordered allocation with deferred freeing
            //
            // The arena and VMM allocators perform eager memory reuse without tracking
            // stream dependencies, causing use-after-free bugs. For now, we use
            // cudaMallocAsync (stream-ordered) and cudaMalloc (direct) exclusively.

            // 1. DISABLED: Arena for small allocations (<1MB)
            // if (bytes < VMM_ALLOCATOR_THRESHOLD && GPUArenaAllocator::instance().is_enabled()) {
            //     ptr = GPUArenaAllocator::instance().allocate(bytes);
            //     if (ptr != nullptr) {
            //         return ptr;
            //     }
            // }

            // 2. DISABLED: VMM for medium/large allocations (≥1MB, <1GB)
            // if (bytes >= VMM_ALLOCATOR_THRESHOLD && bytes < DIRECT_ALLOC_THRESHOLD) {
            //     if (get_or_create_vmm()) {
            //         ptr = vmm_->allocate(bytes, stream);
            //         if (ptr != nullptr) {
            //             std::lock_guard<std::mutex> lock(map_mutex_);
            //             allocation_map_[ptr] = AllocMethod::VMM;
            //             return ptr;
            //         }
            //     }
            // }

            // 3. Direct allocation for very large (≥1GB) or all allocations (arena/VMM disabled)

            if (bytes >= DIRECT_ALLOC_THRESHOLD) {
                // Check available memory before allocation
                size_t free_mem = 0, total_mem = 0;
                cudaMemGetInfo(&free_mem, &total_mem);

                cudaError_t err = cudaMalloc(&ptr, bytes);
                if (err != cudaSuccess) {
                    printf("[POOL ERROR] cudaMalloc (direct) failed: %s\n", cudaGetErrorString(err));
                    printf("[POOL ERROR] This might be due to fragmentation. Trying to free cached memory...\n");

                    // CRITICAL: Free CUDA memory pool cache to release memory back to OS
                    // This releases memory from cudaMallocAsync pool that's cached but unused
                    cudaDeviceSynchronize();  // Ensure all ops complete

#if CUDART_VERSION >= 12080
                    // Trim the memory pool to release unused memory
                    int device;
                    cudaGetDevice(&device);
                    cudaMemPool_t pool;
                    cudaDeviceGetDefaultMemPool(&pool, device);
                    cudaMemPoolTrimTo(pool, 0);  // Release all unused memory
                    printf("[POOL] Trimmed CUDA memory pool\n");
#endif

                    // Check what memory is available after trimming
                    cudaMemGetInfo(&free_mem, &total_mem);
                    printf("[POOL] After trim: free=%.2f GB, requested=%.2f GB\n",
                           free_mem / (1024.0 * 1024.0 * 1024.0),
                           bytes / (1024.0 * 1024.0 * 1024.0));

                    // Retry allocation
                    err = cudaMalloc(&ptr, bytes);
                    if (err != cudaSuccess) {
                        cudaMemGetInfo(&free_mem, &total_mem);
                        printf("[POOL ERROR] Retry failed. Free mem: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
                        LOG_ERROR("cudaMalloc (direct) failed for {} bytes: {}", bytes, cudaGetErrorString(err));
                        LOG_ERROR("Even after trimming pool, not enough memory available.");
                        LOG_ERROR("This indicates intermediate tensors are holding memory.");
                        return nullptr;
                    }
                }

                // CRITICAL: Track that this pointer came from direct cudaMalloc
                // Must use cudaFree (not cudaFreeAsync) for deallocation!
                {
                    std::lock_guard<std::mutex> lock(map_mutex_);
                    allocation_map_[ptr] = AllocMethod::Direct;
                }
                return ptr;
            }

            // Use cudaMallocAsync for medium-large allocations (100MB - 1GB)
            AllocMethod method;
#if CUDART_VERSION >= 12080
            // Use stream-ordered allocation with memory pool (FAST!)
            cudaError_t err = cudaMallocAsync(&ptr, bytes, stream);
            if (err == cudaSuccess) {
                method = AllocMethod::Async;
            } else {
                LOG_ERROR("cudaMallocAsync failed for {} bytes: {}",
                          bytes, cudaGetErrorString(err));
                // Fallback to direct allocation if pool fails
                LOG_WARN("Falling back to direct cudaMalloc");
                err = cudaMalloc(&ptr, bytes);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMalloc (fallback) failed for {} bytes: {}",
                              bytes, cudaGetErrorString(err));
                    return nullptr;
                }
                method = AllocMethod::Direct;  // Fallback uses direct
            }
#else
            // Fallback to synchronous allocation (SLOW) for CUDA < 12.8
            cudaError_t err = cudaMalloc(&ptr, bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaMalloc failed for {} bytes: {}",
                          bytes, cudaGetErrorString(err));
                return nullptr;
            }
            method = AllocMethod::Direct;  // Old CUDA version uses direct
            LOG_WARN("Using cudaMalloc (CUDA < 12.8). Consider upgrading for 50-600× faster allocation");
#endif

            // Track allocation method (not Arena, so must track in map)
            {
                std::lock_guard<std::mutex> lock(map_mutex_);
                allocation_map_[ptr] = method;
            }

            return ptr;
        }

        /**
         * @brief Deallocate memory back to the pool
         * @param ptr Pointer to memory to deallocate
         * @param stream CUDA stream for stream-ordered deallocation
         *
         * CRITICAL: Uses correct deallocation method based on how pointer was allocated:
         * - Arena → Return to arena pool
         * - Async (cudaMallocAsync) → cudaFreeAsync (returns to CUDA cache)
         * - Direct (cudaMalloc) → cudaFree (returns to OS/driver)
         *
         * Mixing allocation/deallocation methods causes memory leaks!
         */
        void deallocate(void* ptr, cudaStream_t stream = nullptr) {
            if (!ptr) {
                return;
            }

            // DISABLED: Arena allocator has stream-ordering bugs
            // if (GPUArenaAllocator::instance().is_enabled() &&
            //     GPUArenaAllocator::instance().owns_pointer(ptr)) {
            //     GPUArenaAllocator::instance().deallocate(ptr);
            //     return;
            // }

            // Look up allocation method from tracking map
            AllocMethod method;
            {
                std::lock_guard<std::mutex> lock(map_mutex_);
                auto it = allocation_map_.find(ptr);
                if (it == allocation_map_.end()) {
                    LOG_ERROR("Attempting to deallocate untracked pointer: {}", ptr);
                    LOG_ERROR("This likely means the pointer was allocated before tracking was added,");
                    LOG_ERROR("or there's a double-free bug. Attempting cudaFree as fallback.");
                    cudaFree(ptr);  // Best-effort fallback
                    return;
                }
                method = it->second;
                allocation_map_.erase(it);  // Remove from tracking
            }

            // CRITICAL: Use correct deallocation function based on allocation method!
            cudaError_t err;
            switch (method) {
                case AllocMethod::VMM:
                    // VMM: Decommit physical (returns RAM to OS), keep virtual mapping
                    if (vmm_) {
                        vmm_->deallocate(ptr, stream);
                    } else {
                        LOG_ERROR("VMM allocation but VMM allocator not available!");
                        cudaFree(ptr);  // Fallback
                    }
                    break;

                case AllocMethod::Direct:
                    // Direct cudaMalloc requires cudaFree (synchronous, returns to OS)
                    err = cudaFree(ptr);
                    if (err != cudaSuccess) {
                        LOG_ERROR("cudaFree failed for Direct allocation: {}", cudaGetErrorString(err));
                    }
                    break;

                case AllocMethod::Async:
                    // cudaMallocAsync requires cudaFreeAsync (returns to CUDA cache)
#if CUDART_VERSION >= 12080
                    err = cudaFreeAsync(ptr, stream);
                    if (err != cudaSuccess) {
                        LOG_ERROR("cudaFreeAsync failed for Async allocation: {}", cudaGetErrorString(err));
                    }
#else
                    // Shouldn't happen (Async not used on old CUDA), but fallback to cudaFree
                    LOG_WARN("Unexpected Async allocation on CUDA < 12.8, using cudaFree");
                    err = cudaFree(ptr);
                    if (err != cudaSuccess) {
                        LOG_ERROR("cudaFree failed: {}", cudaGetErrorString(err));
                    }
#endif
                    break;

                case AllocMethod::Arena:
                    // Should have been caught by owns_pointer check above
                    LOG_ERROR("Arena allocation not caught by owns_pointer check!");
                    GPUArenaAllocator::instance().deallocate(ptr);
                    break;

                default:
                    LOG_ERROR("Unknown allocation method for pointer: {}", ptr);
                    break;
            }
        }

        /**
         * @brief Configure memory pool settings for optimal performance
         */
        void configure() {
#if CUDART_VERSION >= 12080
            int device;
            cudaError_t err = cudaGetDevice(&device);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaGetDevice failed: {}", cudaGetErrorString(err));
                return;
            }

            // Get the default memory pool for this device
            cudaMemPool_t pool;
            err = cudaDeviceGetDefaultMemPool(&pool, device);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaDeviceGetDefaultMemPool failed: {}", cudaGetErrorString(err));
                return;
            }

            // Set pool release threshold - keep memory cached indefinitely
            // This prevents the pool from releasing memory back to the system,
            // maximizing reuse and minimizing allocation overhead
            uint64_t threshold = UINT64_MAX;
            err = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
            if (err != cudaSuccess) {
                LOG_WARN("cudaMemPoolSetAttribute failed: {}", cudaGetErrorString(err));
            }

            LOG_INFO("CUDA memory pool configured for device {} (CUDA {})",
                     device, CUDART_VERSION);
            LOG_INFO("Memory pool will cache allocations for maximum performance");
#else
            LOG_WARN("CUDA memory pooling not available (requires CUDA >= 12.8, current: {})",
                     CUDART_VERSION);
            LOG_WARN("Performance will be 50-600× slower than with memory pooling");
#endif
        }

        /**
         * @brief Get statistics about the memory pool
         * @return String with pool statistics (empty on CUDA < 12.8)
         */
        std::string get_stats() const {
#if CUDART_VERSION >= 12080
            int device;
            cudaGetDevice(&device);

            cudaMemPool_t pool;
            cudaDeviceGetDefaultMemPool(&pool, device);

            // Get used memory
            uint64_t used_memory = 0;
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used_memory);

            // Get reserved memory
            uint64_t reserved_memory = 0;
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved_memory);

            std::ostringstream oss;
            oss << "Memory Pool Stats:\n";
            oss << "  Used:     " << (used_memory / 1024.0 / 1024.0) << " MB\n";
            oss << "  Reserved: " << (reserved_memory / 1024.0 / 1024.0) << " MB\n";
            oss << "  Cached:   " << ((reserved_memory - used_memory) / 1024.0 / 1024.0) << " MB";
            return oss.str();
#else
            return "Memory pool statistics not available (CUDA < 12.8)";
#endif
        }

        /**
         * @brief Trim the memory pool, releasing unused memory back to the system
         *
         * This can be called periodically if memory pressure is high, but generally
         * it's better to keep memory cached for performance.
         */
        void trim() {
#if CUDART_VERSION >= 12080
            int device;
            cudaGetDevice(&device);

            cudaMemPool_t pool;
            cudaDeviceGetDefaultMemPool(&pool, device);

            // Release memory above the threshold
            cudaError_t err = cudaMemPoolTrimTo(pool, 0);
            if (err != cudaSuccess) {
                LOG_WARN("cudaMemPoolTrimTo failed: {}", cudaGetErrorString(err));
            } else {
                LOG_INFO("Memory pool trimmed successfully");
            }
#else
            LOG_DEBUG("Memory pool trim not available (CUDA < 12.8)");
#endif
        }

        // Disable copy and move
        CudaMemoryPool(const CudaMemoryPool&) = delete;
        CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;
        CudaMemoryPool(CudaMemoryPool&&) = delete;
        CudaMemoryPool& operator=(CudaMemoryPool&&) = delete;

    private:
        CudaMemoryPool() {
            configure();
        }

        ~CudaMemoryPool() {
            // Memory pool is automatically cleaned up by CUDA runtime
        }

        /**
         * @brief Get or create VMM allocator (lazy initialization)
         */
        bool get_or_create_vmm() {
            if (vmm_) {
                return true;
            }

            std::lock_guard<std::mutex> lock(vmm_mutex_);
            if (vmm_) {  // Double-check after lock
                return true;
            }

            // Check if VMM is supported
            if (!VMM_Allocator::is_vmm_supported()) {
                LOG_INFO("VMM not supported on this device, using fallback allocators");
                return false;
            }

            try {
                VMM_Allocator::Config config;
                config.virtual_size = (4ULL << 30) - (16 << 20);  // ~4 GB virtual (OffsetAllocator uint32 limit)
                config.initial_commit = 512 << 20;         // 512 MB initial
                config.max_physical = 20ULL << 30;         // 20 GB max
                config.commit_granularity = 16 << 20;      // 16 MB
                config.auto_decommit = true;               // Decommit on free
                config.decommit_threshold = 0.7f;          // Above 70% pressure
                config.enable_statistics = true;

                vmm_ = std::make_unique<VMM_Allocator>(config);
                LOG_INFO("VMM allocator initialized successfully");
                return true;
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to create VMM allocator: {}", e.what());
                return false;
            }
        }

        // Thread-safe tracking of allocation methods
        // CRITICAL: We must know which free function to call for each pointer!
        std::unordered_map<void*, AllocMethod> allocation_map_;
        std::mutex map_mutex_;

        // VMM allocator (lazy initialized)
        std::unique_ptr<VMM_Allocator> vmm_;
        std::mutex vmm_mutex_;
    };

} // namespace lfs::core
