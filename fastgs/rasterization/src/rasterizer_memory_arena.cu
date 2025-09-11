/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rasterizer_memory_arena.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <c10/cuda/CUDACachingAllocator.h>

namespace fast_gs::rasterization {

// Default constructor implementation
RasterizerMemoryArena::RasterizerMemoryArena()
    : RasterizerMemoryArena(Config{}) {
}

RasterizerMemoryArena::RasterizerMemoryArena(const Config& cfg)
    : config_(cfg), creation_time_(std::chrono::steady_clock::now()) {

    // Reduce initial size if it seems too large
    size_t free_memory, total_memory;
    cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
    if (err == cudaSuccess) {
        // Use at most 25% of free memory for initial allocation
        size_t max_initial = free_memory / 4;
        if (config_.initial_size > max_initial) {
            config_.initial_size = max_initial;
            std::cout << "[RasterizerMemoryArena] Reduced initial size to "
                      << (config_.initial_size >> 20) << " MB based on available memory" << std::endl;
        }
    }

    std::cout << "[RasterizerMemoryArena] Created with config:\n"
              << "  Initial size: " << (config_.initial_size >> 20) << " MB\n"
              << "  Max size: " << (config_.max_size >> 20) << " MB\n"
              << "  Growth factor: " << config_.growth_factor << "\n"
              << "  Alignment: " << config_.alignment << " bytes\n"
              << "  Log interval: every " << config_.log_interval << " frames\n";
}

RasterizerMemoryArena::~RasterizerMemoryArena() {
    dump_statistics();

    // Clean up all arenas
    std::lock_guard<std::mutex> lock(arena_mutex_);
    device_arenas_.clear();
    frame_contexts_.clear();
}

RasterizerMemoryArena::RasterizerMemoryArena(RasterizerMemoryArena&& other) noexcept
    : device_arenas_(std::move(other.device_arenas_)),
      frame_contexts_(std::move(other.frame_contexts_)),
      config_(other.config_),
      frame_counter_(other.frame_counter_.load()),
      generation_counter_(other.generation_counter_.load()),
      creation_time_(other.creation_time_),
      total_frames_processed_(other.total_frames_processed_.load()) {
}

RasterizerMemoryArena& RasterizerMemoryArena::operator=(RasterizerMemoryArena&& other) noexcept {
    if (this != &other) {
        std::lock_guard<std::mutex> lock1(arena_mutex_);
        std::lock_guard<std::mutex> lock2(other.arena_mutex_);

        device_arenas_ = std::move(other.device_arenas_);
        frame_contexts_ = std::move(other.frame_contexts_);
        config_ = other.config_;
        frame_counter_ = other.frame_counter_.load();
        generation_counter_ = other.generation_counter_.load();
        creation_time_ = other.creation_time_;
        total_frames_processed_ = other.total_frames_processed_.load();
    }
    return *this;
}

uint64_t RasterizerMemoryArena::begin_frame() {
    uint64_t frame_id = frame_counter_.fetch_add(1, std::memory_order_relaxed);

    // CRITICAL FIX: Reset arena offset at the beginning of each frame!
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaSuccess) {
        std::lock_guard<std::mutex> lock(arena_mutex_);
        auto it = device_arenas_.find(device);
        if (it != device_arenas_.end() && it->second) {
            // Reset the offset to reuse the buffer from the beginning
            it->second->offset.store(0, std::memory_order_release);

            // Log memory status periodically (but not too often)
            bool should_log = (frame_id == 1) || (frame_id % config_.log_interval == 0);

            if (should_log) {
                log_memory_status(frame_id, true);
            }
        }
    }

    std::lock_guard<std::mutex> lock(frame_mutex_);

    // Create new frame context
    FrameContext& ctx = frame_contexts_[frame_id];
    ctx.frame_id = frame_id;
    ctx.generation = generation_counter_.load(std::memory_order_relaxed);
    ctx.is_active = true;
    ctx.timestamp = std::chrono::steady_clock::now();
    ctx.buffers.clear();
    ctx.total_allocated = 0;

    total_frames_processed_.fetch_add(1, std::memory_order_relaxed);

    return frame_id;
}

void RasterizerMemoryArena::end_frame(uint64_t frame_id) {
    // Track peak usage before resetting
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaSuccess) {
        std::lock_guard<std::mutex> lock(arena_mutex_);
        auto it = device_arenas_.find(device);
        if (it != device_arenas_.end() && it->second) {
            size_t frame_usage = it->second->offset.load(std::memory_order_relaxed);

            // Update overall peak
            size_t current_peak = it->second->peak_usage.load(std::memory_order_relaxed);
            while (frame_usage > current_peak) {
                if (it->second->peak_usage.compare_exchange_weak(current_peak, frame_usage)) {
                    break;
                }
            }

            // Update period peak (for logging)
            size_t period_peak = it->second->peak_usage_period.load(std::memory_order_relaxed);
            while (frame_usage > period_peak) {
                if (it->second->peak_usage_period.compare_exchange_weak(period_peak, frame_usage)) {
                    break;
                }
            }
        }
    }

    std::lock_guard<std::mutex> lock(frame_mutex_);

    auto it = frame_contexts_.find(frame_id);
    if (it != frame_contexts_.end()) {
        it->second.is_active = false;
    }

    // Cleanup old frames - keep only last 3
    cleanup_frames(3);
}

void RasterizerMemoryArena::log_memory_status(uint64_t frame_id, bool force) {
    // Called with arena_mutex_ already held
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return;

    auto it = device_arenas_.find(device);
    if (it == device_arenas_.end() || !it->second) return;

    auto& arena = *it->second;

    // Get memory info
    size_t capacity_mb = arena.capacity >> 20;
    size_t current_usage_mb = arena.offset.load() >> 20;
    size_t peak_period_mb = arena.peak_usage_period.load() >> 20;
    size_t peak_overall_mb = arena.peak_usage.load() >> 20;

    // Get GPU memory info
    size_t free_gpu, total_gpu;
    cudaMemGetInfo(&free_gpu, &total_gpu);

    // Calculate utilization
    float utilization = arena.capacity > 0 ?
        (100.0f * arena.peak_usage_period.load() / arena.capacity) : 0.0f;

    // Log the status
    std::cout << "\n[Arena Memory Status] Frame " << frame_id << " | Device " << device << ":\n"
              << "  Reserved: " << capacity_mb << " MB (fixed allocation)\n"
              << "  Peak usage (last " << config_.log_interval << " frames): "
              << peak_period_mb << " MB (" << std::fixed << std::setprecision(1)
              << utilization << "%)\n"
              << "  Peak usage (overall): " << peak_overall_mb << " MB\n"
              << "  Reallocations: " << arena.realloc_count.load() << "\n"
              << "  GPU: " << (free_gpu >> 20) << "/" << (total_gpu >> 20)
              << " MB free (" << std::fixed << std::setprecision(1)
              << (100.0f * free_gpu / total_gpu) << "% available)\n";

    // Reset period peak for next logging interval
    arena.peak_usage_period.store(0, std::memory_order_release);
    arena.last_log_time = std::chrono::steady_clock::now();
}

std::function<char*(size_t)> RasterizerMemoryArena::get_allocator(uint64_t frame_id) {
    return [this, frame_id](size_t size) -> char* {
        if (size == 0) {
            return nullptr;
        }

        int device;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to get CUDA device: " +
                                   std::string(cudaGetErrorString(err)));
        }

        Arena& arena = get_or_create_arena(device);
        return allocate_internal(arena, size, frame_id);
    };
}

std::vector<RasterizerMemoryArena::BufferHandle>
RasterizerMemoryArena::get_frame_buffers(uint64_t frame_id) const {
    std::lock_guard<std::mutex> lock(frame_mutex_);

    auto it = frame_contexts_.find(frame_id);
    if (it != frame_contexts_.end()) {
        return it->second.buffers;
    }

    return {};
}

void RasterizerMemoryArena::reset_frame(uint64_t frame_id) {
    std::lock_guard<std::mutex> lock(frame_mutex_);

    auto it = frame_contexts_.find(frame_id);
    if (it != frame_contexts_.end()) {
        // Keep buffers but mark as reusable
        it->second.total_allocated = 0;
    }
}

void RasterizerMemoryArena::cleanup_frames(int keep_recent) {
    // Called with frame_mutex_ already held

    if (frame_contexts_.size() <= static_cast<size_t>(keep_recent)) {
        return;
    }

    // Find oldest frames to remove
    std::vector<uint64_t> frame_ids;
    frame_ids.reserve(frame_contexts_.size());

    for (const auto& [id, ctx] : frame_contexts_) {
        if (!ctx.is_active) {
            frame_ids.push_back(id);
        }
    }

    if (frame_ids.size() <= static_cast<size_t>(keep_recent)) {
        return;
    }

    // Sort by frame ID (oldest first)
    std::sort(frame_ids.begin(), frame_ids.end());

    // Remove oldest frames
    size_t to_remove = frame_ids.size() - keep_recent;
    for (size_t i = 0; i < to_remove; ++i) {
        frame_contexts_.erase(frame_ids[i]);
    }
}

void RasterizerMemoryArena::emergency_cleanup() {
    std::lock_guard<std::mutex> lock1(arena_mutex_);
    std::lock_guard<std::mutex> lock2(frame_mutex_);

    std::cout << "\n[RasterizerMemoryArena] ⚠️  EMERGENCY CLEANUP ⚠️" << std::endl;

    // Clear all inactive frames
    auto it = frame_contexts_.begin();
    while (it != frame_contexts_.end()) {
        if (!it->second.is_active) {
            it = frame_contexts_.erase(it);
        } else {
            ++it;
        }
    }

    // Reset all arena offsets
    for (auto& [device, arena] : device_arenas_) {
        if (arena) {
            arena->offset.store(0, std::memory_order_release);
            std::cout << "  Reset arena on device " << device
                     << " (capacity: " << (arena->capacity >> 20) << " MB)" << std::endl;
        }
    }

    // Force PyTorch to release cached memory
    c10::cuda::CUDACachingAllocator::emptyCache();

    // Synchronize all devices
    for (const auto& [device, arena] : device_arenas_) {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
    }

    std::cout << "[RasterizerMemoryArena] Emergency cleanup completed\n" << std::endl;
}

RasterizerMemoryArena::Arena& RasterizerMemoryArena::get_or_create_arena(int device) {
    std::lock_guard<std::mutex> lock(arena_mutex_);

    auto& arena_ptr = device_arenas_[device];
    if (!arena_ptr) {
        arena_ptr = std::make_unique<Arena>();
        arena_ptr->device = device;
        arena_ptr->last_log_time = std::chrono::steady_clock::now();

        // Set device before allocating
        cudaSetDevice(device);

        // Check available memory
        size_t free_memory, total_memory;
        cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to query GPU memory: " +
                                   std::string(cudaGetErrorString(err)));
        }

        // Start with a reasonable initial size
        size_t initial_size = std::min({
            config_.initial_size,
            free_memory / 2,  // Use at most half of free memory
            size_t(512) << 20  // Start with 512MB
        });

        if (initial_size < (64 << 20)) {  // Less than 64MB
            throw std::runtime_error("Insufficient GPU memory for arena initialization (need at least 64MB)");
        }

        // Try to allocate with fallback to smaller sizes
        bool allocated = false;
        while (initial_size >= (64 << 20) && !allocated) {
            try {
                auto options = torch::TensorOptions()
                    .dtype(torch::kUInt8)
                    .device(torch::kCUDA, device)
                    .requires_grad(false);

                arena_ptr->buffer = torch::empty({static_cast<long long>(initial_size)}, options);
                arena_ptr->capacity = initial_size;
                arena_ptr->generation = generation_counter_.fetch_add(1, std::memory_order_relaxed);
                arena_ptr->offset.store(0, std::memory_order_release);
                allocated = true;

                // ALWAYS LOG INITIAL ALLOCATION
                std::cout << "\n========================================\n"
                          << "[RasterizerMemoryArena] INITIAL ALLOCATION\n"
                          << "  Device: " << device << "\n"
                          << "  Size: " << (initial_size >> 20) << " MB\n"
                          << "  GPU free before: " << (free_memory >> 20) << " MB\n";

                cudaMemGetInfo(&free_memory, &total_memory);
                std::cout << "  GPU free after: " << (free_memory >> 20) << " MB\n"
                          << "========================================\n" << std::endl;

            } catch (const c10::Error& e) {
                // Try with half the size
                initial_size /= 2;
                std::cout << "[RasterizerMemoryArena] Allocation failed, trying "
                          << (initial_size >> 20) << " MB" << std::endl;
            }
        }

        if (!allocated) {
            throw std::runtime_error("Failed to allocate arena after multiple attempts");
        }
    }

    return *arena_ptr;
}

char* RasterizerMemoryArena::allocate_internal(Arena& arena, size_t size, uint64_t frame_id) {
    size_t aligned_size = align_size(size);

    // Sanity check
    if (aligned_size > config_.max_size) {
        throw std::runtime_error("Single allocation request " + std::to_string(aligned_size >> 20) +
                               " MB exceeds max arena size " + std::to_string(config_.max_size >> 20) + " MB");
    }

    // Try fast path allocation
    size_t offset = arena.offset.fetch_add(aligned_size, std::memory_order_acq_rel);

    if (offset + aligned_size <= arena.capacity) {
        // Success - record allocation
        char* ptr = static_cast<char*>(arena.buffer.data_ptr()) + offset;

        // Update peak usage
        size_t current_usage = offset + aligned_size;
        size_t peak = arena.peak_usage.load(std::memory_order_relaxed);
        while (current_usage > peak) {
            if (arena.peak_usage.compare_exchange_weak(peak, current_usage,
                                                       std::memory_order_relaxed)) {
                break;
            }
        }

        // Update period peak
        size_t period_peak = arena.peak_usage_period.load(std::memory_order_relaxed);
        while (current_usage > period_peak) {
            if (arena.peak_usage_period.compare_exchange_weak(period_peak, current_usage,
                                                              std::memory_order_relaxed)) {
                break;
            }
        }

        // Record buffer handle
        BufferHandle handle;
        handle.ptr = ptr;
        handle.size = aligned_size;
        handle.generation = arena.generation;
        handle.device = arena.device;

        record_allocation(frame_id, handle);

        return ptr;
    }

    // Allocation failed - need to grow
    arena.offset.fetch_sub(aligned_size, std::memory_order_acq_rel);

    // Grow arena under lock
    std::lock_guard<std::mutex> lock(arena_mutex_);

    // Double-check after acquiring lock
    offset = arena.offset.load(std::memory_order_acquire);
    if (offset + aligned_size <= arena.capacity) {
        // Another thread grew it, retry
        return allocate_internal(arena, size, frame_id);
    }

    // Calculate required size
    size_t required_size = offset + aligned_size;

    // We need to grow
    if (!grow_arena(arena, required_size)) {
        throw std::runtime_error("Failed to grow arena for allocation of " +
                               std::to_string(size >> 20) + " MB (current usage: " +
                               std::to_string(offset >> 20) + " MB, capacity: " +
                               std::to_string(arena.capacity >> 20) + " MB)");
    }

    // Retry allocation with new buffer
    return allocate_internal(arena, size, frame_id);
}

bool RasterizerMemoryArena::grow_arena(Arena& arena, size_t required_size) {
    // Called with arena_mutex_ held

    // Get current frame for logging
    uint64_t current_frame = frame_counter_.load(std::memory_order_relaxed);

    size_t old_capacity = arena.capacity;
    size_t new_capacity = std::max(
        required_size * 2,  // Double the required size for headroom
        static_cast<size_t>(arena.capacity * config_.growth_factor)
    );

    // Round up to 128MB boundary
    new_capacity = ((new_capacity + (128 << 20) - 1) / (128 << 20)) * (128 << 20);
    new_capacity = std::min(new_capacity, config_.max_size);

    if (new_capacity <= arena.capacity) {
        std::cerr << "\n[RasterizerMemoryArena] ❌ CANNOT GROW - MAX SIZE REACHED\n"
                  << "  Current capacity: " << (arena.capacity >> 20) << " MB\n"
                  << "  Required: " << (required_size >> 20) << " MB\n"
                  << "  Max allowed: " << (config_.max_size >> 20) << " MB\n" << std::endl;
        return false;
    }

    // Check available memory
    size_t free_memory_before, total_memory;
    cudaError_t err = cudaMemGetInfo(&free_memory_before, &total_memory);
    if (err != cudaSuccess) {
        return false;
    }

    size_t additional_needed = new_capacity - arena.capacity;

    // ALWAYS LOG GROWTH ATTEMPT
    std::cout << "\n========================================\n"
              << "[RasterizerMemoryArena] 📈 GROWING ARENA\n"
              << "  Frame: " << current_frame << "\n"
              << "  Device: " << arena.device << "\n"
              << "  Current capacity: " << (old_capacity >> 20) << " MB\n"
              << "  Required size: " << (required_size >> 20) << " MB\n"
              << "  New capacity: " << (new_capacity >> 20) << " MB\n"
              << "  Additional needed: " << (additional_needed >> 20) << " MB\n"
              << "  GPU free: " << (free_memory_before >> 20) << " MB\n"
              << "  Reallocation #" << (arena.realloc_count.load() + 1) << "\n";

    if (free_memory_before < additional_needed + (200 << 20)) {  // Keep 200MB free
        std::cout << "  ⚠️  Low memory - attempting cleanup...\n";
        // Try to free cached memory
        c10::cuda::CUDACachingAllocator::emptyCache();
        cudaMemGetInfo(&free_memory_before, &total_memory);
        std::cout << "  GPU free after cleanup: " << (free_memory_before >> 20) << " MB\n";

        if (free_memory_before < additional_needed + (200 << 20)) {
            std::cout << "  ❌ INSUFFICIENT MEMORY FOR GROWTH\n"
                      << "========================================\n" << std::endl;
            return false;
        }
    }

    try {
        // Allocate new buffer
        auto options = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::kCUDA, arena.device)
            .requires_grad(false);

        torch::Tensor new_buffer = torch::empty({static_cast<long long>(new_capacity)}, options);

        // Copy existing data
        size_t copy_size = arena.offset.load(std::memory_order_acquire);
        if (copy_size > 0 && arena.buffer.defined()) {
            new_buffer.slice(0, 0, copy_size).copy_(
                arena.buffer.slice(0, 0, copy_size)
            );
        }

        // Replace buffer - old one automatically freed by PyTorch's reference counting
        arena.buffer = std::move(new_buffer);
        arena.capacity = new_capacity;
        arena.generation = generation_counter_.fetch_add(1, std::memory_order_relaxed);
        arena.realloc_count.fetch_add(1, std::memory_order_relaxed);

        // Get memory after allocation
        size_t free_memory_after;
        cudaMemGetInfo(&free_memory_after, &total_memory);

        std::cout << "  ✅ GROWTH SUCCESSFUL\n"
                  << "  GPU free after growth: " << (free_memory_after >> 20) << " MB\n"
                  << "  Memory used for growth: " << ((free_memory_before - free_memory_after) >> 20) << " MB\n"
                  << "========================================\n" << std::endl;

        return true;

    } catch (const c10::Error& e) {
        std::cout << "  ❌ GROWTH FAILED: " << e.what() << "\n"
                  << "========================================\n" << std::endl;
        return false;
    }
}

size_t RasterizerMemoryArena::align_size(size_t size) const {
    return (size + config_.alignment - 1) & ~(config_.alignment - 1);
}

void RasterizerMemoryArena::record_allocation(uint64_t frame_id, const BufferHandle& handle) {
    std::lock_guard<std::mutex> lock(frame_mutex_);

    auto it = frame_contexts_.find(frame_id);
    if (it != frame_contexts_.end()) {
        it->second.buffers.push_back(handle);
        it->second.total_allocated += handle.size;
    }
}

RasterizerMemoryArena::Statistics RasterizerMemoryArena::get_statistics() const {
    Statistics stats;

    std::lock_guard<std::mutex> lock(arena_mutex_);

    for (const auto& [device, arena_ptr] : device_arenas_) {
        if (arena_ptr) {
            stats.current_usage += arena_ptr->offset.load(std::memory_order_relaxed);
            stats.peak_usage = std::max(stats.peak_usage,
                                       arena_ptr->peak_usage.load(std::memory_order_relaxed));
            stats.capacity += arena_ptr->capacity;
            stats.reallocation_count += arena_ptr->realloc_count.load(std::memory_order_relaxed);
        }
    }

    stats.frame_count = total_frames_processed_.load(std::memory_order_relaxed);
    stats.utilization_ratio = stats.capacity > 0 ?
        static_cast<float>(stats.current_usage) / static_cast<float>(stats.capacity) : 0.0f;

    return stats;
}

RasterizerMemoryArena::MemoryInfo RasterizerMemoryArena::get_memory_info() const {
    MemoryInfo info;

    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaSuccess) {
        std::lock_guard<std::mutex> lock(arena_mutex_);
        auto it = device_arenas_.find(device);
        if (it != device_arenas_.end() && it->second) {
            info.arena_capacity = it->second->capacity;
            info.current_usage = it->second->offset.load(std::memory_order_relaxed);
            info.peak_usage = it->second->peak_usage.load(std::memory_order_relaxed);
            info.num_reallocations = it->second->realloc_count.load(std::memory_order_relaxed);
            info.utilization_percent = info.arena_capacity > 0 ?
                (100.0f * info.peak_usage / info.arena_capacity) : 0.0f;
        }
    }

    cudaMemGetInfo(&info.gpu_free, &info.gpu_total);
    return info;
}

void RasterizerMemoryArena::dump_statistics() const {
    auto stats = get_statistics();

    std::stringstream ss;
    ss << "\n========================================\n"
       << "[RasterizerMemoryArena] FINAL STATISTICS\n"
       << "  Total capacity reserved: " << (stats.capacity >> 20) << " MB\n"
       << "  Peak usage: " << (stats.peak_usage >> 20) << " MB\n"
       << "  Frames processed: " << stats.frame_count << "\n"
       << "  Total reallocations: " << stats.reallocation_count << "\n"
       << "  Peak utilization: " << std::fixed << std::setprecision(1)
       << (stats.peak_usage * 100.0f / std::max(size_t(1), stats.capacity)) << "%\n";

    auto runtime = std::chrono::steady_clock::now() - creation_time_;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(runtime).count();
    ss << "  Runtime: " << seconds << " seconds\n";

    if (stats.frame_count > 0) {
        ss << "  Average time per frame: "
           << (seconds * 1000.0 / stats.frame_count) << " ms\n";
    }

    ss << "========================================\n";

    std::cout << ss.str() << std::flush;
}

bool RasterizerMemoryArena::is_under_memory_pressure() const {
    return get_memory_pressure() > 0.8f;
}

float RasterizerMemoryArena::get_memory_pressure() const {
    size_t free_memory, total_memory;
    cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
    if (err != cudaSuccess) {
        return 1.0f;
    }
    return 1.0f - (static_cast<float>(free_memory) / static_cast<float>(total_memory));
}

// Global singleton implementation
GlobalArenaManager& GlobalArenaManager::instance() {
    static GlobalArenaManager instance;
    return instance;
}

RasterizerMemoryArena& GlobalArenaManager::get_arena() {
    std::lock_guard<std::mutex> lock(init_mutex_);

    if (!arena_) {
        // Create with reasonable settings
        RasterizerMemoryArena::Config config;
        config.initial_size = 512 << 20;   // Start with 512MB
        config.max_size = 8ULL << 30;      // Allow up to 4GB
        config.growth_factor = 2.0f;       // Double when growing
        config.alignment = 256;
        config.enable_profiling = false;   // Set to true for even more verbose logging
        config.log_interval = 1000;        // Log every 1000 frames

        arena_ = std::make_unique<RasterizerMemoryArena>(config);
    }
    return *arena_;
}

void GlobalArenaManager::reset() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    arena_.reset();
}

} // namespace fast_gs::rasterization