/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <functional>
#include <chrono>

namespace fast_gs::rasterization {

class RasterizerMemoryArena {
public:
    struct Config {
        size_t initial_size = 1ULL << 30;  // 1GB
        size_t max_size = 4ULL << 30;      // 4GB
        float growth_factor = 1.5f;
        size_t alignment = 256;
        bool enable_profiling = false;
        size_t log_interval = 1000;        // Log every N frames
    };

    struct BufferHandle {
        char* ptr = nullptr;
        size_t size = 0;
        uint64_t generation = 0;
        int device = -1;
    };

    struct FrameContext {
        std::vector<BufferHandle> buffers;
        uint64_t frame_id = 0;
        uint64_t generation = 0;
        size_t total_allocated = 0;
        bool is_active = false;
        std::chrono::steady_clock::time_point timestamp;
    };

    struct Statistics {
        size_t current_usage = 0;
        size_t peak_usage = 0;
        size_t capacity = 0;
        size_t reallocation_count = 0;
        size_t frame_count = 0;
        float utilization_ratio = 0.0f;
    };

    struct MemoryInfo {
        size_t arena_capacity = 0;
        size_t current_usage = 0;
        size_t peak_usage = 0;
        size_t gpu_free = 0;
        size_t gpu_total = 0;
        size_t num_reallocations = 0;
        float utilization_percent = 0.0f;
    };

private:
    struct Arena {
        torch::Tensor buffer;
        std::atomic<size_t> offset{0};
        size_t capacity = 0;
        uint64_t generation = 0;
        int device = -1;

        // Statistics
        std::atomic<size_t> peak_usage{0};
        std::atomic<size_t> peak_usage_period{0};  // Peak since last log
        std::atomic<size_t> total_allocated{0};
        std::atomic<size_t> realloc_count{0};
        std::chrono::steady_clock::time_point last_log_time;
    };

    std::unordered_map<int, std::unique_ptr<Arena>> device_arenas_;
    std::unordered_map<uint64_t, FrameContext> frame_contexts_;
    Config config_;

    mutable std::mutex arena_mutex_;
    mutable std::mutex frame_mutex_;
    std::atomic<uint64_t> frame_counter_{0};
    std::atomic<uint64_t> generation_counter_{0};

    // Performance tracking
    std::chrono::steady_clock::time_point creation_time_;
    std::atomic<size_t> total_frames_processed_{0};

public:
    // Constructors
    RasterizerMemoryArena();
    explicit RasterizerMemoryArena(const Config& cfg);
    ~RasterizerMemoryArena();

    // Delete copy operations
    RasterizerMemoryArena(const RasterizerMemoryArena&) = delete;
    RasterizerMemoryArena& operator=(const RasterizerMemoryArena&) = delete;

    // Allow move operations
    RasterizerMemoryArena(RasterizerMemoryArena&&) noexcept;
    RasterizerMemoryArena& operator=(RasterizerMemoryArena&&) noexcept;

    // Frame-based allocation
    uint64_t begin_frame();
    void end_frame(uint64_t frame_id);

    // Get allocator for specific frame
    std::function<char*(size_t)> get_allocator(uint64_t frame_id);

    // Get existing buffers for backward pass
    std::vector<BufferHandle> get_frame_buffers(uint64_t frame_id) const;

    // Reset frame for reuse (keeps allocation but resets offset)
    void reset_frame(uint64_t frame_id);

    // Cleanup old frames
    void cleanup_frames(int keep_recent = 3);

    // Emergency cleanup
    void emergency_cleanup();

    // Statistics
    Statistics get_statistics() const;
    MemoryInfo get_memory_info() const;
    void dump_statistics() const;
    void log_memory_status(uint64_t frame_id, bool force = false);

    // Memory pressure management
    bool is_under_memory_pressure() const;
    float get_memory_pressure() const;

private:
    Arena& get_or_create_arena(int device);
    char* allocate_internal(Arena& arena, size_t size, uint64_t frame_id);
    bool grow_arena(Arena& arena, size_t required_size);
    size_t align_size(size_t size) const;
    void record_allocation(uint64_t frame_id, const BufferHandle& handle);
};

// Global singleton for arena access
class GlobalArenaManager {
public:
    static GlobalArenaManager& instance();
    RasterizerMemoryArena& get_arena();
    void reset();

private:
    GlobalArenaManager() = default;
    ~GlobalArenaManager() = default;
    std::unique_ptr<RasterizerMemoryArena> arena_;
    std::mutex init_mutex_;
};

} // namespace fast_gs::rasterization