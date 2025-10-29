// vmm_allocator.cpp
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vmm_allocator.hpp"
#include <algorithm>
#include <sstream>

namespace lfs::core {

VMM_Allocator::VMM_Allocator() : VMM_Allocator(Config{}) {}

VMM_Allocator::VMM_Allocator(const Config& cfg)
    : config_(cfg) {
    if (!initialize()) {
        LOG_ERROR("VMM_Allocator: Failed to initialize");
    }
}

VMM_Allocator::~VMM_Allocator() {
    cleanup();
}

VMM_Allocator::VMM_Allocator(VMM_Allocator&& other) noexcept
    : virtual_base_(other.virtual_base_),
      virtual_size_(other.virtual_size_),
      granularity_(other.granularity_),
      all_chunks_(std::move(other.all_chunks_)),
      uncommitted_chunks_(std::move(other.uncommitted_chunks_)),
      offset_allocator_(std::move(other.offset_allocator_)),
      allocations_(std::move(other.allocations_)),
      config_(other.config_),
      device_(other.device_),
      is_initialized_(other.is_initialized_) {
    // Copy atomic stats
    stats_virtual_reserved_ = other.stats_virtual_reserved_.load();
    stats_physical_committed_ = other.stats_physical_committed_.load();
    stats_currently_allocated_ = other.stats_currently_allocated_.load();
    stats_peak_allocated_ = other.stats_peak_allocated_.load();
    stats_num_allocations_ = other.stats_num_allocations_.load();
    stats_num_deallocations_ = other.stats_num_deallocations_.load();
    stats_num_commits_ = other.stats_num_commits_.load();
    stats_num_decommits_ = other.stats_num_decommits_.load();
    stats_num_chunk_reuses_ = other.stats_num_chunk_reuses_.load();
    stats_fragmentation_events_ = other.stats_fragmentation_events_.load();

    other.virtual_base_ = 0;
    other.is_initialized_ = false;
}

VMM_Allocator& VMM_Allocator::operator=(VMM_Allocator&& other) noexcept {
    if (this != &other) {
        cleanup();

        virtual_base_ = other.virtual_base_;
        virtual_size_ = other.virtual_size_;
        granularity_ = other.granularity_;
        all_chunks_ = std::move(other.all_chunks_);
        uncommitted_chunks_ = std::move(other.uncommitted_chunks_);
        offset_allocator_ = std::move(other.offset_allocator_);
        allocations_ = std::move(other.allocations_);
        config_ = other.config_;
        device_ = other.device_;
        is_initialized_ = other.is_initialized_;

        // Copy atomic stats
        stats_virtual_reserved_ = other.stats_virtual_reserved_.load();
        stats_physical_committed_ = other.stats_physical_committed_.load();
        stats_currently_allocated_ = other.stats_currently_allocated_.load();
        stats_peak_allocated_ = other.stats_peak_allocated_.load();
        stats_num_allocations_ = other.stats_num_allocations_.load();
        stats_num_deallocations_ = other.stats_num_deallocations_.load();
        stats_num_commits_ = other.stats_num_commits_.load();
        stats_num_decommits_ = other.stats_num_decommits_.load();
        stats_num_chunk_reuses_ = other.stats_num_chunk_reuses_.load();
        stats_fragmentation_events_ = other.stats_fragmentation_events_.load();

        other.virtual_base_ = 0;
        other.is_initialized_ = false;
    }
    return *this;
}

bool VMM_Allocator::is_vmm_supported(int device) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        return false;
    }

    // Check CUDA version (need 11.2+)
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    if (driver_version < 11020) {  // 11.2 = 11020
        return false;
    }

    // Check if device supports VMM
    CUdevice cu_device;
    if (cuDeviceGet(&cu_device, device) != CUDA_SUCCESS) {
        return false;
    }

    int vmm_supported = 0;
    if (cuDeviceGetAttribute(&vmm_supported,
                             CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                             cu_device) != CUDA_SUCCESS) {
        return false;
    }

    return vmm_supported != 0;
}

bool VMM_Allocator::initialize() {
    cudaGetDevice(&device_);

    if (!is_vmm_supported(device_)) {
        LOG_WARN("VMM not supported on device {}, VMM_Allocator unavailable", device_);
        return false;
    }

    // Get granularity from device
    CUdevice cu_device;
    if (cuDeviceGet(&cu_device, device_) != CUDA_SUCCESS) {
        LOG_ERROR("Failed to get CUdevice");
        return false;
    }

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_;

    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
        LOG_ERROR("Failed to get allocation granularity");
        return false;
    }

    granularity_ = std::max(granularity, config_.commit_granularity);
    virtual_size_ = config_.virtual_size;

    LOG_INFO("VMM_Allocator: Granularity = {} MB", granularity_ / (1024 * 1024));

    // Reserve virtual address space (costs no RAM!)
    CUresult res = cuMemAddressReserve(&virtual_base_, virtual_size_, 0, 0, 0);
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        LOG_ERROR("Failed to reserve virtual address space: {} (code {})", err_str ? err_str : "unknown", static_cast<int>(res));
        return false;
    }

    stats_virtual_reserved_ = virtual_size_;

    // Initialize OffsetAllocator for virtual address space management
    // CRITICAL: OffsetAllocator works in GRANULARITY UNITS, not bytes!
    // This ensures all offsets are naturally aligned to granularity for cuMemMap
    uint32_t num_granules = static_cast<uint32_t>(virtual_size_ / granularity_);

    if (num_granules == 0 || virtual_size_ % granularity_ != 0) {
        LOG_ERROR("Virtual size {} must be multiple of granularity {}", virtual_size_, granularity_);
        cuMemAddressFree(virtual_base_, virtual_size_);
        return false;
    }

    offset_allocator_ = std::make_unique<OffsetAllocator::Allocator>(
        num_granules,          // Size in granularity units (e.g., 256 units of 16MB each)
        128 * 1024             // Support up to 128K concurrent allocations
    );

    is_initialized_ = true;
    LOG_INFO("VMM_Allocator initialized: {} GB virtual space on device {} (using OffsetAllocator)",
             virtual_size_ / (1024.0 * 1024.0 * 1024.0), device_);

    return true;
}

void VMM_Allocator::cleanup() {
    if (!is_initialized_) {
        return;
    }

    // Decommit all committed allocations
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    for (auto& [offset, metadata] : allocations_) {
        if (metadata.chunk) {
            cuMemUnmap(virtual_base_ + offset, metadata.chunk->size);
        }
    }

    // Release physical chunks
    for (auto& chunk : all_chunks_) {
        if (chunk->handle != 0) {
            cuMemRelease(chunk->handle);
        }
    }
    all_chunks_.clear();
    uncommitted_chunks_.clear();

    // Free virtual address space
    if (virtual_base_ != 0) {
        cuMemAddressFree(virtual_base_, virtual_size_);
        virtual_base_ = 0;
    }

    is_initialized_ = false;
}

void* VMM_Allocator::allocate(size_t bytes, cudaStream_t stream) {
    if (!is_initialized_) {
        LOG_ERROR("VMM_Allocator not initialized");
        return nullptr;
    }

    if (bytes == 0) {
        return nullptr;
    }

    // Align and round up to granularity
    size_t aligned_bytes = align_size(bytes);
    size_t rounded_bytes = round_up_to_granularity(aligned_bytes);

    std::lock_guard<std::mutex> lock(allocations_mutex_);

    // 1. Allocate virtual address range using OffsetAllocator (in granularity units)
    uint32_t num_granules_needed = static_cast<uint32_t>(rounded_bytes / granularity_);
    if (num_granules_needed == 0) num_granules_needed = 1;  // Minimum 1 granule

    OffsetAllocator::Allocation alloc = offset_allocator_->allocate(num_granules_needed);
    if (alloc.offset == OffsetAllocator::Allocation::NO_SPACE) {
        LOG_ERROR("OffsetAllocator failed to find virtual range for {} granules ({} bytes)",
                  num_granules_needed, rounded_bytes);
        stats_fragmentation_events_++;
        return nullptr;
    }

    // Convert granule offset to byte offset
    size_t byte_offset = static_cast<size_t>(alloc.offset) * granularity_;

    // 2. Create new physical chunk for this allocation
    // NOTE: VMM doesn't support reusing the same physical chunk at different virtual addresses
    // The benefit comes from reusing the VIRTUAL address, not the physical chunk
    PhysicalChunk* chunk = create_physical_chunk(rounded_bytes);
    if (!chunk) {
        LOG_ERROR("Failed to create physical chunk for {} bytes", rounded_bytes);
        offset_allocator_->free(alloc);  // Return virtual range
        return nullptr;
    }

    // 4. Commit physical memory to virtual address
    if (!commit_physical_to_virtual(byte_offset, rounded_bytes, chunk)) {
        LOG_ERROR("Failed to commit physical to virtual at offset {}", byte_offset);
        offset_allocator_->free(alloc);  // Return virtual range
        return nullptr;
    }

    // 5. Store allocation metadata
    AllocationMetadata metadata;
    metadata.chunk = chunk;
    metadata.allocated_size = bytes;  // Store original requested size
    metadata.alloc = alloc;
    allocations_[byte_offset] = metadata;  // Index by byte offset

    // 6. Update statistics
    update_statistics(bytes, true);

    void* ptr = (void*)(virtual_base_ + byte_offset);

    if (config_.enable_statistics) {
        LOG_DEBUG("VMM alloc: {} bytes at offset {} (ptr={}, chunk_size={})",
                  bytes, byte_offset, ptr, chunk->size);
    }

    return ptr;
}

void VMM_Allocator::deallocate(void* ptr, cudaStream_t stream) {
    if (!is_initialized_ || !ptr) {
        return;
    }

    size_t byte_offset = (char*)ptr - (char*)virtual_base_;

    std::lock_guard<std::mutex> lock(allocations_mutex_);

    // Find the allocation
    auto it = allocations_.find(byte_offset);
    if (it == allocations_.end()) {
        LOG_ERROR("VMM dealloc: Invalid pointer {} (offset {})", ptr, byte_offset);
        return;
    }

    AllocationMetadata& metadata = it->second;
    PhysicalChunk* chunk = metadata.chunk;
    size_t allocated_size = metadata.allocated_size;

    // Decide whether to decommit based on memory pressure
    bool should_decommit = config_.auto_decommit;
    if (!should_decommit && get_memory_pressure() > config_.decommit_threshold) {
        should_decommit = true;
    }

    // CRITICAL: Synchronize CUDA operations before releasing memory!
    // Without this, CUDA operations may still be using this memory when we decommit/release it.
    // This prevents use-after-free bugs that cause "illegal memory access" errors.
    CUresult sync_result = cuCtxSynchronize();
    if (sync_result != CUDA_SUCCESS) {
        const char* err_name;
        const char* err_string;
        cuGetErrorName(sync_result, &err_name);
        cuGetErrorString(sync_result, &err_string);
        LOG_WARN("cuCtxSynchronize failed in VMM deallocate: {} - {}",
                 err_name, err_string);
        // Continue anyway - better to potentially reuse memory than leak it
    }

    // Always decommit and release physical memory (VMM doesn't support chunk reuse)
    // The benefit of VMM is reusing the VIRTUAL address, not the physical chunk
    decommit_physical_from_virtual(byte_offset, chunk->size, chunk);

    // Release the physical chunk
    if (chunk->handle != 0) {
        cuMemRelease(chunk->handle);
        chunk->handle = 0;
    }

    stats_physical_committed_ -= chunk->size;
    stats_num_decommits_++;

    // Free virtual address range using OffsetAllocator
    offset_allocator_->free(metadata.alloc);

    // Remove allocation metadata
    allocations_.erase(it);

    // Update statistics
    update_statistics(allocated_size, false);

    if (config_.enable_statistics) {
        LOG_DEBUG("VMM dealloc: {} bytes at offset {} (decommit={})",
                  allocated_size, byte_offset, should_decommit);
    }
}

VMM_Allocator::PhysicalChunk* VMM_Allocator::create_physical_chunk(size_t size) {
    // Check if we've hit the max physical memory limit
    if (stats_physical_committed_ + size > config_.max_physical) {
        LOG_WARN("Would exceed max physical memory: {} + {} > {}",
                 stats_physical_committed_.load(), size, config_.max_physical);
        return nullptr;
    }

    auto chunk = std::make_unique<PhysicalChunk>();
    chunk->size = size;
    chunk->granularity = granularity_;
    chunk->device = device_;
    chunk->is_uncommitted = false;

    // Create physical memory handle
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_;

    CUresult res = cuMemCreate(&chunk->handle, size, &prop, 0);
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        LOG_ERROR("Failed to create physical memory: {} (code {})", err_str ? err_str : "unknown", static_cast<int>(res));
        return nullptr;
    }

    stats_physical_committed_ += size;
    stats_num_commits_++;

    PhysicalChunk* raw_ptr = chunk.get();
    all_chunks_.push_back(std::move(chunk));

    LOG_DEBUG("Created physical chunk: {} bytes, handle={}", size, raw_ptr->handle);

    return raw_ptr;
}

VMM_Allocator::PhysicalChunk* VMM_Allocator::find_uncommitted_chunk(size_t size) {
    // Find best-fit uncommitted chunk
    PhysicalChunk* best = nullptr;
    size_t best_size = SIZE_MAX;

    for (auto* chunk : uncommitted_chunks_) {
        if (chunk->size >= size && chunk->size < best_size) {
            best = chunk;
            best_size = chunk->size;
            if (chunk->size == size) {
                break;  // Exact match, stop searching
            }
        }
    }

    if (best) {
        uncommitted_chunks_.erase(best);
        best->is_uncommitted = false;
        stats_physical_committed_ += best->size;  // Re-committing
    }

    return best;
}

bool VMM_Allocator::commit_physical_to_virtual(size_t offset, size_t size, PhysicalChunk* chunk) {
    CUdeviceptr va = virtual_base_ + offset;

    // CRITICAL: Must map the ENTIRE physical chunk size, not the requested size!
    // CUDA VMM doesn't allow partial mapping of a physical handle
    size_t map_size = chunk->size;

    if (size > chunk->size) {
        LOG_ERROR("Requested size {} exceeds chunk size {}", size, chunk->size);
        return false;
    }

    // Map physical memory to virtual address
    CUresult res = cuMemMap(va, map_size, 0, chunk->handle, 0);
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        LOG_ERROR("Failed to map physical to virtual: {} (code {})", err_str ? err_str : "unknown", static_cast<int>(res));
        LOG_ERROR("  va={:#x}, map_size={}, chunk->size={}, chunk->handle={}",
                  static_cast<unsigned long long>(va), map_size, chunk->size, chunk->handle);
        return false;
    }

    // Set access permissions (must match the mapped size)
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    res = cuMemSetAccess(va, map_size, &access, 1);
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        LOG_ERROR("Failed to set access permissions: {} (code {})", err_str ? err_str : "unknown", static_cast<int>(res));
        cuMemUnmap(va, map_size);
        return false;
    }

    return true;
}

void VMM_Allocator::decommit_physical_from_virtual(size_t offset, size_t size, PhysicalChunk* chunk) {
    CUdeviceptr va = virtual_base_ + offset;

    // CRITICAL: Must unmap the ENTIRE physical chunk size, not just the requested size!
    // This must match what we mapped in commit_physical_to_virtual
    size_t unmap_size = chunk->size;

    // Unmap physical memory from virtual address
    CUresult res = cuMemUnmap(va, unmap_size);
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        LOG_WARN("Failed to unmap virtual memory: {} (code {})", err_str ? err_str : "unknown", static_cast<int>(res));
        LOG_WARN("  va={:#x}, unmap_size={}, chunk->size={}",
                 static_cast<unsigned long long>(va), unmap_size, chunk->size);
    }

    // Physical memory is now returned to OS!
    // But handle remains valid for re-mapping later
}

size_t VMM_Allocator::align_size(size_t size) const {
    return (size + config_.alignment - 1) & ~(config_.alignment - 1);
}

size_t VMM_Allocator::round_up_to_granularity(size_t size) const {
    return (size + granularity_ - 1) & ~(granularity_ - 1);
}

void VMM_Allocator::update_statistics(size_t size, bool is_alloc) {
    if (is_alloc) {
        stats_currently_allocated_ += size;
        stats_num_allocations_++;

        size_t current = stats_currently_allocated_.load();
        size_t peak = stats_peak_allocated_.load();
        while (current > peak && !stats_peak_allocated_.compare_exchange_weak(peak, current)) {
            // Retry if another thread updated peak
        }
    } else {
        stats_currently_allocated_ -= size;
        stats_num_deallocations_++;
    }
}

float VMM_Allocator::get_memory_pressure() const {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return 1.0f - (float)free / total;
}

void VMM_Allocator::trim_unused_memory() {
    std::lock_guard<std::mutex> lock(allocations_mutex_);

    size_t num_freed = uncommitted_chunks_.size();

    // Release all uncommitted chunks
    for (auto* chunk : uncommitted_chunks_) {
        if (chunk->handle != 0) {
            cuMemRelease(chunk->handle);
            chunk->handle = 0;
            stats_physical_committed_ -= chunk->size;
        }
    }
    uncommitted_chunks_.clear();

    LOG_INFO("VMM trim: Freed {} uncommitted chunks", num_freed);
}

VMM_Allocator::Statistics VMM_Allocator::get_statistics() const {
    Statistics stats;
    stats.virtual_reserved = stats_virtual_reserved_.load();
    stats.physical_committed = stats_physical_committed_.load();
    stats.currently_allocated = stats_currently_allocated_.load();
    stats.peak_allocated = stats_peak_allocated_.load();
    stats.num_allocations = stats_num_allocations_.load();
    stats.num_deallocations = stats_num_deallocations_.load();
    stats.num_commits = stats_num_commits_.load();
    stats.num_decommits = stats_num_decommits_.load();
    stats.num_chunk_reuses = stats_num_chunk_reuses_.load();
    stats.fragmentation_events = stats_fragmentation_events_.load();
    return stats;
}

VMM_Allocator::MemoryInfo VMM_Allocator::get_memory_info() const {
    MemoryInfo info;
    info.virtual_reserved = stats_virtual_reserved_.load();
    info.physical_committed = stats_physical_committed_.load();
    info.currently_allocated = stats_currently_allocated_.load();
    info.peak_allocated = stats_peak_allocated_.load();

    cudaMemGetInfo(&info.gpu_free, &info.gpu_total);

    if (info.physical_committed > 0) {
        info.utilization_percent = 100.0f * info.currently_allocated / info.physical_committed;
        info.fragmentation_ratio = (float)(info.physical_committed - info.currently_allocated) / info.physical_committed;
    }

    return info;
}

void VMM_Allocator::dump_statistics() const {
    auto info = get_memory_info();

    LOG_INFO("VMM_Allocator Statistics:");
    LOG_INFO("  Virtual reserved:     {:.2f} GB", info.virtual_reserved / (1024.0 * 1024.0 * 1024.0));
    LOG_INFO("  Physical committed:   {:.2f} GB", info.physical_committed / (1024.0 * 1024.0 * 1024.0));
    LOG_INFO("  Currently allocated:  {:.2f} GB", info.currently_allocated / (1024.0 * 1024.0 * 1024.0));
    LOG_INFO("  Peak allocated:       {:.2f} GB", info.peak_allocated / (1024.0 * 1024.0 * 1024.0));
    LOG_INFO("  Utilization:          {:.1f}%", info.utilization_percent);
    LOG_INFO("  Fragmentation ratio:  {:.1f}%", info.fragmentation_ratio * 100.0f);
    LOG_INFO("  GPU free:             {:.2f} GB", info.gpu_free / (1024.0 * 1024.0 * 1024.0));
    LOG_INFO("  GPU total:            {:.2f} GB", info.gpu_total / (1024.0 * 1024.0 * 1024.0));
    LOG_INFO("  Total allocations:    {}", stats_num_allocations_.load());
    LOG_INFO("  Total deallocations:  {}", stats_num_deallocations_.load());
    LOG_INFO("  Commits:              {}", stats_num_commits_.load());
    LOG_INFO("  Decommits:            {}", stats_num_decommits_.load());
    LOG_INFO("  Chunk reuses:         {}", stats_num_chunk_reuses_.load());
    LOG_INFO("  Fragmentation events: {}", stats_fragmentation_events_.load());
}

} // namespace lfs::core
