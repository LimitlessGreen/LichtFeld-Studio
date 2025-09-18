/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include "core/tensor_broadcast.hpp"
#include <algorithm>
#include <numeric>
#include <execution>

#define CHECK_CUDA(call) do { \
    if (auto e = call; e != cudaSuccess) { \
        LOG_ERROR("CUDA error: {}", cudaGetErrorString(e)); \
    } \
} while(0)

namespace gs {

// Helper to resolve multiple dimensions
std::vector<size_t> Tensor::resolve_dims(std::span<const int> dims) const {
    std::vector<size_t> resolved;
    resolved.reserve(dims.size());
    
    for (int dim : dims) {
        int r = resolve_dim(dim);
        if (r < 0 || r >= static_cast<int>(shape_.rank())) {
            LOG_ERROR("Dimension {} out of range for tensor with {} dimensions", dim, shape_.rank());
            return {};
        }
        resolved.push_back(static_cast<size_t>(r));
    }
    
    return resolved;
}

// Core reshape with -1 inference
Tensor Tensor::reshape(std::span<const int> sizes) const {
    if (!is_valid()) return {};
    
    std::vector<size_t> new_shape;
    int infer_idx = -1;
    size_t known_size = 1;
    
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == -1) {
            if (infer_idx != -1) {
                LOG_ERROR("Can only infer one dimension");
                return {};
            }
            infer_idx = static_cast<int>(i);
            new_shape.push_back(0); // Placeholder
        } else if (sizes[i] > 0) {
            new_shape.push_back(static_cast<size_t>(sizes[i]));
            known_size *= sizes[i];
        } else {
            LOG_ERROR("Invalid reshape size: {}", sizes[i]);
            return {};
        }
    }
    
    // Infer dimension if needed
    if (infer_idx != -1) {
        if (numel() % known_size != 0) {
            LOG_ERROR("Cannot infer dimension: {} not divisible by {}", 
                     numel(), known_size);
            return {};
        }
        new_shape[infer_idx] = numel() / known_size;
    }
    
    TensorShape target(new_shape);
    if (target.elements() != numel()) {
        LOG_ERROR("Reshape size mismatch: {} != {}", 
                 target.elements(), numel());
        return {};
    }
    
    // Return a view - no data copy
    Tensor result(data_, target, device_, dtype_);
    result.initialized_ = true;
    return result;
}

// Overload for TensorShape
Tensor Tensor::reshape(TensorShape new_shape) const {
    if (!is_valid()) return {};
    
    if (new_shape.elements() != numel()) {
        LOG_ERROR("View shape {} has {} elements, but tensor has {} elements",
                  new_shape.str(), new_shape.elements(), shape_.elements());
        return {};
    }
    
    Tensor result(data_, new_shape, device_, dtype_);
    result.initialized_ = true;
    return result;
}

// Tinygrad-style squeeze
Tensor Tensor::squeeze(std::optional<int> dim) const {
    if (!is_valid()) return {};
    
    std::vector<size_t> new_dims;
    
    if (!dim.has_value()) {
        // Remove ALL size-1 dimensions
        for (size_t s : shape_.dims()) {
            if (s != 1) new_dims.push_back(s);
        }
        // Edge case: scalar (all dims were 1)
        if (new_dims.empty()) new_dims.push_back(1);
    } else {
        // Remove specific dimension if it's size 1
        int d = resolve_dim(dim.value());
        if (d < 0 || d >= static_cast<int>(shape_.rank())) {
            return clone(); // Return copy if dim out of range
        }
        
        if (shape_[d] != 1) {
            return clone(); // No-op if can't squeeze
        }
        
        for (int i = 0; i < static_cast<int>(shape_.rank()); ++i) {
            if (i != d) new_dims.push_back(shape_[i]);
        }
    }
    
    return reshape(TensorShape(new_dims));
}

// Unsqueeze at specific dimension
Tensor Tensor::unsqueeze(int dim) const {
    if (!is_valid()) return {};
    
    // Allow inserting at end (rank + 1 positions)
    dim = (dim < 0) ? static_cast<int>(shape_.rank()) + dim + 1 : dim;
    if (dim < 0 || dim > static_cast<int>(shape_.rank())) {
        LOG_ERROR("Invalid unsqueeze dimension: {}", dim);
        return {};
    }
    
    std::vector<size_t> new_dims = shape_.dims();
    new_dims.insert(new_dims.begin() + dim, 1);
    return reshape(TensorShape(new_dims));
}

// Tinygrad-style expand
Tensor Tensor::expand(std::span<const int> sizes) const {
    if (!is_valid()) return {};
    
    // Build target shape, -1 means keep original
    std::vector<size_t> target_shape;
    
    // Pad with 1s if needed to match sizes length
    size_t rank_diff = 0;
    if (sizes.size() > shape_.rank()) {
        rank_diff = sizes.size() - shape_.rank();
    }
    
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == -1) {
            // Keep original dimension if it exists
            if (i >= rank_diff && (i - rank_diff) < shape_.rank()) {
                target_shape.push_back(shape_[i - rank_diff]);
            } else {
                LOG_ERROR("Cannot infer dimension at position {}", i);
                return {};
            }
        } else if (sizes[i] > 0) {
            // Verify broadcast compatibility
            if (i >= rank_diff && (i - rank_diff) < shape_.rank()) {
                size_t orig_dim = shape_[i - rank_diff];
                if (orig_dim != 1 && orig_dim != static_cast<size_t>(sizes[i])) {
                    LOG_ERROR("Cannot expand size {} to {} at dimension {}", 
                             orig_dim, sizes[i], i);
                    return {};
                }
            }
            target_shape.push_back(static_cast<size_t>(sizes[i]));
        } else {
            LOG_ERROR("Invalid expand size: {}", sizes[i]);
            return {};
        }
    }
    
    return broadcast_to(TensorShape(target_shape));
}

// Overload for TensorShape
Tensor Tensor::expand(const TensorShape& target_shape) const {
    return broadcast_to(target_shape);
}

// Clean flatten implementation
Tensor Tensor::flatten(int start_dim, int end_dim) const {
    if (!is_valid()) return {};
    
    start_dim = resolve_dim(start_dim);
    end_dim = (end_dim == -1) ? static_cast<int>(shape_.rank()) - 1 : resolve_dim(end_dim);
    
    if (start_dim < 0 || end_dim >= static_cast<int>(shape_.rank()) || start_dim > end_dim) {
        LOG_ERROR("Invalid flatten range: [{}, {}] for tensor with {} dimensions", 
                 start_dim, end_dim, shape_.rank());
        return {};
    }
    
    std::vector<size_t> new_shape;
    
    // Keep dims before start
    for (int i = 0; i < start_dim; ++i) {
        new_shape.push_back(shape_[i]);
    }
    
    // Flatten range
    size_t flattened = 1;
    for (int i = start_dim; i <= end_dim; ++i) {
        flattened *= shape_[i];
    }
    new_shape.push_back(flattened);
    
    // Keep dims after end
    for (size_t i = end_dim + 1; i < shape_.rank(); ++i) {
        new_shape.push_back(shape_[i]);
    }
    
    return reshape(TensorShape(new_shape));
}

// Permute implementation
Tensor Tensor::permute(std::span<const int> axes) const {
    if (!is_valid()) return {};
    
    if (axes.size() != shape_.rank()) {
        LOG_ERROR("Permute requires {} axes, got {}", shape_.rank(), axes.size());
        return {};
    }
    
    // Validate and resolve axes
    std::vector<int> resolved_axes;
    std::vector<bool> used(shape_.rank(), false);
    
    for (int axis : axes) {
        int resolved = resolve_dim(axis);
        if (resolved < 0 || resolved >= static_cast<int>(shape_.rank())) {
            LOG_ERROR("Invalid permute axis: {}", axis);
            return {};
        }
        if (used[resolved]) {
            LOG_ERROR("Duplicate axis in permute: {}", axis);
            return {};
        }
        used[resolved] = true;
        resolved_axes.push_back(resolved);
    }
    
    // Build new shape
    std::vector<size_t> new_shape;
    for (int axis : resolved_axes) {
        new_shape.push_back(shape_[axis]);
    }
    
    // Check for simple cases
    bool is_identity = true;
    for (size_t i = 0; i < resolved_axes.size(); ++i) {
        if (resolved_axes[i] != static_cast<int>(i)) {
            is_identity = false;
            break;
        }
    }
    
    if (is_identity) {
        return clone(); // No permutation needed
    }
    
    // Special case: 2D transpose
    if (shape_.rank() == 2 && resolved_axes[0] == 1 && resolved_axes[1] == 0) {
        auto result = empty(TensorShape(new_shape), device_, dtype_);
        
        if (device_ == Device::CUDA) {
            tensor_ops::launch_transpose(ptr<float>(), result.ptr<float>(),
                                        shape_[0], shape_[1], 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();
            
            for (size_t i = 0; i < shape_[0]; ++i) {
                for (size_t j = 0; j < shape_[1]; ++j) {
                    dst[j * shape_[0] + i] = src[i * shape_[1] + j];
                }
            }
        }
        
        return result;
    }
    
    // General permute
    auto result = empty(TensorShape(new_shape), device_, dtype_);
    
    if (device_ == Device::CUDA) {
        // For now, do permute on CPU and copy back
        LOG_WARN("General permute not optimized for CUDA");
        auto cpu_copy = to(Device::CPU);
        auto cpu_result = cpu_copy.permute(axes);
        return cpu_result.to(Device::CUDA);
    } else {
        // CPU implementation
        const float* src = ptr<float>();
        float* dst = result.ptr<float>();
        
        // Calculate strides for source
        std::vector<size_t> src_strides(shape_.rank());
        src_strides.back() = 1;
        for (int i = static_cast<int>(shape_.rank()) - 2; i >= 0; --i) {
            src_strides[i] = src_strides[i + 1] * shape_[i + 1];
        }
        
        // Calculate strides for destination
        std::vector<size_t> dst_strides(new_shape.size());
        dst_strides.back() = 1;
        for (int i = static_cast<int>(new_shape.size()) - 2; i >= 0; --i) {
            dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];
        }
        
        // Permute by iterating through all elements
        size_t n = numel();
        
        #pragma omp parallel for if(n > 10000)
        for (size_t linear_idx = 0; linear_idx < n; ++linear_idx) {
            // Convert linear index to multi-dimensional indices in source
            std::vector<size_t> src_indices(shape_.rank());
            size_t temp = linear_idx;
            for (size_t d = 0; d < shape_.rank(); ++d) {
                src_indices[d] = temp / src_strides[d];
                temp %= src_strides[d];
            }
            
            // Permute indices and calculate destination linear index
            size_t dst_idx = 0;
            for (size_t d = 0; d < resolved_axes.size(); ++d) {
                dst_idx += src_indices[resolved_axes[d]] * dst_strides[d];
            }
            
            dst[dst_idx] = src[linear_idx];
        }
    }
    
    return result;
}

// Transpose using permute
Tensor Tensor::transpose(int dim1, int dim2) const {
    if (!is_valid()) return {};
    
    if (shape_.rank() < 2) {
        LOG_ERROR("Transpose requires at least 2D tensor");
        return {};
    }
    
    dim1 = resolve_dim(dim1);
    dim2 = resolve_dim(dim2);
    
    if (dim1 < 0 || dim1 >= static_cast<int>(shape_.rank()) ||
        dim2 < 0 || dim2 >= static_cast<int>(shape_.rank())) {
        LOG_ERROR("Transpose dimensions out of bounds");
        return {};
    }
    
    if (dim1 == dim2) {
        return clone(); // No-op
    }
    
    // Build permutation
    std::vector<int> perm;
    for (int i = 0; i < static_cast<int>(shape_.rank()); ++i) {
        if (i == dim1) {
            perm.push_back(dim2);
        } else if (i == dim2) {
            perm.push_back(dim1);
        } else {
            perm.push_back(i);
        }
    }
    
    return permute(perm);
}

// Simple transpose for 2D tensors
Tensor Tensor::t() const {
    if (!is_valid()) return {};
    
    if (shape_.rank() <= 1) {
        return clone(); // 0D and 1D tensors are unchanged
    }
    
    // Transpose last two dimensions
    return transpose(-2, -1);
}

// Helper: check if slice is contiguous
bool Tensor::is_contiguous_slice(const std::vector<size_t>& starts,
                                 const std::vector<size_t>& ends) const {
    // A slice is contiguous if we're only slicing the first dimensions
    // and taking full slices of the remaining dimensions
    bool found_partial = false;
    
    for (size_t i = 0; i < shape_.rank(); ++i) {
        if (starts[i] != 0 || ends[i] != shape_[i]) {
            found_partial = true;
        } else if (found_partial) {
            // Found a full slice after a partial slice - not contiguous
            return false;
        }
    }
    
    return true;
}

// Helper: calculate linear offset
size_t Tensor::calculate_offset(const std::vector<size_t>& indices) const {
    size_t offset = 0;
    size_t stride = 1;
    
    for (int i = static_cast<int>(shape_.rank()) - 1; i >= 0; --i) {
        offset += indices[i] * stride;
        stride *= shape_[i];
    }
    
    return offset;
}

// Helper: copy non-contiguous slice
Tensor Tensor::copy_slice(const std::vector<size_t>& starts,
                          const std::vector<size_t>& ends,
                          const std::vector<size_t>& new_shape) const {
    auto result = empty(TensorShape(new_shape), device_, dtype_);
    
    if (device_ == Device::CUDA) {
        // For now, do slicing on CPU
        auto cpu_copy = to(Device::CPU);
        auto cpu_result = cpu_copy.copy_slice(starts, ends, new_shape);
        return cpu_result.to(Device::CUDA);
    } else {
        // CPU implementation
        const float* src = ptr<float>();
        float* dst = result.ptr<float>();
        
        // Calculate total elements in slice
        size_t total = 1;
        for (size_t s : new_shape) {
            total *= s;
        }
        
        // Iterate through slice
        std::vector<size_t> indices(shape_.rank());
        for (size_t i = 0; i < shape_.rank(); ++i) {
            indices[i] = starts[i];
        }
        
        for (size_t dst_idx = 0; dst_idx < total; ++dst_idx) {
            // Calculate source index
            size_t src_idx = calculate_offset(indices);
            dst[dst_idx] = src[src_idx];
            
            // Increment indices
            for (int d = static_cast<int>(shape_.rank()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < ends[d]) {
                    break;
                }
                indices[d] = starts[d];
            }
        }
    }
    
    return result;
}

// Better slice API
Tensor Tensor::slice(std::span<const std::pair<int, int>> ranges) const {
    if (!is_valid()) return {};
    
    if (ranges.size() > shape_.rank()) {
        LOG_ERROR("Too many slice ranges for tensor rank");
        return {};
    }
    
    std::vector<size_t> starts, ends;
    std::vector<size_t> new_shape;
    
    for (size_t i = 0; i < shape_.rank(); ++i) {
        int start = 0, end = static_cast<int>(shape_[i]);
        
        if (i < ranges.size()) {
            start = ranges[i].first;
            end = ranges[i].second;
            
            // Handle negative indices
            if (start < 0) start += static_cast<int>(shape_[i]);
            if (end < 0) end += static_cast<int>(shape_[i]);
            
            // Clamp to valid range
            start = std::clamp(start, 0, static_cast<int>(shape_[i]));
            end = std::clamp(end, start, static_cast<int>(shape_[i]));
        }
        
        starts.push_back(start);
        ends.push_back(end);
        new_shape.push_back(end - start);
    }
    
    // Check for empty slice
    for (size_t s : new_shape) {
        if (s == 0) {
            return empty(TensorShape(new_shape), device_, dtype_);
        }
    }
    
    // For contiguous slices, we can return a view
    if (is_contiguous_slice(starts, ends)) {
        size_t offset = calculate_offset(starts);
        void* new_data = static_cast<char*>(data_) + offset * dtype_size(dtype_);
        Tensor result(new_data, TensorShape(new_shape), device_, dtype_);
        result.initialized_ = true;
        return result;
    }
    
    // Non-contiguous slice requires copy
    return copy_slice(starts, ends, new_shape);
}

// Single-dimension slice (compatibility)
Tensor Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (!is_valid()) return {};
    
    if (dim >= shape_.rank()) {
        LOG_ERROR("Slice dimension {} out of range for rank {}", dim, shape_.rank());
        return {};
    }
    
    if (start >= end || end > shape_[dim]) {
        LOG_ERROR("Invalid slice range [{}, {}) for dimension {} of size {}",
                  start, end, dim, shape_[dim]);
        return {};
    }
    
    // Build ranges for all dimensions
    std::vector<std::pair<int, int>> ranges;
    for (size_t i = 0; i < shape_.rank(); ++i) {
        if (i == dim) {
            ranges.push_back({static_cast<int>(start), static_cast<int>(end)});
        } else {
            ranges.push_back({0, static_cast<int>(shape_[i])});
        }
    }
    
    return slice(ranges);
}

#undef CHECK_CUDA

} // namespace gs
