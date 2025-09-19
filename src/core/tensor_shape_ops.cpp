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

// These are now the actual implementations called by movement ops
// We keep them here for the complex logic, but they're called through movement()

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

Tensor Tensor::t() const {
    if (!is_valid()) return {};
    
    if (shape_.rank() <= 1) {
        return clone(); // 0D and 1D tensors are unchanged
    }
    
    // Transpose last two dimensions using movement op
    return movement(MovementOp::Transpose, {std::pair<int, int>{-2, -1}});
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
