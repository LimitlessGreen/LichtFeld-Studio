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

// These are helper implementations that movement ops use

Tensor Tensor::reshape(TensorShape new_shape) const {
    if (!is_valid()) return {};

    if (new_shape.elements() != numel()) {
        LOG_ERROR("View shape {} has {} elements, but tensor has {} elements",
                  new_shape.str(), new_shape.elements(), shape_.elements());
        return {};
    }

    // Create a view that shares memory (non-owning)
    return Tensor(data_, new_shape, device_, dtype_);
}

Tensor Tensor::t() const {
    if (!is_valid()) return {};

    if (shape_.rank() <= 1) {
        return clone(); // 0D and 1D tensors are unchanged
    }

    // Transpose last two dimensions
    return transpose(-2, -1);
}

// Implementation of permute (was inline in header, now moved here)
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
            LOG_ERROR("Duplicate permute axis: {}", axis);
            return {};
        }
        used[resolved] = true;
        resolved_axes.push_back(resolved);
    }

    // Create new shape
    std::vector<size_t> new_dims(shape_.rank());
    for (size_t i = 0; i < shape_.rank(); ++i) {
        new_dims[i] = shape_[resolved_axes[i]];
    }

    auto result = empty(TensorShape(new_dims), device_, dtype_);

    if (device_ == Device::CUDA) {
        // For now, do permutation on CPU
        auto cpu_copy = to(Device::CPU);
        auto cpu_result = cpu_copy.permute(axes);
        return cpu_result.to(Device::CUDA);
    } else {
        // CPU implementation
        const float* src = ptr<float>();
        float* dst = result.ptr<float>();

        // Calculate strides for source and destination
        std::vector<size_t> src_strides(shape_.rank());
        std::vector<size_t> dst_strides(shape_.rank());

        src_strides.back() = 1;
        dst_strides.back() = 1;

        for (int i = shape_.rank() - 2; i >= 0; --i) {
            src_strides[i] = src_strides[i + 1] * shape_[i + 1];
            dst_strides[i] = dst_strides[i + 1] * new_dims[i + 1];
        }

        // Perform permutation
        for (size_t dst_idx = 0; dst_idx < result.numel(); ++dst_idx) {
            // Convert destination index to coordinates
            std::vector<size_t> dst_coords(shape_.rank());
            size_t temp = dst_idx;
            for (size_t i = 0; i < shape_.rank(); ++i) {
                dst_coords[i] = temp / dst_strides[i];
                temp %= dst_strides[i];
            }

            // Map to source coordinates using permutation
            size_t src_idx = 0;
            for (size_t i = 0; i < shape_.rank(); ++i) {
                src_idx += dst_coords[i] * src_strides[resolved_axes[i]];
            }

            dst[dst_idx] = src[src_idx];
        }
    }

    return result;
}

Tensor Tensor::expand(const TensorShape& target_shape) const {
    if (!is_valid()) return {};

    // Check if expansion is valid
    if (target_shape.rank() < shape_.rank()) {
        LOG_ERROR("Cannot expand to fewer dimensions");
        return {};
    }

    // Prepend 1s to match rank if needed
    std::vector<size_t> padded_shape = shape_.dims();
    while (padded_shape.size() < target_shape.rank()) {
        padded_shape.insert(padded_shape.begin(), 1);
    }

    // Check compatibility and build final shape
    std::vector<size_t> final_shape(target_shape.rank());
    for (size_t i = 0; i < target_shape.rank(); ++i) {
        size_t target_dim = target_shape[i];

        // Handle -1 in target shape (keep original dimension)
        if (target_dim == static_cast<size_t>(-1)) {
            if (i < padded_shape.size()) {
                final_shape[i] = padded_shape[i];
            } else {
                LOG_ERROR("Cannot use -1 for new dimension");
                return {};
            }
        } else {
            // Check if expansion is valid
            if (padded_shape[i] != 1 && padded_shape[i] != target_dim) {
                LOG_ERROR("Cannot expand dimension {} from {} to {}",
                         i, padded_shape[i], target_dim);
                return {};
            }
            final_shape[i] = target_dim;
        }
    }

    // First reshape to add 1s if needed
    auto reshaped = reshape(TensorShape(padded_shape));

    // Then broadcast to final shape
    return reshaped.broadcast_to(TensorShape(final_shape));
}

// Implementation of slice (was inline in header, now moved here)
Tensor Tensor::slice(std::span<const std::pair<int, int>> ranges) const {
    if (!is_valid()) return {};

    if (ranges.size() > shape_.rank()) {
        LOG_ERROR("Too many slice ranges for tensor rank");
        return {};
    }

    // Build start and end vectors
    std::vector<size_t> starts(shape_.rank());
    std::vector<size_t> ends(shape_.rank());

    for (size_t i = 0; i < shape_.rank(); ++i) {
        if (i < ranges.size()) {
            int start = ranges[i].first;
            int end = ranges[i].second;

            // Handle negative indices
            if (start < 0) start = shape_[i] + start;
            if (end < 0) end = shape_[i] + end;

            // Clamp to valid range
            start = std::max(0, std::min(start, static_cast<int>(shape_[i])));
            end = std::max(start, std::min(end, static_cast<int>(shape_[i])));

            starts[i] = start;
            ends[i] = end;
        } else {
            starts[i] = 0;
            ends[i] = shape_[i];
        }
    }

    // Calculate new shape
    std::vector<size_t> new_shape;
    for (size_t i = 0; i < shape_.rank(); ++i) {
        new_shape.push_back(ends[i] - starts[i]);
    }

    // Check if slice is contiguous
    bool is_contiguous = is_contiguous_slice(starts, ends);

    if (is_contiguous) {
        // Can create a view - calculate the offset into the original data
        size_t offset = calculate_offset(starts);
        void* new_data = static_cast<char*>(data_) + offset * dtype_size(dtype_);

        // Return a view (non-owning tensor)
        return Tensor(new_data, TensorShape(new_shape), device_, dtype_);
    } else {
        // Need to copy
        return copy_slice(starts, ends, new_shape);
    }
}

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

// Helper: check if slice is contiguous
bool Tensor::is_contiguous_slice(const std::vector<size_t>& starts,
                                 const std::vector<size_t>& ends) const {
    // A slice is contiguous if:
    // 1. We're only slicing the first dimension
    // 2. All other dimensions are fully selected (start=0, end=shape[dim])

    for (size_t i = 1; i < shape_.rank(); ++i) {
        if (starts[i] != 0 || ends[i] != shape_[i]) {
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

// Helper: resolve dimensions
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

#undef CHECK_CUDA

} // namespace gs