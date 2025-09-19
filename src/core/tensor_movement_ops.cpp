/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include "core/tensor_broadcast.hpp"
#include <algorithm>
#include <numeric>

namespace gs {

// ============= Unified Movement Operation =============
Tensor Tensor::movement(MovementOp op, const MovementArgs& args) const {
    if (!is_valid()) return {};
    
    switch (op) {
        case MovementOp::Reshape: {
            if (auto* vec = std::get_if<std::vector<int>>(&args.args)) {
                return reshape(std::span<const int>(*vec));
            }
            LOG_ERROR("Reshape requires vector<int> args");
            return {};
        }
        
        case MovementOp::Permute: {
            if (auto* vec = std::get_if<std::vector<int>>(&args.args)) {
                return permute(std::span<const int>(*vec));
            }
            LOG_ERROR("Permute requires vector<int> args");
            return {};
        }
        
        case MovementOp::Expand: {
            if (auto* vec = std::get_if<std::vector<int>>(&args.args)) {
                return expand(std::span<const int>(*vec));
            }
            LOG_ERROR("Expand requires vector<int> args");
            return {};
        }
        
        case MovementOp::Transpose: {
            if (auto* pair = std::get_if<std::pair<int, int>>(&args.args)) {
                return transpose(pair->first, pair->second);
            }
            // Default transpose
            return transpose();
        }
        
        case MovementOp::Squeeze: {
            if (auto* dim = std::get_if<int>(&args.args)) {
                return squeeze(*dim);
            }
            // Squeeze all dimensions
            return squeeze();
        }
        
        case MovementOp::Unsqueeze: {
            if (auto* dim = std::get_if<int>(&args.args)) {
                return unsqueeze(*dim);
            }
            LOG_ERROR("Unsqueeze requires int dim arg");
            return {};
        }
        
        case MovementOp::Flatten: {
            if (auto* pair = std::get_if<std::pair<int, int>>(&args.args)) {
                return flatten(pair->first, pair->second);
            }
            // Default flatten
            return flatten();
        }
        
        case MovementOp::Shrink: {  // This is slice
            if (auto* ranges = std::get_if<std::vector<std::pair<int, int>>>(&args.args)) {
                return slice(std::span<const std::pair<int, int>>(*ranges));
            }
            LOG_ERROR("Shrink requires vector<pair<int,int>> args");
            return {};
        }
        
        case MovementOp::Pad: {
            if (auto* padding = std::get_if<std::vector<std::pair<int, int>>>(&args.args)) {
                // Pad implementation
                std::vector<size_t> new_shape = shape_.dims();
                std::vector<size_t> pad_before(shape_.rank(), 0);
                std::vector<size_t> pad_after(shape_.rank(), 0);
                
                for (size_t i = 0; i < padding->size() && i < shape_.rank(); ++i) {
                    pad_before[i] = (*padding)[i].first;
                    pad_after[i] = (*padding)[i].second;
                    new_shape[i] += pad_before[i] + pad_after[i];
                }
                
                auto result = zeros(TensorShape(new_shape), device_, dtype_);
                
                // Copy original data to padded tensor
                // This is simplified - full implementation would handle this properly
                if (device_ == Device::CPU && dtype_ == DataType::Float32) {
                    const float* src = ptr<float>();
                    float* dst = result.ptr<float>();
                    
                    // Calculate strides
                    std::vector<size_t> src_strides(shape_.rank());
                    std::vector<size_t> dst_strides(shape_.rank());
                    
                    src_strides.back() = 1;
                    dst_strides.back() = 1;
                    
                    for (int i = shape_.rank() - 2; i >= 0; --i) {
                        src_strides[i] = src_strides[i + 1] * shape_[i + 1];
                        dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];
                    }
                    
                    // Copy data
                    for (size_t i = 0; i < numel(); ++i) {
                        // Convert linear index to coordinates
                        std::vector<size_t> coords(shape_.rank());
                        size_t temp = i;
                        for (size_t d = 0; d < shape_.rank(); ++d) {
                            coords[d] = temp / src_strides[d];
                            temp %= src_strides[d];
                        }
                        
                        // Add padding offset
                        size_t dst_idx = 0;
                        for (size_t d = 0; d < shape_.rank(); ++d) {
                            dst_idx += (coords[d] + pad_before[d]) * dst_strides[d];
                        }
                        
                        dst[dst_idx] = src[i];
                    }
                } else {
                    LOG_WARN("Pad not fully implemented for CUDA");
                }
                
                return result;
            }
            LOG_ERROR("Pad requires vector<pair<int,int>> args");
            return {};
        }
        
        case MovementOp::Flip: {
            if (auto* vec = std::get_if<std::vector<int>>(&args.args)) {
                // Flip along specified axes
                auto result = clone();
                
                if (device_ == Device::CPU && dtype_ == DataType::Float32) {
                    float* data = result.ptr<float>();
                    
                    for (int axis : *vec) {
                        axis = resolve_dim(axis);
                        if (axis < 0 || axis >= static_cast<int>(shape_.rank())) continue;
                        
                        // Calculate stride for this axis
                        size_t stride = 1;
                        for (size_t i = axis + 1; i < shape_.rank(); ++i) {
                            stride *= shape_[i];
                        }
                        
                        size_t outer_size = 1;
                        for (int i = 0; i < axis; ++i) {
                            outer_size *= shape_[i];
                        }
                        
                        // Flip along axis
                        for (size_t o = 0; o < outer_size; ++o) {
                            for (size_t i = 0; i < shape_[axis] / 2; ++i) {
                                size_t j = shape_[axis] - 1 - i;
                                
                                // Swap elements
                                for (size_t inner = 0; inner < stride; ++inner) {
                                    size_t idx1 = o * shape_[axis] * stride + i * stride + inner;
                                    size_t idx2 = o * shape_[axis] * stride + j * stride + inner;
                                    std::swap(data[idx1], data[idx2]);
                                }
                            }
                        }
                    }
                } else {
                    LOG_WARN("Flip not fully implemented for CUDA");
                }
                
                return result;
            }
            LOG_ERROR("Flip requires vector<int> axes");
            return {};
        }
        
        default:
            LOG_ERROR("Unknown movement operation");
            return {};
    }
}

// Now the shape operations just delegate to movement
Tensor Tensor::reshape(std::span<const int> sizes) const {
    return movement(MovementOp::Reshape, {std::vector<int>(sizes.begin(), sizes.end())});
}

Tensor Tensor::permute(std::span<const int> axes) const {
    return movement(MovementOp::Permute, {std::vector<int>(axes.begin(), axes.end())});
}

Tensor Tensor::expand(std::span<const int> sizes) const {
    return movement(MovementOp::Expand, {std::vector<int>(sizes.begin(), sizes.end())});
}

Tensor Tensor::transpose(int dim1, int dim2) const {
    return movement(MovementOp::Transpose, {std::pair<int, int>{dim1, dim2}});
}

Tensor Tensor::squeeze(std::optional<int> dim) const {
    if (dim.has_value()) {
        return movement(MovementOp::Squeeze, {dim.value()});
    }
    return movement(MovementOp::Squeeze, {-1});  // -1 means squeeze all
}

Tensor Tensor::unsqueeze(int dim) const {
    return movement(MovementOp::Unsqueeze, {dim});
}

Tensor Tensor::flatten(int start_dim, int end_dim) const {
    return movement(MovementOp::Flatten, {std::pair<int, int>{start_dim, end_dim}});
}

Tensor Tensor::slice(std::span<const std::pair<int, int>> ranges) const {
    return movement(MovementOp::Shrink, 
                   {std::vector<std::pair<int, int>>(ranges.begin(), ranges.end())});
}

} // namespace gs
