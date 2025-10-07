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
                // Convert to size_t and handle -1
                std::vector<size_t> new_shape;
                int infer_dim = -1;
                size_t known_size = 1;

                for (size_t i = 0; i < vec->size(); ++i) {
                    if ((*vec)[i] == -1) {
                        if (infer_dim != -1) {
                            LOG_ERROR("Only one dimension can be inferred");
                            return {};
                        }
                        infer_dim = i;
                        new_shape.push_back(1); // Placeholder
                    } else if ((*vec)[i] < 0) {
                        LOG_ERROR("Invalid reshape dimension: {}", (*vec)[i]);
                        return {};
                    } else {
                        new_shape.push_back((*vec)[i]);
                        known_size *= (*vec)[i];
                    }
                }

                // Infer dimension if needed
                if (infer_dim != -1) {
                    if (numel() % known_size != 0) {
                        LOG_ERROR("Cannot infer dimension for reshape");
                        return {};
                    }
                    new_shape[infer_dim] = numel() / known_size;
                }

                // Check if total elements match
                size_t total = 1;
                for (auto d : new_shape) {
                    total *= d;
                }

                if (total != numel()) {
                    LOG_ERROR("View shape {} has {} elements, but tensor has {} elements",
                              TensorShape(new_shape).str(), total, numel());
                    return {};
                }

                // Create a view that shares memory (non-owning)
                Tensor view(data_, TensorShape(new_shape), device_, dtype_);
                view.data_owner_ = data_owner_;  // Share ownership!
                view.is_view_ = true;  // Mark as view
                return view;
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
                std::vector<size_t> target_shape;
                for (int dim : *vec) {
                    target_shape.push_back(static_cast<size_t>(dim));
                }
                return expand(TensorShape(target_shape));
            }
            LOG_ERROR("Expand requires vector<int> args");
            return {};
        }

        case MovementOp::Transpose: {
            if (auto* pair = std::get_if<std::pair<int, int>>(&args.args)) {
                int dim1 = resolve_dim(pair->first);
                int dim2 = resolve_dim(pair->second);

                if (dim1 < 0 || dim1 >= static_cast<int>(shape_.rank()) ||
                    dim2 < 0 || dim2 >= static_cast<int>(shape_.rank())) {
                    LOG_ERROR("Invalid transpose dimensions");
                    return {};
                }

                // Create permutation
                std::vector<int> perm(shape_.rank());
                for (size_t i = 0; i < shape_.rank(); ++i) {
                    perm[i] = i;
                }
                std::swap(perm[dim1], perm[dim2]);

                return permute(perm);
            }
            // Default transpose (swap last two dimensions)
            if (shape_.rank() < 2) return clone();
            return transpose(-2, -1);
        }

        case MovementOp::Squeeze: {
            if (auto* dim = std::get_if<int>(&args.args)) {
                std::vector<size_t> new_shape;
                int resolved = *dim;

                // INT_MIN is the sentinel for "squeeze all dimensions with size 1"
                bool squeeze_all = (*dim == std::numeric_limits<int>::min());

                if (!squeeze_all) {
                    // Resolve negative index normally
                    if (resolved < 0) {
                        resolved = shape_.rank() + resolved;
                    }

                    if (resolved < 0 || resolved >= static_cast<int>(shape_.rank())) {
                        LOG_ERROR("Invalid squeeze dimension: {} for rank {}", *dim, shape_.rank());
                        return {};
                    }
                }

                // Build new shape
                for (size_t i = 0; i < shape_.rank(); ++i) {
                    if (squeeze_all) {
                        // Squeeze all dimensions with size 1
                        if (shape_[i] != 1) {
                            new_shape.push_back(shape_[i]);
                        }
                    } else {
                        // Squeeze specific dimension only if it has size 1
                        if (i != static_cast<size_t>(resolved) || shape_[i] != 1) {
                            new_shape.push_back(shape_[i]);
                        }
                    }
                }

                if (new_shape.empty()) {
                    new_shape.push_back(1); // Scalar
                }

                // Create a view that shares memory
                Tensor view(data_, TensorShape(new_shape), device_, dtype_);
                view.data_owner_ = data_owner_;  // Share ownership
                view.is_view_ = true;  // Mark as view
                return view;
            }
            LOG_ERROR("Squeeze requires int dim arg");
            return {};
        }

        case MovementOp::Unsqueeze: {
            if (auto* dim = std::get_if<int>(&args.args)) {
                int resolved = *dim;
                if (resolved < 0) {
                    resolved = shape_.rank() + resolved + 1;
                }
                if (resolved < 0 || resolved > static_cast<int>(shape_.rank())) {
                    LOG_ERROR("Invalid unsqueeze dimension");
                    return {};
                }

                std::vector<size_t> new_shape;
                for (int i = 0; i < resolved; ++i) {
                    new_shape.push_back(shape_[i]);
                }
                new_shape.push_back(1);
                for (size_t i = resolved; i < shape_.rank(); ++i) {
                    new_shape.push_back(shape_[i]);
                }

                // Create a view that shares memory
                Tensor view(data_, TensorShape(new_shape), device_, dtype_);
                view.data_owner_ = data_owner_;  // Share ownership
                view.is_view_ = true;  // Mark as view
                return view;
            }
            LOG_ERROR("Unsqueeze requires int dim arg");
            return {};
        }

        case MovementOp::Flatten: {
            if (auto* pair = std::get_if<std::pair<int, int>>(&args.args)) {
                int start = resolve_dim(pair->first);
                int end = resolve_dim(pair->second);

                if (start < 0 || start >= static_cast<int>(shape_.rank()) ||
                    end < 0 || end >= static_cast<int>(shape_.rank()) ||
                    start > end) {
                    LOG_ERROR("Invalid flatten dimensions");
                    return {};
                    }

                std::vector<size_t> new_shape;
                for (int i = 0; i < start; ++i) {
                    new_shape.push_back(shape_[i]);
                }

                size_t flattened_size = 1;
                for (int i = start; i <= end; ++i) {
                    flattened_size *= shape_[i];
                }
                new_shape.push_back(flattened_size);

                for (size_t i = end + 1; i < shape_.rank(); ++i) {
                    new_shape.push_back(shape_[i]);
                }

                // Create a view that shares memory
                Tensor view(data_, TensorShape(new_shape), device_, dtype_);
                view.data_owner_ = data_owner_;  // Share ownership
                view.is_view_ = true;  // Mark as view
                return view;
            }
            // Default flatten (all dimensions)
            Tensor view(data_, TensorShape({numel()}), device_, dtype_);
            view.data_owner_ = data_owner_;  // Share ownership
            view.is_view_ = true;  // Mark as view
            return view;
        }

        case MovementOp::Slice: {
            if (auto* ranges = std::get_if<std::vector<std::pair<int, int>>>(&args.args)) {
                return slice(std::span<const std::pair<int, int>>(*ranges));
            }
            LOG_ERROR("Slice requires vector<pair<int,int>> args");
            return {};
        }

        case MovementOp::Cat: {
            if (auto* cat_args = std::get_if<std::pair<void*, int>>(&args.args)) {
                const Tensor& other = *static_cast<const Tensor*>(cat_args->first);
                int dim = resolve_dim(cat_args->second);

                if (!other.is_valid() || shape_.rank() != other.shape().rank()) {
                    LOG_ERROR("Cannot concatenate tensors with different ranks");
                    return {};
                }

                // Check all dimensions match except cat dimension
                for (size_t i = 0; i < shape_.rank(); ++i) {
                    if (i != static_cast<size_t>(dim) && shape_[i] != other.shape()[i]) {
                        LOG_ERROR("Dimension {} size mismatch for concatenation", i);
                        return {};
                    }
                }

                // For now, only implement dim=0 concatenation
                if (dim != 0) {
                    LOG_ERROR("Concatenation only implemented for dim=0");
                    return {};
                }

                std::vector<size_t> result_dims = shape_.dims();
                result_dims[dim] = shape_[dim] + other.shape()[dim];

                auto result = empty(TensorShape(result_dims), device_, dtype_);

                // Copy data
                size_t bytes_per_row = numel() / shape_[0] * dtype_size(dtype_);
                size_t self_bytes = bytes();
                size_t other_bytes = other.bytes();

                if (device_ == Device::CUDA) {
                    cudaMemcpy(result.raw_ptr(), raw_ptr(), self_bytes, cudaMemcpyDeviceToDevice);
                    cudaMemcpy(static_cast<char*>(result.raw_ptr()) + self_bytes,
                              other.raw_ptr(), other_bytes, cudaMemcpyDeviceToDevice);
                } else {
                    std::memcpy(result.raw_ptr(), raw_ptr(), self_bytes);
                    std::memcpy(static_cast<char*>(result.raw_ptr()) + self_bytes,
                               other.raw_ptr(), other_bytes);
                }

                return result;
            }
            LOG_ERROR("Cat requires (Tensor*, dim) pair");
            return {};
        }

        case MovementOp::Pad: {
            if (auto* padding = std::get_if<std::vector<std::pair<int, int>>>(&args.args)) {
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

} // namespace gs