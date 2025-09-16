/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_broadcast.hpp"
#include "core/tensor_ops.hpp"
#include <algorithm>
#include <numeric>

namespace gs {

    // ============= BroadcastHelper Implementation =============

    bool BroadcastHelper::can_broadcast(const TensorShape& a, const TensorShape& b) {
        size_t max_rank = std::max(a.rank(), b.rank());

        // Check each dimension from right to left
        for (size_t i = 0; i < max_rank; ++i) {
            // Get dimension from the right (padding with 1 if needed)
            size_t dim_a = (i < a.rank()) ? a[a.rank() - 1 - i] : 1;
            size_t dim_b = (i < b.rank()) ? b[b.rank() - 1 - i] : 1;

            // Dimensions are compatible if they're equal or one is 1
            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                return false;
            }
        }

        return true;
    }

    TensorShape BroadcastHelper::broadcast_shape(const TensorShape& a, const TensorShape& b) {
        if (!can_broadcast(a, b)) {
            LOG_ERROR("Cannot broadcast shapes {} and {}", a.str(), b.str());
            return TensorShape();
        }

        size_t max_rank = std::max(a.rank(), b.rank());
        std::vector<size_t> result_dims(max_rank);

        // Compute result shape from right to left
        for (size_t i = 0; i < max_rank; ++i) {
            size_t dim_a = (i < a.rank()) ? a[a.rank() - 1 - i] : 1;
            size_t dim_b = (i < b.rank()) ? b[b.rank() - 1 - i] : 1;

            // Take the maximum dimension (1 broadcasts to any size)
            result_dims[max_rank - 1 - i] = std::max(dim_a, dim_b);
        }

        return TensorShape(result_dims);
    }

    TensorShape BroadcastHelper::expand_dims(const TensorShape& shape, size_t target_rank) {
        if (shape.rank() >= target_rank) {
            return shape;
        }

        std::vector<size_t> new_dims(target_rank - shape.rank(), 1);
        for (size_t i = 0; i < shape.rank(); ++i) {
            new_dims.push_back(shape[i]);
        }

        return TensorShape(new_dims);
    }

    Tensor BroadcastHelper::expand(const Tensor& tensor, const TensorShape& target_shape) {
        if (!tensor.is_valid()) {
            return Tensor();
        }

        // Check if broadcasting is possible
        if (!can_broadcast(tensor.shape(), target_shape)) {
            LOG_ERROR("Cannot broadcast tensor shape {} to {}",
                      tensor.shape().str(), target_shape.str());
            return Tensor();
        }

        // If shapes are already the same, return clone
        if (tensor.shape() == target_shape) {
            return tensor.clone();
        }

        // Create result tensor
        auto result = Tensor::empty(target_shape, tensor.device(), tensor.dtype());

        // Perform the broadcast expansion
        if (tensor.device() == Device::CUDA) {
            // Launch CUDA kernel for broadcast
            tensor_ops::launch_broadcast(
                tensor.ptr<float>(),
                result.ptr<float>(),
                tensor.shape().dims().data(),
                target_shape.dims().data(),
                tensor.shape().rank(),
                target_shape.rank(),
                target_shape.elements(),
                0 // stream
            );
            cudaDeviceSynchronize();
        } else {
            // CPU implementation
            BroadcastIterator src_iter(tensor.shape(), target_shape);
            const float* src_data = tensor.ptr<float>();
            float* dst_data = result.ptr<float>();

            size_t dst_idx = 0;
            while (!src_iter.done()) {
                dst_data[dst_idx++] = src_data[src_iter.index()];
                src_iter.next();
            }
        }

        return result;
    }

    std::pair<Tensor, Tensor> BroadcastHelper::broadcast_tensors(const Tensor& a, const Tensor& b) {
        if (!a.is_valid() || !b.is_valid()) {
            return std::make_pair(Tensor(), Tensor());
        }

        // Get broadcast shape
        TensorShape broadcast_shape = BroadcastHelper::broadcast_shape(a.shape(), b.shape());

        if (!broadcast_shape.is_initialized()) {
            return std::make_pair(Tensor(), Tensor());
        }

        // Expand both tensors to broadcast shape
        Tensor expanded_a = expand(a, broadcast_shape);
        Tensor expanded_b = expand(b, broadcast_shape);

        return std::make_pair(std::move(expanded_a), std::move(expanded_b));
    }

    std::vector<size_t> BroadcastHelper::compute_broadcast_strides(const TensorShape& shape,
                                                                   const TensorShape& target_shape) {
        size_t shape_rank = shape.rank();
        size_t target_rank = target_shape.rank();
        std::vector<size_t> strides(target_rank, 0);

        // Compute strides from right to left
        size_t stride = 1;
        for (int i = target_rank - 1; i >= 0; --i) {
            // Map target dimension to source dimension
            int src_dim_idx = shape_rank - (target_rank - i);

            if (src_dim_idx >= 0 && shape[src_dim_idx] > 1) {
                // This dimension exists in source and is not broadcast
                strides[i] = stride;
                stride *= shape[src_dim_idx];
            } else {
                // This dimension is broadcast (size 1) or doesn't exist in source
                strides[i] = 0;
            }
        }

        return strides;
    }

    // ============= TensorShapeEx Implementation =============

    bool TensorShapeEx::can_broadcast_to(const TensorShape& target) const {
        return BroadcastHelper::can_broadcast(*this, target);
    }

    TensorShape TensorShapeEx::broadcast_with(const TensorShape& other) const {
        return BroadcastHelper::broadcast_shape(*this, other);
    }

    std::vector<size_t> TensorShapeEx::strides() const {
        std::vector<size_t> result(rank());
        if (rank() == 0) {
            return result;
        }

        result[rank() - 1] = 1;
        for (int i = rank() - 2; i >= 0; --i) {
            result[i] = result[i + 1] * operator[](i + 1);
        }

        return result;
    }

    std::vector<size_t> TensorShapeEx::broadcast_strides(const TensorShape& target) const {
        return BroadcastHelper::compute_broadcast_strides(*this, target);
    }

    // ============= BroadcastIterator Implementation =============

    BroadcastIterator::BroadcastIterator(const TensorShape& shape, const TensorShape& broadcast_shape)
        : shape_(shape),
          broadcast_shape_(broadcast_shape),
          current_index_(0),
          element_count_(0),
          done_(false) {

        total_elements_ = broadcast_shape.elements();
        indices_.resize(broadcast_shape.rank(), 0);

        // Compute strides for both shapes
        strides_.resize(shape.rank());
        if (shape.rank() > 0) {
            strides_[shape.rank() - 1] = 1;
            for (int i = shape.rank() - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape[i + 1];
            }
        }

        broadcast_strides_ = BroadcastHelper::compute_broadcast_strides(shape, broadcast_shape);

        compute_index();
    }

    void BroadcastIterator::next() {
        if (done_) {
            return;
        }

        element_count_++;
        if (element_count_ >= total_elements_) {
            done_ = true;
            return;
        }

        // Increment indices from right to left
        for (int i = broadcast_shape_.rank() - 1; i >= 0; --i) {
            indices_[i]++;
            if (indices_[i] < broadcast_shape_[i]) {
                break;
            }
            indices_[i] = 0;
        }

        compute_index();
    }

    void BroadcastIterator::reset() {
        std::fill(indices_.begin(), indices_.end(), 0);
        current_index_ = 0;
        element_count_ = 0;
        done_ = false;
        compute_index();
    }

    void BroadcastIterator::compute_index() {
        current_index_ = 0;
        int rank_diff = broadcast_shape_.rank() - shape_.rank();

        for (size_t i = 0; i < broadcast_shape_.rank(); ++i) {
            int src_dim_idx = i - rank_diff;

            if (src_dim_idx >= 0) {
                // This dimension exists in source
                if (shape_[src_dim_idx] > 1) {
                    // Normal dimension
                    current_index_ += indices_[i] * strides_[src_dim_idx];
                }
                // If dimension is 1, it broadcasts (index stays 0)
            }
            // If dimension doesn't exist in source, it's implicitly broadcast
        }
    }

} // namespace gs
