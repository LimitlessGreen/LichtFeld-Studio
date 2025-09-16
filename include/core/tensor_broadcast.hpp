/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <algorithm>
#include <numeric>

namespace gs {

    // Broadcasting utilities
    class BroadcastHelper {
    public:
        // Check if two shapes can be broadcast together
        static bool can_broadcast(const TensorShape& a, const TensorShape& b);

        // Get the resulting shape after broadcasting two shapes
        static TensorShape broadcast_shape(const TensorShape& a, const TensorShape& b);

        // Expand a tensor to a target shape (adds dimensions of size 1)
        static Tensor expand(const Tensor& tensor, const TensorShape& target_shape);

        // Broadcast two tensors to have compatible shapes
        static std::pair<Tensor, Tensor> broadcast_tensors(const Tensor& a, const Tensor& b);

        // Calculate strides for broadcasting
        static std::vector<size_t> compute_broadcast_strides(const TensorShape& shape,
                                                             const TensorShape& target_shape);

    private:
        // Helper to expand dimensions (add leading 1s)
        static TensorShape expand_dims(const TensorShape& shape, size_t target_rank);
    };

    // Extended TensorShape with broadcasting support
    class TensorShapeEx : public TensorShape {
    public:
        using TensorShape::TensorShape;

        // Check if this shape can broadcast to target
        bool can_broadcast_to(const TensorShape& target) const;

        // Get broadcast shape with another shape
        TensorShape broadcast_with(const TensorShape& other) const;

        // Get strides for iteration
        std::vector<size_t> strides() const;

        // Get broadcast strides for target shape
        std::vector<size_t> broadcast_strides(const TensorShape& target) const;
    };

    // Broadcast iterator for efficient element access
    class BroadcastIterator {
    public:
        BroadcastIterator(const TensorShape& shape, const TensorShape& broadcast_shape);

        // Get current index in original tensor
        size_t index() const { return current_index_; }

        // Move to next element
        void next();

        // Check if done
        bool done() const { return done_; }

        // Reset to beginning
        void reset();

    private:
        TensorShape shape_;
        TensorShape broadcast_shape_;
        std::vector<size_t> strides_;
        std::vector<size_t> broadcast_strides_;
        std::vector<size_t> indices_;
        size_t current_index_;
        size_t total_elements_;
        size_t element_count_;
        bool done_;

        void compute_index();
    };

} // namespace gs
