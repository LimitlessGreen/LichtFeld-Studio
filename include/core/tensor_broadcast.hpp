/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <algorithm>
#include <bit>
#include <concepts>
#include <execution>
#include <ranges>
#include <span>

namespace gs {

// Modern C++23 broadcasting utilities
class BroadcastUtil {
public:
    // Check if shapes can broadcast - simplified with ranges
    static bool can_broadcast(std::span<const size_t> a, std::span<const size_t> b) {
        auto check = [](size_t x, size_t y) { return x == y || x == 1 || y == 1; };

        // Compare from right to left using ranges
        return std::ranges::equal(
            a | std::views::reverse | std::views::take(std::min(a.size(), b.size())),
            b | std::views::reverse | std::views::take(std::min(a.size(), b.size())),
            check
        ) && std::ranges::all_of(
            std::views::iota(std::min(a.size(), b.size()), std::max(a.size(), b.size())),
            [&](size_t) { return true; }  // Extra dims are always broadcastable
        );
    }

    // Compute broadcast shape - cleaner with C++23
    static std::vector<size_t> broadcast_shape(std::span<const size_t> a,
                                               std::span<const size_t> b) {
        size_t max_rank = std::max(a.size(), b.size());
        std::vector<size_t> result(max_rank);

        // Fill from right to left
        std::ranges::generate(result | std::views::reverse, [&, i = 0]() mutable {
            size_t dim_a = (i < a.size()) ? a[a.size() - 1 - i] : 1;
            size_t dim_b = (i < b.size()) ? b[b.size() - 1 - i] : 1;
            i++;
            return std::max(dim_a, dim_b);
        });

        return result;
    }

    // Fast index mapping for broadcasting - no iterator needed
    static constexpr size_t map_broadcast_index(size_t linear_idx,
                                                std::span<const size_t> out_shape,
                                                std::span<const size_t> in_shape) {
        size_t in_idx = 0;
        size_t out_stride = 1;

        // Work backwards through dimensions
        for (int i = out_shape.size() - 1; i >= 0; --i) {
            size_t out_coord = (linear_idx / out_stride) % out_shape[i];

            // Map to input dimension
            int in_dim = i - (out_shape.size() - in_shape.size());
            if (in_dim >= 0) {
                size_t in_coord = (in_shape[in_dim] == 1) ? 0 : out_coord;
                size_t in_stride = 1;
                for (int j = in_dim + 1; j < in_shape.size(); ++j) {
                    in_stride *= in_shape[j];
                }
                in_idx += in_coord * in_stride;
            }

            out_stride *= out_shape[i];
        }

        return in_idx;
    }

    // Optimized broadcasting patterns detection
    enum class Pattern {
        None,
        Scalar,      // One operand is scalar
        Outer,       // Outer product pattern
        Aligned,     // Same shape
        Simple       // Simple broadcast (last dims match)
    };

    static Pattern detect_pattern(std::span<const size_t> a, std::span<const size_t> b) {
        if (std::ranges::equal(a, b)) return Pattern::Aligned;
        if (a.size() == 0 || b.size() == 0) return Pattern::Scalar;

        // Check for outer product
        if (std::ranges::all_of(a, [](size_t d) { return d == 1; }) ||
            std::ranges::all_of(b, [](size_t d) { return d == 1; })) {
            return Pattern::Outer;
        }

        // Check if trailing dimensions match
        auto min_size = std::min(a.size(), b.size());
        if (std::ranges::equal(
            a | std::views::reverse | std::views::take(min_size),
            b | std::views::reverse | std::views::take(min_size))) {
            return Pattern::Simple;
        }

        return Pattern::None;
    }
};

// Simplified broadcast operation for CPU
template<typename Op>
inline void broadcast_op_cpu(const float* a, const float* b, float* c,
                     std::span<const size_t> a_shape,
                     std::span<const size_t> b_shape,
                     std::span<const size_t> c_shape,
                     Op op) {
    size_t total = std::ranges::fold_left(c_shape, 1uz, std::multiplies{});

    auto pattern = BroadcastUtil::detect_pattern(a_shape, b_shape);

    // Fast path for aligned shapes
    if (pattern == BroadcastUtil::Pattern::Aligned) {
        // Use parallel algorithm if available, fallback to serial
        #ifdef __cpp_lib_parallel_algorithm
        std::transform(std::execution::par_unseq, a, a + total, b, c, op);
        #else
        std::transform(a, a + total, b, c, op);
        #endif
        return;
    }

    // Generic broadcasting - using simple loop for compatibility
    #ifdef __cpp_lib_parallel_algorithm
    std::for_each(std::execution::par_unseq,
                  std::views::iota(0uz, total).begin(),
                  std::views::iota(0uz, total).end(),
                  [&](size_t i) {
        size_t a_idx = BroadcastUtil::map_broadcast_index(i, c_shape, a_shape);
        size_t b_idx = BroadcastUtil::map_broadcast_index(i, c_shape, b_shape);
        c[i] = op(a[a_idx], b[b_idx]);
    });
    #else
    // Fallback for systems without parallel algorithms
    for (size_t i = 0; i < total; ++i) {
        size_t a_idx = BroadcastUtil::map_broadcast_index(i, c_shape, a_shape);
        size_t b_idx = BroadcastUtil::map_broadcast_index(i, c_shape, b_shape);
        c[i] = op(a[a_idx], b[b_idx]);
    }
    #endif
}

// Overload for unsigned char
template<typename Op>
inline void broadcast_op_cpu(const unsigned char* a, const unsigned char* b, unsigned char* c,
                     std::span<const size_t> a_shape,
                     std::span<const size_t> b_shape,
                     std::span<const size_t> c_shape,
                     Op op) {
    size_t total = std::ranges::fold_left(c_shape, 1uz, std::multiplies{});

    auto pattern = BroadcastUtil::detect_pattern(a_shape, b_shape);

    // Fast path for aligned shapes
    if (pattern == BroadcastUtil::Pattern::Aligned) {
        #ifdef __cpp_lib_parallel_algorithm
        std::transform(std::execution::par_unseq, a, a + total, b, c, op);
        #else
        std::transform(a, a + total, b, c, op);
        #endif
        return;
    }

    // Generic broadcasting
    #ifdef __cpp_lib_parallel_algorithm
    std::for_each(std::execution::par_unseq,
                  std::views::iota(0uz, total).begin(),
                  std::views::iota(0uz, total).end(),
                  [&](size_t i) {
        size_t a_idx = BroadcastUtil::map_broadcast_index(i, c_shape, a_shape);
        size_t b_idx = BroadcastUtil::map_broadcast_index(i, c_shape, b_shape);
        c[i] = op(a[a_idx], b[b_idx]);
    });
    #else
    for (size_t i = 0; i < total; ++i) {
        size_t a_idx = BroadcastUtil::map_broadcast_index(i, c_shape, a_shape);
        size_t b_idx = BroadcastUtil::map_broadcast_index(i, c_shape, b_shape);
        c[i] = op(a[a_idx], b[b_idx]);
    }
    #endif
}

// Concept for broadcast-compatible operations
template<typename T>
concept BroadcastOp = requires(T op, float a, float b) {
    { op(a, b) } -> std::convertible_to<float>;
};

// Unified broadcast expansion - replaces old expand() method
class BroadcastExpander {
public:
    static Tensor expand(const Tensor& tensor, const TensorShape& target_shape) {
        if (!BroadcastUtil::can_broadcast(tensor.shape().dims(), target_shape.dims())) {
            LOG_ERROR("Cannot broadcast {} to {}", tensor.shape().str(), target_shape.str());
            return {};
        }

        if (tensor.shape() == target_shape) {
            return tensor.clone();
        }

        auto result = Tensor::empty(target_shape, tensor.device(), tensor.dtype());

        if (tensor.device() == Device::CUDA) {
            // Will be handled in tensor_broadcast_ops.cu
            launch_broadcast_kernel(tensor, result);
        } else {
            expand_cpu(tensor, result);
        }

        return result;
    }

private:
    static void expand_cpu(const Tensor& src, Tensor& dst);
    static void launch_broadcast_kernel(const Tensor& src, Tensor& dst);
};

// ============= Backward Compatibility =============
// Provide aliases for old class names to avoid breaking existing code

class BroadcastHelper {
public:
    static bool can_broadcast(const TensorShape& a, const TensorShape& b) {
        return BroadcastUtil::can_broadcast(a.dims(), b.dims());
    }

    static TensorShape broadcast_shape(const TensorShape& a, const TensorShape& b) {
        auto result = BroadcastUtil::broadcast_shape(a.dims(), b.dims());
        return TensorShape(result);
    }

    static Tensor expand(const Tensor& tensor, const TensorShape& target) {
        return BroadcastExpander::expand(tensor, target);
    }
};

// Simple iterator for backward compatibility
class BroadcastIterator {
private:
    std::span<const size_t> shape_;
    std::span<const size_t> broadcast_shape_;
    size_t current_index_ = 0;
    size_t element_count_ = 0;
    size_t total_elements_;
    bool done_ = false;

public:
    BroadcastIterator(const TensorShape& shape, const TensorShape& broadcast_shape)
        : shape_(shape.dims()),
          broadcast_shape_(broadcast_shape.dims()),
          total_elements_(broadcast_shape.elements()) {}

    size_t index() const {
        return BroadcastUtil::map_broadcast_index(element_count_, broadcast_shape_, shape_);
    }

    void next() {
        element_count_++;
        done_ = (element_count_ >= total_elements_);
    }

    bool done() const { return done_; }

    void reset() {
        element_count_ = 0;
        done_ = false;
    }
};

} // namespace gs