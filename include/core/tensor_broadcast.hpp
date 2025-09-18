/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <algorithm>
#include <ranges>

namespace gs {

// Just pure functions - no classes!
namespace broadcast {

// Get broadcast shape for two shapes
inline std::vector<size_t> shape(std::span<const size_t> a, std::span<const size_t> b) {
    size_t max_rank = std::max(a.size(), b.size());
    std::vector<size_t> result(max_rank);

    // Work backwards (numpy-style)
    for (size_t i = 0; i < max_rank; ++i) {
        size_t dim_a = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        size_t dim_b = (i < b.size()) ? b[b.size() - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return {}; // Incompatible
        }
        result[max_rank - 1 - i] = std::max(dim_a, dim_b);
    }
    return result;
}

// Map output index to input index
inline size_t index(size_t out_idx, std::span<const size_t> out_shape,
                   std::span<const size_t> in_shape) {
    size_t in_idx = 0;
    size_t stride = 1;

    for (int i = out_shape.size() - 1; i >= 0; --i) {
        size_t coord = (out_idx / stride) % out_shape[i];
        stride *= out_shape[i];

        int in_dim = i - (out_shape.size() - in_shape.size());
        if (in_dim >= 0) {
            size_t in_coord = (in_shape[in_dim] == 1) ? 0 : coord;
            size_t in_stride = 1;
            for (size_t j = in_dim + 1; j < in_shape.size(); ++j) {
                in_stride *= in_shape[j];
            }
            in_idx += in_coord * in_stride;
        }
    }
    return in_idx;
}

} // namespace broadcast

// Single function to handle all broadcasting
Tensor broadcast_to(const Tensor& src, const TensorShape& target);

} // namespace gs