/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_broadcast.hpp"
#include "core/tensor_ops.hpp"
#include <execution>

namespace gs {

// CPU expansion implementation
void BroadcastExpander::expand_cpu(const Tensor& src, Tensor& dst) {
    auto src_shape = src.shape().dims();
    auto dst_shape = dst.shape().dims();
    size_t total = dst.numel();

    if (src.dtype() == DataType::Bool) {
        const auto* src_data = src.ptr<unsigned char>();
        auto* dst_data = dst.ptr<unsigned char>();

        #ifdef __cpp_lib_parallel_algorithm
        std::for_each(std::execution::par_unseq,
                     std::views::iota(0uz, total).begin(),
                     std::views::iota(0uz, total).end(),
                     [&](size_t i) {
            size_t src_idx = BroadcastUtil::map_broadcast_index(i, dst_shape, src_shape);
            dst_data[i] = src_data[src_idx];
        });
        #else
        for (size_t i = 0; i < total; ++i) {
            size_t src_idx = BroadcastUtil::map_broadcast_index(i, dst_shape, src_shape);
            dst_data[i] = src_data[src_idx];
        }
        #endif
    } else {
        const float* src_data = src.ptr<float>();
        float* dst_data = dst.ptr<float>();

        #ifdef __cpp_lib_parallel_algorithm
        std::for_each(std::execution::par_unseq,
                     std::views::iota(0uz, total).begin(),
                     std::views::iota(0uz, total).end(),
                     [&](size_t i) {
            size_t src_idx = BroadcastUtil::map_broadcast_index(i, dst_shape, src_shape);
            dst_data[i] = src_data[src_idx];
        });
        #else
        for (size_t i = 0; i < total; ++i) {
            size_t src_idx = BroadcastUtil::map_broadcast_index(i, dst_shape, src_shape);
            dst_data[i] = src_data[src_idx];
        }
        #endif
    }
}

// Bridge to CUDA kernels
void BroadcastExpander::launch_broadcast_kernel(const Tensor& src, Tensor& dst) {
    if (src.dtype() == DataType::Bool) {
        tensor_ops::launch_broadcast_bool(
            src.ptr<unsigned char>(), dst.ptr<unsigned char>(),
            src.shape().dims().data(), dst.shape().dims().data(),
            src.shape().rank(), dst.shape().rank(),
            dst.numel(), 0);
    } else {
        tensor_ops::launch_broadcast(
            src.ptr<float>(), dst.ptr<float>(),
            src.shape().dims().data(), dst.shape().dims().data(),
            src.shape().rank(), dst.shape().rank(),
            dst.numel(), 0);
    }
    cudaDeviceSynchronize();
}

} // namespace gs