/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_broadcast.hpp"
#include "core/tensor_ops.hpp"
#include <execution>

namespace gs {

Tensor broadcast_to(const Tensor& src, const TensorShape& target) {
    if (src.shape() == target) return src.clone();

    // CRITICAL FIX: Handle empty tensors
    if (src.numel() == 0 || target.elements() == 0) {
        // If source is empty or target is empty, check compatibility
        auto bcast = broadcast::shape(src.shape().dims(), target.dims());
        if (bcast.empty() || bcast != target.dims()) {
            LOG_ERROR("Cannot broadcast empty tensor {} to {}", src.shape().str(), target.str());
            return {};
        }
        // Return empty tensor with target shape
        return Tensor::empty(target, src.device(), src.dtype());
    }

    // Check if valid broadcast
    auto bcast = broadcast::shape(src.shape().dims(), target.dims());
    if (bcast.empty() || bcast != target.dims()) {
        LOG_ERROR("Cannot broadcast {} to {}", src.shape().str(), target.str());
        return {};
    }

    auto result = Tensor::empty(target, src.device(), src.dtype());

    if (src.device() == Device::CUDA) {
        if (src.dtype() == DataType::Bool) {
            tensor_ops::launch_broadcast_bool(
                src.ptr<unsigned char>(), result.ptr<unsigned char>(),
                src.shape().dims().data(), target.dims().data(),
                src.shape().rank(), target.rank(),
                result.numel(), 0);
        } else {
            tensor_ops::launch_broadcast(
                src.ptr<float>(), result.ptr<float>(),
                src.shape().dims().data(), target.dims().data(),
                src.shape().rank(), target.rank(),
                result.numel(), 0);
        }
        cudaDeviceSynchronize();
    } else {
        // CPU implementation
        size_t total = result.numel();
        auto src_shape = src.shape().dims();
        auto dst_shape = target.dims();

        if (src.dtype() == DataType::Bool) {
            const auto* src_data = src.ptr<unsigned char>();
            auto* dst_data = result.ptr<unsigned char>();

            #pragma omp parallel for if(total > 1024)
            for (size_t i = 0; i < total; ++i) {
                dst_data[i] = src_data[broadcast::index(i, dst_shape, src_shape)];
            }
        } else {
            const float* src_data = src.ptr<float>();
            float* dst_data = result.ptr<float>();

            #pragma omp parallel for if(total > 1024)
            for (size_t i = 0; i < total; ++i) {
                dst_data[i] = src_data[broadcast::index(i, dst_shape, src_shape)];
            }
        }
    }

    return result;
}

} // namespace gs