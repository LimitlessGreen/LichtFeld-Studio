/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include <cuda_runtime.h>

namespace gs {
namespace tensor_ops {

    template<typename T>
    __global__ void strided_copy_kernel(
        const T* __restrict__ input,
        T* __restrict__ output,
        const size_t* __restrict__ shape,
        const size_t* __restrict__ strides,
        size_t rank,
        size_t total_elements) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements)
            return;

        // Convert flat output index to multi-dimensional coordinates
        size_t tmp = idx;
        size_t input_offset = 0;

        for (int d = rank - 1; d >= 0; --d) {
            size_t coord = tmp % shape[d];
            tmp /= shape[d];
            input_offset += coord * strides[d];
        }

        // Copy element from strided input to contiguous output
        output[idx] = input[input_offset];
    }

    void launch_strided_copy(
        const void* input,
        void* output,
        const size_t* shape,
        const size_t* strides,
        size_t rank,
        size_t total_elements,
        DataType dtype,
        cudaStream_t stream) {
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;

        switch (dtype) {
        case DataType::Float32:
            strided_copy_kernel<<<num_blocks, block_size, 0, stream>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                shape, strides, rank, total_elements);
            break;
        case DataType::Int32:
            strided_copy_kernel<<<num_blocks, block_size, 0, stream>>>(
                static_cast<const int32_t*>(input),
                static_cast<int32_t*>(output),
                shape, strides, rank, total_elements);
            break;
        case DataType::Int64:
            strided_copy_kernel<<<num_blocks, block_size, 0, stream>>>(
                static_cast<const int64_t*>(input),
                static_cast<int64_t*>(output),
                shape, strides, rank, total_elements);
            break;
        case DataType::UInt8:
            strided_copy_kernel<<<num_blocks, block_size, 0, stream>>>(
                static_cast<const uint8_t*>(input),
                static_cast<uint8_t*>(output),
                shape, strides, rank, total_elements);
            break;
        case DataType::Bool:
            strided_copy_kernel<<<num_blocks, block_size, 0, stream>>>(
                static_cast<const bool*>(input),
                static_cast<bool*>(output),
                shape, strides, rank, total_elements);
            break;
        default:
            // Unsupported dtype - do nothing
            break;
        }
    }

}  // namespace tensor_ops
}  // namespace gs
