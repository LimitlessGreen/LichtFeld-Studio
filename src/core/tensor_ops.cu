/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include "core/tensor.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

namespace gs::tensor_ops {

    // ============= Unified Binary Operation Kernels =============

    template<typename T>
    __device__ inline T binary_op_impl(T a, T b, BinaryOp op) {
        switch (op) {
            case BinaryOp::Add: return a + b;
            case BinaryOp::Sub: return a - b;
            case BinaryOp::Mul: return a * b;
            case BinaryOp::Div: return a / (b + T(1e-8));
            case BinaryOp::Pow: return powf(a, b);
            case BinaryOp::Mod: return fmodf(a, b);
            case BinaryOp::Maximum: return fmaxf(a, b);
            case BinaryOp::Minimum: return fminf(a, b);
            default: return T(0);
        }
    }

    template<typename T>
    __device__ inline bool comparison_op_impl(T a, T b, BinaryOp op) {
        switch (op) {
            case BinaryOp::Equal: return a == b;
            case BinaryOp::NotEqual: return a != b;
            case BinaryOp::Less: return a < b;
            case BinaryOp::LessEqual: return a <= b;
            case BinaryOp::Greater: return a > b;
            case BinaryOp::GreaterEqual: return a >= b;
            default: return false;
        }
    }

    __device__ inline bool logical_op_impl(bool a, bool b, BinaryOp op) {
        switch (op) {
            case BinaryOp::LogicalAnd: return a && b;
            case BinaryOp::LogicalOr: return a || b;
            case BinaryOp::LogicalXor: return a != b;
            default: return false;
        }
    }

    // Unified binary operation kernel
    __global__ void unified_binary_op_kernel(const float* a, const float* b, float* c,
                                            size_t n, BinaryOp op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = binary_op_impl(a[idx], b[idx], op);
        }
    }

    // Unified binary operation kernel with output as bool
    __global__ void unified_comparison_kernel(const float* a, const float* b,
                                             unsigned char* c, size_t n, BinaryOp op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = comparison_op_impl(a[idx], b[idx], op) ? 1 : 0;
        }
    }

    // Unified logical operation kernel
    __global__ void unified_logical_kernel(const unsigned char* a, const unsigned char* b,
                                          unsigned char* c, size_t n, BinaryOp op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = logical_op_impl(a[idx] != 0, b[idx] != 0, op) ? 1 : 0;
        }
    }

    // Scalar operation kernels
    __global__ void unified_scalar_op_kernel(const float* data, float scalar,
                                            float* result, size_t n, BinaryOp op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            result[idx] = binary_op_impl(data[idx], scalar, op);
        }
    }

    __global__ void unified_scalar_comparison_kernel(const float* data, float scalar,
                                                    unsigned char* result, size_t n, BinaryOp op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            result[idx] = comparison_op_impl(data[idx], scalar, op) ? 1 : 0;
        }
    }

    // In-place operation kernels
    __global__ void unified_inplace_op_kernel(float* a, const float* b, size_t n, BinaryOp op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            a[idx] = binary_op_impl(a[idx], b[idx], op);
        }
    }

    __global__ void unified_scalar_inplace_kernel(float* data, float scalar, size_t n, BinaryOp op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = binary_op_impl(data[idx], scalar, op);
        }
    }

    // ============= Launch Functions Implementation =============

    void launch_binary_op(const void* a, const void* b, void* c,
                         size_t n, BinaryOp op,
                         DataType a_dtype, DataType b_dtype, DataType c_dtype,
                         cudaStream_t stream) {
        if (n == 0) return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        // Dispatch based on operation type and data types
        if (c_dtype == DataType::Bool && a_dtype == DataType::Float32) {
            // Comparison operations
            unified_comparison_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(a),
                static_cast<const float*>(b),
                static_cast<unsigned char*>(c),
                n, op);
        } else if (a_dtype == DataType::Bool && b_dtype == DataType::Bool && c_dtype == DataType::Bool) {
            // Logical operations
            unified_logical_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const unsigned char*>(a),
                static_cast<const unsigned char*>(b),
                static_cast<unsigned char*>(c),
                n, op);
        } else {
            // Arithmetic operations
            unified_binary_op_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(a),
                static_cast<const float*>(b),
                static_cast<float*>(c),
                n, op);
        }
    }

    void launch_binary_scalar(const void* data, float scalar, void* result,
                             size_t n, BinaryOp op,
                             DataType src_dtype, DataType dst_dtype,
                             cudaStream_t stream) {
        if (n == 0) return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        if (dst_dtype == DataType::Bool) {
            // Comparison operations
            unified_scalar_comparison_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(data),
                scalar,
                static_cast<unsigned char*>(result),
                n, op);
        } else {
            // Arithmetic operations
            unified_scalar_op_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(data),
                scalar,
                static_cast<float*>(result),
                n, op);
        }
    }

    void launch_binary_op_inplace(void* a, const void* b, size_t n,
                                 BinaryOp op, cudaStream_t stream) {
        if (n == 0) return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        unified_inplace_op_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<float*>(a),
            static_cast<const float*>(b),
            n, op);
    }

    void launch_binary_scalar_inplace(void* data, float scalar, size_t n,
                                     BinaryOp op, cudaStream_t stream) {
        if (n == 0) return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        unified_scalar_inplace_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<float*>(data),
            scalar,
            n, op);
    }

    // ============= Legacy kernels (kept for compatibility) =============

    __global__ void bool_to_float_kernel(const unsigned char* src, float* dst, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = src[idx] ? 1.0f : 0.0f;
        }
    }

    __global__ void float_to_bool_kernel(const float* src, unsigned char* dst, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = (src[idx] != 0.0f) ? 1 : 0;
        }
    }

    __global__ void abs_kernel(float* data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = fabsf(data[idx]);
        }
    }

    __global__ void sqrt_kernel(float* data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = sqrtf(fmaxf(0.0f, data[idx]));
        }
    }

    __global__ void exp_kernel(float* data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = expf(data[idx]);
        }
    }

    __global__ void log_kernel(float* data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = logf(fmaxf(1e-10f, data[idx]));
        }
    }

    __global__ void sigmoid_kernel(float* data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float val = data[idx];
            data[idx] = 1.0f / (1.0f + expf(-val));
        }
    }

    __global__ void logit_kernel(const float* input, float* output, size_t n, float eps) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float x = input[idx];
            x = fmaxf(fminf(x, 1.0f - eps), eps);
            output[idx] = logf(x / (1.0f - x));
        }
    }

    __global__ void relu_kernel(float* data, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = fmaxf(0.0f, data[idx]);
        }
    }

    __global__ void clamp_kernel(float* data, float min_val, float max_val, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = fminf(fmaxf(data[idx], min_val), max_val);
        }
    }

    // Simple reduction kernel that works correctly with grid-stride loop
    __global__ void reduce_sum_kernel_simple(const float* data, float* partial_sums, size_t n) {
        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;
        unsigned int gridSize = blockDim.x * gridDim.x;

        float mySum = 0.0f;
        while (i < n) {
            mySum += data[i];
            i += gridSize;
        }

        sdata[tid] = mySum;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0)
            partial_sums[blockIdx.x] = sdata[0];
    }

    __global__ void reduce_min_kernel_simple(const float* data, float* partial_mins, size_t n) {
        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;

        float myMin = (i < n) ? data[i] : FLT_MAX;
        sdata[tid] = myMin;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0)
            partial_mins[blockIdx.x] = sdata[0];
    }

    __global__ void reduce_max_kernel_simple(const float* data, float* partial_maxs, size_t n) {
        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;

        float myMax = (i < n) ? data[i] : -FLT_MAX;
        sdata[tid] = myMax;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0)
            partial_maxs[blockIdx.x] = sdata[0];
    }

    // ============= Launch Functions for Legacy Operations =============

    void launch_bool_to_float(const unsigned char* src, float* dst, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        bool_to_float_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, n);
    }

    void launch_float_to_bool(const float* src, unsigned char* dst, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        float_to_bool_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, n);
    }

    void launch_abs(float* data, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        abs_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_sqrt(float* data, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        sqrt_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_exp(float* data, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        exp_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_log(float* data, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        log_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_sigmoid(float* data, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        sigmoid_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_logit(const float* input, float* output, size_t n, float eps, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        logit_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n, eps);
    }

    void launch_relu(float* data, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        relu_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
    }

    void launch_clamp(float* data, float min_val, float max_val, size_t n, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        clamp_kernel<<<grid_size, block_size, 0, stream>>>(data, min_val, max_val, n);
    }

    void launch_reduce_sum(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            *result = 0.0f;
            return;
        }

        const int threads = 256;
        const int blocks = std::min(1024, (int)((n + threads - 1) / threads));

        float* d_block_results;
        cudaMalloc(&d_block_results, blocks * sizeof(float));

        reduce_sum_kernel_simple<<<blocks, threads, threads * sizeof(float), stream>>>(
            data, d_block_results, n);

        if (stream == 0) {
            cudaDeviceSynchronize();
        }

        if (blocks > 1) {
            std::vector<float> block_results(blocks);
            cudaMemcpy(block_results.data(), d_block_results, blocks * sizeof(float), cudaMemcpyDeviceToHost);
            float sum = 0.0f;
            for (float val : block_results) {
                sum += val;
            }
            *result = sum;
        } else {
            cudaMemcpy(result, d_block_results, sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_block_results);
    }

    void launch_reduce_mean(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            *result = 0.0f;
            return;
        }

        float sum = 0.0f;
        launch_reduce_sum(data, &sum, n, stream);
        *result = sum / (float)n;
    }

    void launch_reduce_min(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            *result = 0.0f;
            return;
        }

        const int threads = 256;
        const int blocks = std::min(1024, (int)((n + threads - 1) / threads));

        float* d_block_results;
        cudaMalloc(&d_block_results, blocks * sizeof(float));

        reduce_min_kernel_simple<<<blocks, threads, threads * sizeof(float), stream>>>(
            data, d_block_results, n);

        if (stream == 0) {
            cudaDeviceSynchronize();
        }

        if (blocks > 1) {
            std::vector<float> block_results(blocks);
            cudaMemcpy(block_results.data(), d_block_results, blocks * sizeof(float), cudaMemcpyDeviceToHost);
            *result = *std::min_element(block_results.begin(), block_results.end());
        } else {
            cudaMemcpy(result, d_block_results, sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_block_results);
    }

    void launch_reduce_max(const float* data, float* result, size_t n, cudaStream_t stream) {
        if (n == 0) {
            *result = 0.0f;
            return;
        }

        const int threads = 256;
        const int blocks = std::min(1024, (int)((n + threads - 1) / threads));

        float* d_block_results;
        cudaMalloc(&d_block_results, blocks * sizeof(float));

        reduce_max_kernel_simple<<<blocks, threads, threads * sizeof(float), stream>>>(
            data, d_block_results, n);

        if (stream == 0) {
            cudaDeviceSynchronize();
        }

        if (blocks > 1) {
            std::vector<float> block_results(blocks);
            cudaMemcpy(block_results.data(), d_block_results, blocks * sizeof(float), cudaMemcpyDeviceToHost);
            *result = *std::max_element(block_results.begin(), block_results.end());
        } else {
            cudaMemcpy(result, d_block_results, sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_block_results);
    }

} // namespace gs::tensor_ops