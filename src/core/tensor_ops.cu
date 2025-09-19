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

    // ============= Unified Unary Operation Kernels =============

    template<typename T>
    __device__ inline T unary_op_impl(T x, UnaryOp op) {
        switch (op) {
            case UnaryOp::Neg: return -x;
            case UnaryOp::Abs: return fabsf(x);
            case UnaryOp::Sign: return (x > 0) - (x < 0);
            case UnaryOp::Reciprocal: return 1.0f / (x + 1e-8f);

            case UnaryOp::Exp: return expf(x);
            case UnaryOp::Exp2: return exp2f(x);
            case UnaryOp::Log: return logf(fmaxf(1e-10f, x));
            case UnaryOp::Log2: return log2f(fmaxf(1e-10f, x));
            case UnaryOp::Log10: return log10f(fmaxf(1e-10f, x));
            case UnaryOp::Log1p: return log1pf(x);

            case UnaryOp::Sqrt: return sqrtf(fmaxf(0.0f, x));
            case UnaryOp::Rsqrt: return rsqrtf(fmaxf(1e-10f, x));
            case UnaryOp::Square: return x * x;

            case UnaryOp::Sin: return sinf(x);
            case UnaryOp::Cos: return cosf(x);
            case UnaryOp::Tan: return tanf(x);
            case UnaryOp::Asin: return asinf(fminf(fmaxf(x, -1.0f), 1.0f));
            case UnaryOp::Acos: return acosf(fminf(fmaxf(x, -1.0f), 1.0f));
            case UnaryOp::Atan: return atanf(x);

            case UnaryOp::Sinh: return sinhf(x);
            case UnaryOp::Cosh: return coshf(x);
            case UnaryOp::Tanh: return tanhf(x);

            case UnaryOp::Sigmoid: return 1.0f / (1.0f + expf(-x));
            case UnaryOp::Relu: return fmaxf(0.0f, x);
            case UnaryOp::Gelu: {
                // Approximate GELU
                float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x);
                return 0.5f * x * (1.0f + tanhf(inner));
            }
            case UnaryOp::Swish: return x / (1.0f + expf(-x));

            case UnaryOp::Floor: return floorf(x);
            case UnaryOp::Ceil: return ceilf(x);
            case UnaryOp::Round: return roundf(x);
            case UnaryOp::Trunc: return truncf(x);

            default: return x;
        }
    }

    template<typename T>
    __global__ void unified_unary_kernel(const T* input, T* output, size_t n, UnaryOp op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = unary_op_impl(input[idx], op);
        }
    }

    template<typename T>
    __global__ void unified_unary_inplace_kernel(T* data, size_t n, UnaryOp op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = unary_op_impl(data[idx], op);
        }
    }

    // Special kernels for boolean output
    __global__ void unary_to_bool_kernel(const float* input, unsigned char* output, size_t n, UnaryOp op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            bool result = false;
            switch (op) {
                case UnaryOp::IsNan: result = isnan(input[idx]); break;
                case UnaryOp::IsInf: result = isinf(input[idx]); break;
                case UnaryOp::IsFinite: result = isfinite(input[idx]); break;
                default: result = false;
            }
            output[idx] = result ? 1 : 0;
        }
    }

    __global__ void logical_not_bool_kernel(const unsigned char* input, unsigned char* output, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = !input[idx];
        }
    }

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
    template<typename T>
    __global__ void unified_binary_op_kernel(const T* a, const T* b, T* c, size_t n, BinaryOp op) {
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
    template<typename T>
    __global__ void unified_scalar_op_kernel(const T* data, T scalar, T* result, size_t n, BinaryOp op) {
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
    template<typename T>
    __global__ void unified_inplace_op_kernel(T* a, const T* b, size_t n, BinaryOp op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            a[idx] = binary_op_impl(a[idx], b[idx], op);
        }
    }

    template<typename T>
    __global__ void unified_scalar_inplace_kernel(T* data, T scalar, size_t n, BinaryOp op) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = binary_op_impl(data[idx], scalar, op);
        }
    }

    // ============= Unified Ternary Operation Kernels =============

    __global__ void muladd_kernel(const float* a, const float* b, const float* c,
                                  float* output, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = a[idx] * b[idx] + c[idx];
        }
    }

    __global__ void clamp_ternary_kernel(const float* x, const float* min_vals,
                                         const float* max_vals, float* output, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = fminf(fmaxf(x[idx], min_vals[idx]), max_vals[idx]);
        }
    }

    // Simpler clamp for scalar bounds
    __global__ void clamp_kernel(float* data, float min_val, float max_val, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = fminf(fmaxf(data[idx], min_val), max_val);
        }
    }

    // ============= Unified Reduce Operation Kernels =============

    // Generic reduction kernel
    __global__ void reduce_sum_kernel(const float* data, float* partial_sums, size_t n) {
        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;
        unsigned int gridSize = blockDim.x * gridDim.x;

        float mySum = 0.0f;

        // Grid-stride loop
        while (i < n) {
            mySum += data[i];
            i += gridSize;
        }

        sdata[tid] = mySum;
        __syncthreads();

        // Reduce in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            partial_sums[blockIdx.x] = sdata[0];
        }
    }

    __global__ void reduce_max_kernel(const float* data, float* partial_maxs, size_t n) {
        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;
        unsigned int gridSize = blockDim.x * gridDim.x;

        float myMax = -FLT_MAX;

        while (i < n) {
            myMax = fmaxf(myMax, data[i]);
            i += gridSize;
        }

        sdata[tid] = myMax;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            partial_maxs[blockIdx.x] = sdata[0];
        }
    }

    __global__ void reduce_min_kernel(const float* data, float* partial_mins, size_t n) {
        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;
        unsigned int gridSize = blockDim.x * gridDim.x;

        float myMin = FLT_MAX;

        while (i < n) {
            myMin = fminf(myMin, data[i]);
            i += gridSize;
        }

        sdata[tid] = myMin;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            partial_mins[blockIdx.x] = sdata[0];
        }
    }

    __global__ void reduce_prod_kernel(const float* data, float* partial_prods, size_t n) {
        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + tid;
        unsigned int gridSize = blockDim.x * gridDim.x;

        float myProd = 1.0f;

        while (i < n) {
            myProd *= data[i];
            i += gridSize;
        }

        sdata[tid] = myProd;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] *= sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            partial_prods[blockIdx.x] = sdata[0];
        }
    }

    // ============= Type Conversion Kernels =============
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

    // ============= Launch Functions Implementation =============

    // Unary operations
    void launch_unary_op(const void* input, void* output,
                        size_t n, UnaryOp op,
                        DataType dtype, cudaStream_t stream) {
        if (n == 0) return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        if (op == UnaryOp::LogicalNot && dtype == DataType::Bool) {
            logical_not_bool_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const unsigned char*>(input),
                static_cast<unsigned char*>(output), n);
        } else if ((op == UnaryOp::IsNan || op == UnaryOp::IsInf || op == UnaryOp::IsFinite)
                   && dtype == DataType::Float32) {
            unary_to_bool_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input),
                static_cast<unsigned char*>(output), n, op);
        } else if (dtype == DataType::Float32) {
            unified_unary_kernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output), n, op);
        } else if (dtype == DataType::Int32) {
            // For int32, we'd need specialized implementations
            // For now, just copy
            cudaMemcpyAsync(output, input, n * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        }
    }

    void launch_unary_op_inplace(void* data, size_t n,
                                UnaryOp op, DataType dtype,
                                cudaStream_t stream) {
        if (n == 0) return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        if (dtype == DataType::Float32) {
            unified_unary_inplace_kernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<float*>(data), n, op);
        }
    }

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
        } else if (a_dtype == DataType::Float32 && b_dtype == DataType::Float32 && c_dtype == DataType::Float32) {
            // Arithmetic operations on float
            unified_binary_op_kernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(a),
                static_cast<const float*>(b),
                static_cast<float*>(c),
                n, op);
        } else if (a_dtype == DataType::Int32 && b_dtype == DataType::Int32 && c_dtype == DataType::Int32) {
            // Arithmetic operations on int
            // Would need int-specific binary_op_impl
            // For now, just copy a to c
            cudaMemcpyAsync(c, a, n * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        }
    }

    void launch_binary_scalar(const void* data, float scalar, void* result,
                             size_t n, BinaryOp op,
                             DataType src_dtype, DataType dst_dtype,
                             cudaStream_t stream) {
        if (n == 0) return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        if (dst_dtype == DataType::Bool && src_dtype == DataType::Float32) {
            // Comparison operations
            unified_scalar_comparison_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(data),
                scalar,
                static_cast<unsigned char*>(result),
                n, op);
        } else if (src_dtype == DataType::Float32 && dst_dtype == DataType::Float32) {
            // Arithmetic operations
            unified_scalar_op_kernel<float><<<grid_size, block_size, 0, stream>>>(
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

        unified_inplace_op_kernel<float><<<grid_size, block_size, 0, stream>>>(
            static_cast<float*>(a),
            static_cast<const float*>(b),
            n, op);
    }

    void launch_binary_scalar_inplace(void* data, float scalar, size_t n,
                                     BinaryOp op, cudaStream_t stream) {
        if (n == 0) return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        unified_scalar_inplace_kernel<float><<<grid_size, block_size, 0, stream>>>(
            static_cast<float*>(data),
            scalar,
            n, op);
    }

    // Ternary operations
    void launch_ternary_op(const void* a, const void* b, const void* c, void* output,
                          size_t n, TernaryOp op,
                          DataType dtype, cudaStream_t stream) {
        if (n == 0) return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        switch (op) {
            case TernaryOp::MulAdd:
                muladd_kernel<<<grid_size, block_size, 0, stream>>>(
                    static_cast<const float*>(a),
                    static_cast<const float*>(b),
                    static_cast<const float*>(c),
                    static_cast<float*>(output), n);
                break;
            case TernaryOp::Clamp:
                clamp_ternary_kernel<<<grid_size, block_size, 0, stream>>>(
                    static_cast<const float*>(a),
                    static_cast<const float*>(b),
                    static_cast<const float*>(c),
                    static_cast<float*>(output), n);
                break;
            default:
                break;
        }
    }

    // Reduce operations - simplified version
    void launch_reduce_op(const void* input, void* output,
                         const size_t* shape, size_t rank,
                         const int* axes, size_t num_axes,
                         bool keepdim, ReduceOp op,
                         DataType dtype, cudaStream_t stream) {
        // This is a simplified implementation
        // A full implementation would handle partial reductions properly

        size_t n = 1;
        for (size_t i = 0; i < rank; ++i) {
            n *= shape[i];
        }

        if (n == 0) return;

        const int threads = 256;
        const int blocks = std::min(1024, (int)((n + threads - 1) / threads));

        float* d_block_results;
        cudaMalloc(&d_block_results, blocks * sizeof(float));

        // Launch appropriate reduction kernel based on op
        switch (op) {
            case ReduceOp::Sum:
            case ReduceOp::Mean:
                reduce_sum_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
                    static_cast<const float*>(input), d_block_results, n);
                break;
            case ReduceOp::Max:
                reduce_max_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
                    static_cast<const float*>(input), d_block_results, n);
                break;
            case ReduceOp::Min:
                reduce_min_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
                    static_cast<const float*>(input), d_block_results, n);
                break;
            case ReduceOp::Prod:
                reduce_prod_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
                    static_cast<const float*>(input), d_block_results, n);
                break;
            default:
                // For other ops, just copy first element as placeholder
                cudaMemcpyAsync(d_block_results, input, sizeof(float),
                               cudaMemcpyDeviceToDevice, stream);
                break;
        }

        if (stream == 0) {
            cudaDeviceSynchronize();
        }

        // Final reduction on CPU if needed
        if (blocks > 1) {
            std::vector<float> block_results(blocks);
            cudaMemcpy(block_results.data(), d_block_results, blocks * sizeof(float),
                      cudaMemcpyDeviceToHost);

            float final_result = 0.0f;
            switch (op) {
                case ReduceOp::Sum:
                    for (float val : block_results) final_result += val;
                    break;
                case ReduceOp::Mean:
                    for (float val : block_results) final_result += val;
                    final_result /= n;
                    break;
                case ReduceOp::Max:
                    final_result = *std::max_element(block_results.begin(), block_results.end());
                    break;
                case ReduceOp::Min:
                    final_result = *std::min_element(block_results.begin(), block_results.end());
                    break;
                case ReduceOp::Prod:
                    final_result = 1.0f;
                    for (float val : block_results) final_result *= val;
                    break;
            }

            cudaMemcpy(output, &final_result, sizeof(float), cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(output, d_block_results, sizeof(float), cudaMemcpyDeviceToDevice);

            if (op == ReduceOp::Mean) {
                // Divide by n for mean
                float mean_val;
                cudaMemcpy(&mean_val, output, sizeof(float), cudaMemcpyDeviceToHost);
                mean_val /= n;
                cudaMemcpy(output, &mean_val, sizeof(float), cudaMemcpyHostToDevice);
            }
        }

        cudaFree(d_block_results);
    }

    // Type conversions
    void launch_bool_to_float(const unsigned char* src, float* dst, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        bool_to_float_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, n);
    }

    void launch_float_to_bool(const float* src, unsigned char* dst, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        float_to_bool_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, n);
    }

    // Special operations
    void launch_clamp(float* data, float min_val, float max_val, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        clamp_kernel<<<grid_size, block_size, 0, stream>>>(data, min_val, max_val, n);
    }

    // Legacy operations - now just delegate to unified ops
    void launch_abs(float* data, size_t n, cudaStream_t stream) {
        launch_unary_op_inplace(data, n, UnaryOp::Abs, DataType::Float32, stream);
    }

    void launch_sqrt(float* data, size_t n, cudaStream_t stream) {
        launch_unary_op_inplace(data, n, UnaryOp::Sqrt, DataType::Float32, stream);
    }

    void launch_exp(float* data, size_t n, cudaStream_t stream) {
        launch_unary_op_inplace(data, n, UnaryOp::Exp, DataType::Float32, stream);
    }

    void launch_log(float* data, size_t n, cudaStream_t stream) {
        launch_unary_op_inplace(data, n, UnaryOp::Log, DataType::Float32, stream);
    }

    void launch_sigmoid(float* data, size_t n, cudaStream_t stream) {
        launch_unary_op_inplace(data, n, UnaryOp::Sigmoid, DataType::Float32, stream);
    }

    void launch_relu(float* data, size_t n, cudaStream_t stream) {
        launch_unary_op_inplace(data, n, UnaryOp::Relu, DataType::Float32, stream);
    }

    // Logit is special - not a simple unary op
    __global__ void logit_kernel(const float* input, float* output, size_t n, float eps) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float x = input[idx];
            x = fmaxf(fminf(x, 1.0f - eps), eps);
            output[idx] = logf(x / (1.0f - x));
        }
    }

    void launch_logit(const float* input, float* output, size_t n, float eps, cudaStream_t stream) {
        if (n == 0) return;
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        logit_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n, eps);
    }

    // Power scalar - special case
    __global__ void pow_scalar_kernel(float* d, float e, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) d[idx] = powf(d[idx], e);
    }

    void launch_pow_scalar(float* data, float exponent, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        pow_scalar_kernel<<<(n + 255) / 256, 256, 0, stream>>>(data, exponent, n);
    }

    __global__ void int_to_float_kernel(const int* src, float* dst, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = static_cast<float>(src[idx]);
        }
    }

    void launch_int_to_float(const int* src, float* dst, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        int_to_float_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, n);
    }

    // Legacy reduce operations - delegate to unified
    void launch_reduce_sum(const float* data, float* result, size_t n, cudaStream_t stream) {
        launch_reduce_op(data, result, &n, 1, nullptr, 0, false,
                        ReduceOp::Sum, DataType::Float32, stream);
    }

    void launch_reduce_mean(const float* data, float* result, size_t n, cudaStream_t stream) {
        launch_reduce_op(data, result, &n, 1, nullptr, 0, false,
                        ReduceOp::Mean, DataType::Float32, stream);
    }

    void launch_reduce_min(const float* data, float* result, size_t n, cudaStream_t stream) {
        launch_reduce_op(data, result, &n, 1, nullptr, 0, false,
                        ReduceOp::Min, DataType::Float32, stream);
    }

    void launch_reduce_max(const float* data, float* result, size_t n, cudaStream_t stream) {
        launch_reduce_op(data, result, &n, 1, nullptr, 0, false,
                        ReduceOp::Max, DataType::Float32, stream);
    }

    // Movement operations stub (would be implemented fully)
    void launch_movement_op(const void* input, void* output,
                           const size_t* input_shape, const size_t* output_shape,
                           size_t input_rank, size_t output_rank,
                           MovementOp op, const void* args,
                           DataType dtype, cudaStream_t stream) {
        // This would handle reshape, permute, etc.
        // For now, just copy
        size_t n = 1;
        for (size_t i = 0; i < input_rank; ++i) {
            n *= input_shape[i];
        }

        if (n > 0) {
            size_t bytes = n * dtype_size(dtype);
            cudaMemcpyAsync(output, input, bytes, cudaMemcpyDeviceToDevice, stream);
        }
    }

    __global__ void fill_const_kernel(float* data, float value, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = value;
        }
    }
    // Load operations stub
    void launch_load_op(void* output, const size_t* shape, size_t rank,
                   LoadOp op, const void* args,
                   DataType dtype, cudaStream_t stream) {
        if (op == LoadOp::Const && dtype == DataType::Float32) {
            float value = *static_cast<const float*>(args);

            // Calculate total elements
            size_t n = 1;
            for (size_t i = 0; i < rank; ++i) {
                n *= shape[i];
            }

            if (n > 0) {
                int block_size = 256;
                int grid_size = (n + block_size - 1) / block_size;
                fill_const_kernel<<<grid_size, block_size, 0, stream>>>(
                    static_cast<float*>(output), value, n);
            }
        }
        // Handle other load operations...
    }

} // namespace gs::tensor_ops