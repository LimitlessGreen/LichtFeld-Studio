/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>

// Thrust headers
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace gs::tensor_ops {

    // Helper template to execute Thrust operations with correct policy
    template<typename Func>
    void run_with_thrust_policy(cudaStream_t stream, Func&& func) {
        if (stream) {
            func(thrust::cuda::par.on(stream));
        } else {
            func(thrust::cuda::par);
        }
    }

    // ============= CLAMP SCALAR OPERATIONS =============

    struct ClampScalarFunctor {
        float min_val;
        float max_val;

        ClampScalarFunctor(float min, float max) : min_val(min), max_val(max) {}

        __device__ float operator()(float x) const {
            if (isnan(x)) return x;
            return fmax(min_val, fmin(max_val, x));
        }
    };

    struct ClampScalarIntFunctor {
        int min_val;
        int max_val;

        ClampScalarIntFunctor(int min, int max) : min_val(min), max_val(max) {}

        __device__ int operator()(int x) const {
            return max(min_val, min(max_val, x));
        }
    };

    void launch_clamp_scalar(float* data, float min_val, float max_val, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        auto data_ptr = thrust::device_pointer_cast(data);
        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, data_ptr, data_ptr + n, data_ptr,
                            ClampScalarFunctor(min_val, max_val));
        });
    }

    void launch_clamp_scalar_int(int* data, int min_val, int max_val, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        auto data_ptr = thrust::device_pointer_cast(data);
        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, data_ptr, data_ptr + n, data_ptr,
                            ClampScalarIntFunctor(min_val, max_val));
        });
    }

    // ============= UNARY OPERATION FUNCTORS =============

    template <typename T>
    struct UnaryOpFunctor {
        UnaryOp op;
        UnaryOpFunctor(UnaryOp op_) : op(op_) {}

        __device__ T operator()(T x) const {
            switch (op) {
            case UnaryOp::Neg: return -x;
            case UnaryOp::Abs: return abs(x);
            case UnaryOp::Sign: return (x > T(0)) - (x < T(0));
            case UnaryOp::Reciprocal: return T(1) / (x + T(1e-8));
            case UnaryOp::Exp: return exp(x);
            case UnaryOp::Exp2: return exp2(x);
            case UnaryOp::Log: return log(fmax(x, T(1e-10)));
            case UnaryOp::Log2: return log2(fmax(x, T(1e-10)));
            case UnaryOp::Log10: return log10(fmax(x, T(1e-10)));
            case UnaryOp::Log1p: return log1p(x);
            case UnaryOp::Sqrt: return sqrt(fmax(x, T(0)));
            case UnaryOp::Rsqrt: return rsqrt(fmax(x, T(1e-10)));
            case UnaryOp::Square: return x * x;
            case UnaryOp::Sin: return sin(x);
            case UnaryOp::Cos: return cos(x);
            case UnaryOp::Tan: return tan(x);
            case UnaryOp::Asin: return asin(fmin(fmax(x, T(-1)), T(1)));
            case UnaryOp::Acos: return acos(fmin(fmax(x, T(-1)), T(1)));
            case UnaryOp::Atan: return atan(x);
            case UnaryOp::Sinh: return sinh(x);
            case UnaryOp::Cosh: return cosh(x);
            case UnaryOp::Tanh: return tanh(x);
            case UnaryOp::Sigmoid: return T(1) / (T(1) + exp(-x));
            case UnaryOp::Relu: return fmax(x, T(0));
            case UnaryOp::Gelu: {
                T inner = sqrt(T(2) / T(M_PI)) * (x + T(0.044715) * x * x * x);
                return T(0.5) * x * (T(1) + tanh(inner));
            }
            case UnaryOp::Swish: return x / (T(1) + exp(-x));
            case UnaryOp::Floor: return floor(x);
            case UnaryOp::Ceil: return ceil(x);
            case UnaryOp::Round: return round(x);
            case UnaryOp::Trunc: return trunc(x);
            default: return x;
            }
        }
    };

    struct LogicalNotFunctor {
        __device__ unsigned char operator()(unsigned char x) const {
            return !x;
        }
    };

    void launch_unary_op(const void* input, void* output, size_t n, UnaryOp op,
                         DataType dtype, cudaStream_t stream) {
        if (n == 0) return;

        if (dtype == DataType::Float32) {
            auto in_ptr = thrust::device_pointer_cast(static_cast<const float*>(input));
            auto out_ptr = thrust::device_pointer_cast(static_cast<float*>(output));
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, in_ptr, in_ptr + n, out_ptr, UnaryOpFunctor<float>(op));
            });
        } else if (dtype == DataType::Bool && op == UnaryOp::LogicalNot) {
            auto in_ptr = thrust::device_pointer_cast(static_cast<const unsigned char*>(input));
            auto out_ptr = thrust::device_pointer_cast(static_cast<unsigned char*>(output));
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, in_ptr, in_ptr + n, out_ptr, LogicalNotFunctor());
            });
        }
    }

    void launch_unary_op_inplace(void* data, size_t n, UnaryOp op, DataType dtype, cudaStream_t stream) {
        launch_unary_op(data, data, n, op, dtype, stream);
    }

    // ============= BINARY OPERATION FUNCTORS =============

    template <typename T>
    struct BinaryOpFunctor {
        BinaryOp op;
        BinaryOpFunctor(BinaryOp op_) : op(op_) {}

        __device__ T operator()(T a, T b) const {
            switch (op) {
                case BinaryOp::Add: return a + b;
                case BinaryOp::Sub: return a - b;
                case BinaryOp::Mul: return a * b;
                case BinaryOp::Div: return a / b;
                case BinaryOp::Pow: return powf(a, b);
                case BinaryOp::Mod: return fmodf(a, b);
                case BinaryOp::Maximum: return fmaxf(a, b);
                case BinaryOp::Minimum: return fminf(a, b);
                default: return T(0);
            }
        }
    };

    template <typename T>
    struct ComparisonOpFunctor {
        BinaryOp op;
        ComparisonOpFunctor(BinaryOp op_) : op(op_) {}

        __device__ unsigned char operator()(T a, T b) const {
            bool result = false;
            switch (op) {
            case BinaryOp::Equal: result = (a == b); break;
            case BinaryOp::NotEqual: result = (a != b); break;
            case BinaryOp::Less: result = (a < b); break;
            case BinaryOp::LessEqual: result = (a <= b); break;
            case BinaryOp::Greater: result = (a > b); break;
            case BinaryOp::GreaterEqual: result = (a >= b); break;
            default: result = false; break;
            }
            return result ? 1 : 0;
        }
    };

    struct LogicalOpFunctor {
        BinaryOp op;
        LogicalOpFunctor(BinaryOp op_) : op(op_) {}

        __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
            bool result = false;
            switch (op) {
            case BinaryOp::LogicalAnd: result = (a && b); break;
            case BinaryOp::LogicalOr: result = (a || b); break;
            case BinaryOp::LogicalXor: result = (a != b); break;
            case BinaryOp::BitwiseOr: result = (a || b); break;
            default: result = false; break;
            }
            return result ? 1 : 0;
        }
    };

    void launch_binary_op(const void* a, const void* b, void* c, size_t n, BinaryOp op,
                          DataType a_dtype, DataType b_dtype, DataType c_dtype,
                          cudaStream_t stream) {
        if (n == 0) return;

        if (c_dtype == DataType::Bool && a_dtype == DataType::Int32 && b_dtype == DataType::Int32) {
            auto a_ptr = thrust::device_pointer_cast(static_cast<const int*>(a));
            auto b_ptr = thrust::device_pointer_cast(static_cast<const int*>(b));
            auto c_ptr = thrust::device_pointer_cast(static_cast<unsigned char*>(c));
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, c_ptr, ComparisonOpFunctor<int>(op));
            });
        }
        else if (c_dtype == DataType::Bool && a_dtype == DataType::Float32 && b_dtype == DataType::Float32) {
            auto a_ptr = thrust::device_pointer_cast(static_cast<const float*>(a));
            auto b_ptr = thrust::device_pointer_cast(static_cast<const float*>(b));
            auto c_ptr = thrust::device_pointer_cast(static_cast<unsigned char*>(c));
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, c_ptr, ComparisonOpFunctor<float>(op));
            });
        }
        else if (a_dtype == DataType::Bool && b_dtype == DataType::Bool) {
            auto a_ptr = thrust::device_pointer_cast(static_cast<const unsigned char*>(a));
            auto b_ptr = thrust::device_pointer_cast(static_cast<const unsigned char*>(b));
            auto c_ptr = thrust::device_pointer_cast(static_cast<unsigned char*>(c));
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, c_ptr, LogicalOpFunctor(op));
            });
        }
        else if (a_dtype == DataType::Float32 && b_dtype == DataType::Float32 && c_dtype == DataType::Float32) {
            auto a_ptr = thrust::device_pointer_cast(static_cast<const float*>(a));
            auto b_ptr = thrust::device_pointer_cast(static_cast<const float*>(b));
            auto c_ptr = thrust::device_pointer_cast(static_cast<float*>(c));
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, c_ptr, BinaryOpFunctor<float>(op));
            });
        }
        else if (a_dtype == DataType::Int32 && b_dtype == DataType::Int32 && c_dtype == DataType::Int32) {
            auto a_ptr = thrust::device_pointer_cast(static_cast<const int*>(a));
            auto b_ptr = thrust::device_pointer_cast(static_cast<const int*>(b));
            auto c_ptr = thrust::device_pointer_cast(static_cast<int*>(c));
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, c_ptr, BinaryOpFunctor<int>(op));
            });
        }
    }

    // ============= BINARY SCALAR OPERATIONS =============

    template <typename T>
    struct BinaryScalarOpFunctor {
        T scalar;
        BinaryOp op;
        BinaryScalarOpFunctor(T scalar_, BinaryOp op_) : scalar(scalar_), op(op_) {}

        __device__ T operator()(T a) const {
            switch (op) {
                case BinaryOp::Add: return a + scalar;
                case BinaryOp::Sub: return a - scalar;
                case BinaryOp::Mul: return a * scalar;
                case BinaryOp::Div: return a / scalar;
                case BinaryOp::Pow: {
                    if (scalar == 2.0f) return a * a;
                    if (scalar == 0.5f) return sqrtf(fabsf(a));
                    if (scalar == 1.0f) return a;
                    if (scalar == 0.0f) return 1.0f;
                    return powf(a, scalar);
                }
                case BinaryOp::Mod: return fmodf(a, scalar);
                case BinaryOp::Maximum: return fmaxf(a, scalar);
                case BinaryOp::Minimum: return fminf(a, scalar);
                default: return T(0);
            }
        }
    };

    template <typename T>
    struct ComparisonScalarOpFunctor {
        T scalar;
        BinaryOp op;
        ComparisonScalarOpFunctor(T scalar_, BinaryOp op_) : scalar(scalar_), op(op_) {}

        __device__ unsigned char operator()(T a) const {
            bool result = false;
            switch (op) {
            case BinaryOp::Equal: result = (a == scalar); break;
            case BinaryOp::NotEqual: result = (a != scalar); break;
            case BinaryOp::Less: result = (a < scalar); break;
            case BinaryOp::LessEqual: result = (a <= scalar); break;
            case BinaryOp::Greater: result = (a > scalar); break;
            case BinaryOp::GreaterEqual: result = (a >= scalar); break;
            default: result = false; break;
            }
            return result ? 1 : 0;
        }
    };

    void launch_binary_scalar(const void* data, float scalar, void* result, size_t n, BinaryOp op,
                              DataType src_dtype, DataType dst_dtype, cudaStream_t stream) {
        if (n == 0) return;

        if (dst_dtype == DataType::Bool && src_dtype == DataType::Int32) {
            auto data_ptr = thrust::device_pointer_cast(static_cast<const int*>(data));
            auto result_ptr = thrust::device_pointer_cast(static_cast<unsigned char*>(result));
            int scalar_int = static_cast<int>(scalar);
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, data_ptr, data_ptr + n, result_ptr,
                                ComparisonScalarOpFunctor<int>(scalar_int, op));
            });
        }
        else if (dst_dtype == DataType::Bool && src_dtype == DataType::Float32) {
            auto data_ptr = thrust::device_pointer_cast(static_cast<const float*>(data));
            auto result_ptr = thrust::device_pointer_cast(static_cast<unsigned char*>(result));
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, data_ptr, data_ptr + n, result_ptr,
                                ComparisonScalarOpFunctor<float>(scalar, op));
            });
        }
        else if (src_dtype == DataType::Float32 && dst_dtype == DataType::Float32) {
            auto data_ptr = thrust::device_pointer_cast(static_cast<const float*>(data));
            auto result_ptr = thrust::device_pointer_cast(static_cast<float*>(result));
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, data_ptr, data_ptr + n, result_ptr,
                                BinaryScalarOpFunctor<float>(scalar, op));
            });
        }
        else if (src_dtype == DataType::Int32 && dst_dtype == DataType::Int32) {
            auto data_ptr = thrust::device_pointer_cast(static_cast<const int*>(data));
            auto result_ptr = thrust::device_pointer_cast(static_cast<int*>(result));
            int scalar_int = static_cast<int>(scalar);
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, data_ptr, data_ptr + n, result_ptr,
                                BinaryScalarOpFunctor<int>(scalar_int, op));
            });
        }
    }

    void launch_binary_op_inplace(void* a, const void* b, size_t n, BinaryOp op, cudaStream_t stream) {
        if (n == 0) return;
        auto a_ptr = thrust::device_pointer_cast(static_cast<float*>(a));
        auto b_ptr = thrust::device_pointer_cast(static_cast<const float*>(b));
        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, a_ptr, BinaryOpFunctor<float>(op));
        });
    }

    void launch_binary_scalar_inplace(void* data, float scalar, size_t n, BinaryOp op, cudaStream_t stream) {
        if (n == 0) return;
        auto data_ptr = thrust::device_pointer_cast(static_cast<float*>(data));
        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, data_ptr, data_ptr + n, data_ptr,
                            BinaryScalarOpFunctor<float>(scalar, op));
        });
    }

    // ============= REDUCE OPERATION FUNCTORS =============

    struct ReduceSumFunctor {
        __device__ float operator()(float a, float b) const { return a + b; }
    };

    struct ReduceMaxFunctor {
        __device__ float operator()(float a, float b) const { return fmax(a, b); }
    };

    struct ReduceMinFunctor {
        __device__ float operator()(float a, float b) const { return fmin(a, b); }
    };

    struct ReduceProdFunctor {
        __device__ float operator()(float a, float b) const { return a * b; }
    };

    // ============= OPTIMIZED SEGMENTED REDUCTION =============

    template<typename T, typename Op>
    void launch_segmented_reduce(
        const T* input, T* output,
        size_t outer_size, size_t reduce_size, size_t inner_size,
        T init_value, Op op, cudaStream_t stream)
    {
        if (outer_size == 0 || reduce_size == 0 || inner_size == 0) return;

        size_t output_size = outer_size * inner_size;

        if (reduce_size == 1) {
            auto in_ptr = thrust::device_pointer_cast(input);
            auto out_ptr = thrust::device_pointer_cast(output);
            if (stream) {
                thrust::copy(thrust::cuda::par.on(stream), in_ptr, in_ptr + output_size, out_ptr);
            } else {
                thrust::copy(thrust::cuda::par, in_ptr, in_ptr + output_size, out_ptr);
            }
            return;
        }

        auto out_ptr = thrust::device_pointer_cast(output);
        auto counting = thrust::counting_iterator<size_t>(0);

        if (stream) {
            thrust::transform(thrust::cuda::par.on(stream), counting, counting + output_size, out_ptr,
                [=] __device__ (size_t out_idx) -> T {
                    size_t outer_idx = out_idx / inner_size;
                    size_t inner_idx = out_idx % inner_size;
                    size_t base = outer_idx * reduce_size * inner_size + inner_idx;

                    T result = init_value;
                    if (reduce_size <= 8) {
                        #pragma unroll
                        for (size_t r = 0; r < reduce_size; ++r) {
                            result = op(result, input[base + r * inner_size]);
                        }
                    } else {
                        for (size_t r = 0; r < reduce_size; ++r) {
                            result = op(result, input[base + r * inner_size]);
                        }
                    }
                    return result;
                }
            );
        } else {
            thrust::transform(thrust::cuda::par, counting, counting + output_size, out_ptr,
                [=] __device__ (size_t out_idx) -> T {
                    size_t outer_idx = out_idx / inner_size;
                    size_t inner_idx = out_idx % inner_size;
                    size_t base = outer_idx * reduce_size * inner_size + inner_idx;

                    T result = init_value;
                    if (reduce_size <= 8) {
                        #pragma unroll
                        for (size_t r = 0; r < reduce_size; ++r) {
                            result = op(result, input[base + r * inner_size]);
                        }
                    } else {
                        for (size_t r = 0; r < reduce_size; ++r) {
                            result = op(result, input[base + r * inner_size]);
                        }
                    }
                    return result;
                }
            );
        }
    }

    // ============= MULTI-AXIS REDUCTION HELPER =============

    // Template-based approach to avoid device function pointer issues
    template<int OpType>
    struct MultiAxisReduceOp {
        __device__ float operator()(float a, float b) const;
    };

    template<>
    struct MultiAxisReduceOp<0> { // Sum
        __device__ float operator()(float a, float b) const { return a + b; }
    };

    template<>
    struct MultiAxisReduceOp<1> { // Max
        __device__ float operator()(float a, float b) const { return fmax(a, b); }
    };

    template<>
    struct MultiAxisReduceOp<2> { // Min
        __device__ float operator()(float a, float b) const { return fmin(a, b); }
    };

    template<>
    struct MultiAxisReduceOp<3> { // Prod
        __device__ float operator()(float a, float b) const { return a * b; }
    };

    template<int OpType>
    __global__ void multi_axis_reduce_kernel(
        const float* input, float* output,
        const size_t* input_shape, const bool* is_reduced_dim,
        size_t input_rank, size_t output_elements, float init_val)
    {
        size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (out_idx >= output_elements) return;

        MultiAxisReduceOp<OpType> op;

        size_t input_strides[10];
        input_strides[input_rank - 1] = 1;
        for (int i = input_rank - 2; i >= 0; --i) {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        size_t out_shape[10];
        size_t out_rank = 0;
        for (size_t i = 0; i < input_rank; ++i) {
            if (!is_reduced_dim[i]) {
                out_shape[out_rank++] = input_shape[i];
            }
        }

        size_t output_strides[10];
        if (out_rank > 0) {
            output_strides[out_rank - 1] = 1;
            for (int i = out_rank - 2; i >= 0; --i) {
                output_strides[i] = output_strides[i + 1] * out_shape[i + 1];
            }
        }

        size_t out_coords[10] = {0};
        size_t temp = out_idx;
        for (size_t i = 0; i < out_rank; ++i) {
            out_coords[i] = temp / output_strides[i];
            temp %= output_strides[i];
        }

        size_t base_input_coords[10];
        size_t out_coord_idx = 0;
        for (size_t i = 0; i < input_rank; ++i) {
            base_input_coords[i] = is_reduced_dim[i] ? 0 : out_coords[out_coord_idx++];
        }

        size_t reduce_count = 1;
        size_t reduced_dims[10];
        size_t num_reduced = 0;
        for (size_t i = 0; i < input_rank; ++i) {
            if (is_reduced_dim[i]) {
                reduced_dims[num_reduced++] = i;
                reduce_count *= input_shape[i];
            }
        }

        float result = init_val;
        for (size_t r = 0; r < reduce_count; ++r) {
            size_t temp_r = r;
            size_t full_input_coords[10];

            for (size_t i = 0; i < input_rank; ++i) {
                full_input_coords[i] = base_input_coords[i];
            }

            for (int rd_idx = num_reduced - 1; rd_idx >= 0; --rd_idx) {
                size_t dim = reduced_dims[rd_idx];
                full_input_coords[dim] = temp_r % input_shape[dim];
                temp_r /= input_shape[dim];
            }

            size_t in_idx = 0;
            for (size_t i = 0; i < input_rank; ++i) {
                in_idx += full_input_coords[i] * input_strides[i];
            }

            result = op(result, input[in_idx]);
        }

        output[out_idx] = result;
    }

    // Helper to launch the correct template
    void launch_multi_axis_reduce(
        const float* input, float* output,
        const size_t* input_shape, const bool* is_reduced_dim,
        size_t input_rank, size_t output_elements,
        float init_val, ReduceOp op, cudaStream_t stream)
    {
        int blocks = (output_elements + 255) / 256;

        switch (op) {
            case ReduceOp::Sum:
            case ReduceOp::Mean:
                multi_axis_reduce_kernel<0><<<blocks, 256, 0, stream>>>(
                    input, output, input_shape, is_reduced_dim,
                    input_rank, output_elements, init_val);
                break;
            case ReduceOp::Max:
                multi_axis_reduce_kernel<1><<<blocks, 256, 0, stream>>>(
                    input, output, input_shape, is_reduced_dim,
                    input_rank, output_elements, init_val);
                break;
            case ReduceOp::Min:
                multi_axis_reduce_kernel<2><<<blocks, 256, 0, stream>>>(
                    input, output, input_shape, is_reduced_dim,
                    input_rank, output_elements, init_val);
                break;
            case ReduceOp::Prod:
                multi_axis_reduce_kernel<3><<<blocks, 256, 0, stream>>>(
                    input, output, input_shape, is_reduced_dim,
                    input_rank, output_elements, init_val);
                break;
        }
    }

    // Functor for division (to avoid nested lambda)
    struct DivideByFunctor {
        float divisor;
        DivideByFunctor(float d) : divisor(d) {}
        __device__ float operator()(float x) const { return x / divisor; }
    };

    // ============= MAIN REDUCE OPERATION DISPATCH =============

    void launch_reduce_op(const void* input, void* output, const size_t* shape, size_t rank,
                          const int* axes, size_t num_axes, bool keepdim, ReduceOp op,
                          DataType dtype, cudaStream_t stream) {
        if (dtype != DataType::Float32) return;

        size_t n = 1;
        for (size_t i = 0; i < rank; ++i) n *= shape[i];
        if (n == 0) return;

        auto input_ptr = thrust::device_pointer_cast(static_cast<const float*>(input));
        auto output_ptr = thrust::device_pointer_cast(static_cast<float*>(output));

        // Full reduction to scalar
        if (num_axes == 0 || num_axes == rank) {
            float result = 0.0f;
            run_with_thrust_policy(stream, [&](auto policy) {
                switch (op) {
                case ReduceOp::Sum:
                    result = thrust::reduce(policy, input_ptr, input_ptr + n, 0.0f, ReduceSumFunctor());
                    break;
                case ReduceOp::Mean:
                    result = thrust::reduce(policy, input_ptr, input_ptr + n, 0.0f, ReduceSumFunctor()) / static_cast<float>(n);
                    break;
                case ReduceOp::Max:
                    result = thrust::reduce(policy, input_ptr, input_ptr + n, -std::numeric_limits<float>::infinity(), ReduceMaxFunctor());
                    break;
                case ReduceOp::Min:
                    result = thrust::reduce(policy, input_ptr, input_ptr + n, std::numeric_limits<float>::infinity(), ReduceMinFunctor());
                    break;
                case ReduceOp::Prod:
                    result = thrust::reduce(policy, input_ptr, input_ptr + n, 1.0f, ReduceProdFunctor());
                    break;
                default:
                    result = 0.0f;
                }
            });
            cudaMemcpyAsync(output, &result, sizeof(float), cudaMemcpyHostToDevice, stream);
            return;
        }

        // Single-axis reduction - optimized path
        if (num_axes == 1) {
            int dim = axes[0];
            size_t outer_size = 1;
            for (int i = 0; i < dim; ++i) outer_size *= shape[i];
            size_t reduce_size = shape[dim];
            size_t inner_size = 1;
            for (size_t i = dim + 1; i < rank; ++i) inner_size *= shape[i];

            const float* input_f = static_cast<const float*>(input);
            float* output_f = static_cast<float*>(output);
            float init_val = 0.0f;

            switch (op) {
                case ReduceOp::Sum:
                    init_val = 0.0f;
                    launch_segmented_reduce(input_f, output_f, outer_size, reduce_size, inner_size,
                                          init_val, thrust::plus<float>(), stream);
                    break;
                case ReduceOp::Mean:
                    init_val = 0.0f;
                    launch_segmented_reduce(input_f, output_f, outer_size, reduce_size, inner_size,
                                          init_val, thrust::plus<float>(), stream);
                    {
                        auto out_ptr = thrust::device_pointer_cast(output_f);
                        size_t output_size = outer_size * inner_size;
                        run_with_thrust_policy(stream, [&](auto policy) {
                            thrust::transform(policy, out_ptr, out_ptr + output_size, out_ptr,
                                            DivideByFunctor(static_cast<float>(reduce_size)));
                        });
                    }
                    break;
                case ReduceOp::Max:
                    init_val = -std::numeric_limits<float>::infinity();
                    launch_segmented_reduce(input_f, output_f, outer_size, reduce_size, inner_size,
                                          init_val, thrust::maximum<float>(), stream);
                    break;
                case ReduceOp::Min:
                    init_val = std::numeric_limits<float>::infinity();
                    launch_segmented_reduce(input_f, output_f, outer_size, reduce_size, inner_size,
                                          init_val, thrust::minimum<float>(), stream);
                    break;
                case ReduceOp::Prod:
                    init_val = 1.0f;
                    launch_segmented_reduce(input_f, output_f, outer_size, reduce_size, inner_size,
                                          init_val, thrust::multiplies<float>(), stream);
                    break;
                default:
                    break;
            }
            return;
        }

        // Multi-axis reduction
        thrust::device_vector<bool> d_is_reduced(rank, false);
        for (size_t i = 0; i < num_axes; ++i) {
            d_is_reduced[axes[i]] = true;
        }

        size_t output_elements = 1;
        for (size_t i = 0; i < rank; ++i) {
            if (!d_is_reduced[i] || keepdim) {
                output_elements *= (d_is_reduced[i] ? 1 : shape[i]);
            }
        }

        thrust::device_vector<size_t> d_input_shape(shape, shape + rank);

        float init_val = 0.0f;
        switch (op) {
            case ReduceOp::Sum:
            case ReduceOp::Mean:
                init_val = 0.0f;
                break;
            case ReduceOp::Max:
                init_val = -std::numeric_limits<float>::infinity();
                break;
            case ReduceOp::Min:
                init_val = std::numeric_limits<float>::infinity();
                break;
            case ReduceOp::Prod:
                init_val = 1.0f;
                break;
        }

        // Use the template-based launcher
        launch_multi_axis_reduce(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            thrust::raw_pointer_cast(d_input_shape.data()),
            thrust::raw_pointer_cast(d_is_reduced.data()),
            rank, output_elements, init_val, op, stream
        );

        if (op == ReduceOp::Mean) {
            size_t reduce_count = 1;
            for (size_t i = 0; i < num_axes; ++i) {
                reduce_count *= shape[axes[i]];
            }
            float scale = 1.0f / reduce_count;
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, output_ptr, output_ptr + output_elements,
                                thrust::make_constant_iterator(scale), output_ptr,
                                thrust::multiplies<float>());
            });
        }
    }

    // ============= TERNARY OPERATIONS =============

    struct ClampFunctor {
        __device__ float operator()(const thrust::tuple<float, float, float>& t) const {
            float val = thrust::get<0>(t);
            if (isnan(val)) return val;
            float min_val = thrust::get<1>(t);
            float max_val = thrust::get<2>(t);
            return fmax(min_val, fmin(max_val, val));
        }
    };

    void launch_ternary_op(const void* a, const void* b, const void* c, void* output,
                           size_t n, TernaryOp op, DataType dtype, cudaStream_t stream) {
        if (n == 0 || dtype != DataType::Float32) return;

        if (op == TernaryOp::Clamp) {
            auto a_ptr = thrust::device_pointer_cast(static_cast<const float*>(a));
            auto b_ptr = thrust::device_pointer_cast(static_cast<const float*>(b));
            auto c_ptr = thrust::device_pointer_cast(static_cast<const float*>(c));
            auto out_ptr = thrust::device_pointer_cast(static_cast<float*>(output));

            auto begin = thrust::make_zip_iterator(thrust::make_tuple(a_ptr, b_ptr, c_ptr));
            auto end = thrust::make_zip_iterator(thrust::make_tuple(a_ptr + n, b_ptr + n, c_ptr + n));

            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, begin, end, out_ptr, ClampFunctor());
            });
        }
    }

    // ============= TYPE CONVERSION =============

    struct BoolToFloatFunctor {
        __device__ float operator()(unsigned char x) const { return x ? 1.0f : 0.0f; }
    };

    struct FloatToBoolFunctor {
        __device__ unsigned char operator()(float x) const { return (x != 0.0f) ? 1 : 0; }
    };

    struct FloatToIntFunctor {
        __device__ int operator()(float x) const { return static_cast<int>(x); }
    };

    struct IntToFloatFunctor {
        __device__ float operator()(int x) const { return static_cast<float>(x); }
    };

    void launch_bool_to_float(const unsigned char* src, float* dst, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        auto src_ptr = thrust::device_pointer_cast(src);
        auto dst_ptr = thrust::device_pointer_cast(dst);
        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, src_ptr, src_ptr + n, dst_ptr, BoolToFloatFunctor());
        });
    }

    void launch_float_to_bool(const float* src, unsigned char* dst, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        auto src_ptr = thrust::device_pointer_cast(src);
        auto dst_ptr = thrust::device_pointer_cast(dst);
        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, src_ptr, src_ptr + n, dst_ptr, FloatToBoolFunctor());
        });
    }

    void launch_float_to_int(const float* src, int* dst, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        auto src_ptr = thrust::device_pointer_cast(src);
        auto dst_ptr = thrust::device_pointer_cast(dst);
        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, src_ptr, src_ptr + n, dst_ptr, FloatToIntFunctor());
        });
    }

    void launch_int_to_float(const int* src, float* dst, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        auto src_ptr = thrust::device_pointer_cast(src);
        auto dst_ptr = thrust::device_pointer_cast(dst);
        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, src_ptr, src_ptr + n, dst_ptr, IntToFloatFunctor());
        });
    }

    // ============= LOAD OPERATIONS =============

    void launch_load_op(void* output, const size_t* shape, size_t rank, LoadOp op,
                        const void* args, DataType dtype, cudaStream_t stream) {
        if (dtype != DataType::Float32) return;

        size_t n = 1;
        for (size_t i = 0; i < rank; ++i) n *= shape[i];
        if (n == 0) return;

        if (op == LoadOp::Const && args) {
            float value = *static_cast<const float*>(args);
            auto output_ptr = thrust::device_pointer_cast(static_cast<float*>(output));
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::fill(policy, output_ptr, output_ptr + n, value);
            });
        }
    }

    // ============= OPTIMIZED CUMULATIVE SUM =============

    template<typename T>
    __global__ void cumsum_noncontiguous_kernel(T* data, size_t outer_size, size_t dim_size, size_t inner_size) {
        // Each thread handles one scan line
        size_t scan_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (scan_idx >= outer_size * inner_size) return;

        size_t outer_idx = scan_idx / inner_size;
        size_t inner_idx = scan_idx % inner_size;
        size_t base = outer_idx * dim_size * inner_size + inner_idx;

        T accumulator = data[base];
        for (size_t d = 1; d < dim_size; ++d) {
            size_t idx = base + d * inner_size;
            accumulator = accumulator + data[idx];
            data[idx] = accumulator;
        }
    }

    template<typename T>
    void launch_cumsum_optimized(T* data, size_t outer_size, size_t dim_size,
                                 size_t inner_size, cudaStream_t stream)
    {
        if (outer_size == 0 || dim_size == 0 || inner_size == 0) return;

        if (inner_size == 1 && dim_size > 1) {
            // Contiguous segments - use Thrust's optimized segmented scan
            auto data_ptr = thrust::device_pointer_cast(data);
            size_t total_elements = outer_size * dim_size;
            thrust::device_vector<int> keys(total_elements);

            // Generate keys: 0,0,0,1,1,1,2,2,2,... (each segment gets same key)
            if (stream) {
                thrust::transform(thrust::cuda::par.on(stream),
                    thrust::counting_iterator<size_t>(0),
                    thrust::counting_iterator<size_t>(total_elements),
                    keys.begin(),
                    [=] __device__ (size_t idx) -> int {
                        return static_cast<int>(idx / dim_size);
                    }
                );

                thrust::inclusive_scan_by_key(thrust::cuda::par.on(stream),
                    keys.begin(), keys.end(),
                    data_ptr, data_ptr
                );
            } else {
                thrust::transform(thrust::cuda::par,
                    thrust::counting_iterator<size_t>(0),
                    thrust::counting_iterator<size_t>(total_elements),
                    keys.begin(),
                    [=] __device__ (size_t idx) -> int {
                        return static_cast<int>(idx / dim_size);
                    }
                );

                thrust::inclusive_scan_by_key(thrust::cuda::par,
                    keys.begin(), keys.end(),
                    data_ptr, data_ptr
                );
            }
        } else {
            // Non-contiguous OR single element - use custom kernel
            size_t total_scans = outer_size * inner_size;
            int blocks = (total_scans + 255) / 256;
            cumsum_noncontiguous_kernel<<<blocks, 256, 0, stream>>>(data, outer_size, dim_size, inner_size);
        }
    }

    void launch_cumsum(void* data, const size_t* shape, size_t rank,
                       int dim, DataType dtype, cudaStream_t stream) {
        if (dtype != DataType::Float32 && dtype != DataType::Int32) return;

        size_t total = 1;
        for (size_t i = 0; i < rank; ++i) total *= shape[i];
        if (total == 0) return;

        if (rank == 1) {
            if (dtype == DataType::Float32) {
                auto data_ptr = thrust::device_pointer_cast(static_cast<float*>(data));
                if (stream) {
                    thrust::inclusive_scan(thrust::cuda::par.on(stream), data_ptr, data_ptr + total, data_ptr);
                } else {
                    thrust::inclusive_scan(thrust::cuda::par, data_ptr, data_ptr + total, data_ptr);
                }
            } else if (dtype == DataType::Int32) {
                auto data_ptr = thrust::device_pointer_cast(static_cast<int*>(data));
                if (stream) {
                    thrust::inclusive_scan(thrust::cuda::par.on(stream), data_ptr, data_ptr + total, data_ptr);
                } else {
                    thrust::inclusive_scan(thrust::cuda::par, data_ptr, data_ptr + total, data_ptr);
                }
            }
            return;
        }

        size_t outer_size = 1;
        for (int i = 0; i < dim; ++i) outer_size *= shape[i];
        size_t dim_size = shape[dim];
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < rank; ++i) inner_size *= shape[i];

        if (dtype == DataType::Float32) {
            launch_cumsum_optimized<float>(static_cast<float*>(data), outer_size, dim_size, inner_size, stream);
        } else if (dtype == DataType::Int32) {
            launch_cumsum_optimized<int>(static_cast<int*>(data), outer_size, dim_size, inner_size, stream);
        }
    }

    // ============= HIGHLY OPTIMIZED PAIRWISE DISTANCE =============

// Optimized L2 distance using shared memory and tiling
template<int BLOCK_SIZE = 16>
__global__ void cdist_l2_optimized_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    size_t N, size_t M, size_t D)
{
    __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 to avoid bank conflicts
    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE + 1];

    size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Tile over dimension D
    size_t num_tiles = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (size_t tile = 0; tile < num_tiles; ++tile) {
        // Load tiles into shared memory with coalescing
        size_t d_idx = tile * BLOCK_SIZE + threadIdx.x;
        if (row < N && d_idx < D) {
            tile_a[threadIdx.y][threadIdx.x] = a[row * D + d_idx];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        d_idx = tile * BLOCK_SIZE + threadIdx.y;
        if (col < M && d_idx < D) {
            tile_b[threadIdx.y][threadIdx.x] = b[col * D + d_idx];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial distance for this tile
        size_t d_start = tile * BLOCK_SIZE;
        size_t d_end = (d_start + BLOCK_SIZE < D) ? (d_start + BLOCK_SIZE) : D;
        size_t tile_size = d_end - d_start;

        #pragma unroll
        for (size_t k = 0; k < BLOCK_SIZE; ++k) {
            if (k < tile_size) {
                float diff = tile_a[threadIdx.y][k] - tile_b[k][threadIdx.x];
                sum += diff * diff;
            }
        }

        __syncthreads();
    }

    if (row < N && col < M) {
        out[row * M + col] = sqrtf(sum);
    }
}

// Vectorized L2 for large D using float4
__global__ void cdist_l2_vectorized_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    size_t N, size_t M, size_t D)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= M) return;

    float sum = 0.0f;
    const float4* a_vec = reinterpret_cast<const float4*>(a + i * D);
    const float4* b_vec = reinterpret_cast<const float4*>(b + j * D);

    size_t vec_d = D / 4;

    // Vectorized loop - process 4 elements at a time
    for (size_t d = 0; d < vec_d; ++d) {
        float4 va = a_vec[d];
        float4 vb = b_vec[d];

        float diff_x = va.x - vb.x;
        float diff_y = va.y - vb.y;
        float diff_z = va.z - vb.z;
        float diff_w = va.w - vb.w;

        sum += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w;
    }

    // Handle remaining elements
    for (size_t d = vec_d * 4; d < D; ++d) {
        float diff = a[i * D + d] - b[j * D + d];
        sum += diff * diff;
    }

    out[i * M + j] = sqrtf(sum);
}

// Optimized L1 distance
template<int BLOCK_SIZE = 16>
__global__ void cdist_l1_optimized_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    size_t N, size_t M, size_t D)
{
    __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE + 1];

    size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    size_t num_tiles = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (size_t tile = 0; tile < num_tiles; ++tile) {
        size_t d_idx = tile * BLOCK_SIZE + threadIdx.x;
        if (row < N && d_idx < D) {
            tile_a[threadIdx.y][threadIdx.x] = a[row * D + d_idx];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        d_idx = tile * BLOCK_SIZE + threadIdx.y;
        if (col < M && d_idx < D) {
            tile_b[threadIdx.y][threadIdx.x] = b[col * D + d_idx];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        size_t d_start = tile * BLOCK_SIZE;
        size_t d_end = (d_start + BLOCK_SIZE < D) ? (d_start + BLOCK_SIZE) : D;
        size_t tile_size = d_end - d_start;

        #pragma unroll
        for (size_t k = 0; k < BLOCK_SIZE; ++k) {
            if (k < tile_size) {
                sum += fabsf(tile_a[threadIdx.y][k] - tile_b[k][threadIdx.x]);
            }
        }

        __syncthreads();
    }

    if (row < N && col < M) {
        out[row * M + col] = sum;
    }
}

// General Lp distance kernel (fallback)
__global__ void cdist_lp_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    size_t N, size_t M, size_t D, float p)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= M) return;

    float dist = 0.0f;
    for (size_t d = 0; d < D; ++d) {
        float diff = fabsf(a[i * D + d] - b[j * D + d]);
        dist += powf(diff, p);
    }
    out[i * M + j] = powf(dist, 1.0f / p);
}

void launch_cdist(const float* a, const float* b, float* out,
                  size_t N, size_t M, size_t D, float p, cudaStream_t stream) {
    if (N == 0 || M == 0) return;

    constexpr int BLOCK_SIZE = 16;

    if (p == 2.0f) {
        // L2 distance - choose best kernel based on D
        if (D >= 128 && D % 4 == 0) {
            // Use vectorized kernel for large, aligned dimensions
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
            cdist_l2_vectorized_kernel<<<grid, block, 0, stream>>>(a, b, out, N, M, D);
        } else {
            // Use tiled shared memory kernel
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
            cdist_l2_optimized_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(a, b, out, N, M, D);
        }
    } else if (p == 1.0f) {
        // L1 distance with tiling
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        cdist_l1_optimized_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(a, b, out, N, M, D);
    } else {
        // General Lp distance - fall back to simple kernel
        dim3 block(16, 16);
        dim3 grid((M + 15) / 16, (N + 15) / 16);
        cdist_lp_kernel<<<grid, block, 0, stream>>>(a, b, out, N, M, D, p);
    }
}

    // ============= SORTING =============

    void launch_sort_1d(float* values, int64_t* indices, size_t n, bool descending, cudaStream_t stream) {
        if (n == 0) return;

        auto values_ptr = thrust::device_pointer_cast(values);
        auto indices_ptr = thrust::device_pointer_cast(indices);

        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::sequence(policy, indices_ptr, indices_ptr + n);
        });

        if (stream) {
            if (descending) {
                thrust::sort_by_key(thrust::cuda::par.on(stream), values_ptr, values_ptr + n,
                                   indices_ptr, thrust::greater<float>());
            } else {
                thrust::sort_by_key(thrust::cuda::par.on(stream), values_ptr, values_ptr + n,
                                   indices_ptr, thrust::less<float>());
            }
        } else {
            if (descending) {
                thrust::sort_by_key(thrust::cuda::par, values_ptr, values_ptr + n,
                                   indices_ptr, thrust::greater<float>());
            } else {
                thrust::sort_by_key(thrust::cuda::par, values_ptr, values_ptr + n,
                                   indices_ptr, thrust::less<float>());
            }
        }
    }

    __global__ void extract_slice_kernel(const float* input, float* output,
                                     size_t outer_size, size_t dim_size, size_t inner_size,
                                     size_t outer_idx, size_t inner_idx) {
        size_t d = blockIdx.x * blockDim.x + threadIdx.x;
        if (d < dim_size) {
            size_t src_idx = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
            output[d] = input[src_idx];
        }
    }

    __global__ void write_slice_kernel(float* output, int64_t* output_idx,
                                       const float* sorted_vals, const int64_t* sorted_idx,
                                       size_t outer_size, size_t dim_size, size_t inner_size,
                                       size_t outer_idx, size_t inner_idx) {
        size_t d = blockIdx.x * blockDim.x + threadIdx.x;
        if (d < dim_size) {
            size_t dst_idx = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
            output[dst_idx] = sorted_vals[d];
            output_idx[dst_idx] = sorted_idx[d];
        }
    }

    void launch_sort_2d(float* values, int64_t* indices,
                        size_t outer_size, size_t dim_size, size_t inner_size,
                        int dim, bool descending, cudaStream_t stream) {
        if (dim_size == 0 || outer_size == 0 || inner_size == 0) return;

        thrust::device_vector<float> temp_vals(dim_size);
        thrust::device_vector<int64_t> temp_idx(dim_size);
        int blocks = (dim_size + 255) / 256;

        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                extract_slice_kernel<<<blocks, 256, 0, stream>>>(
                    values, thrust::raw_pointer_cast(temp_vals.data()),
                    outer_size, dim_size, inner_size, outer, inner
                );

                thrust::sequence(thrust::cuda::par.on(stream), temp_idx.begin(), temp_idx.end(), 0LL);

                if (descending) {
                    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        temp_vals.begin(), temp_vals.end(), temp_idx.begin(),
                        thrust::greater<float>());
                } else {
                    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        temp_vals.begin(), temp_vals.end(), temp_idx.begin(),
                        thrust::less<float>());
                }

                write_slice_kernel<<<blocks, 256, 0, stream>>>(
                    values, indices,
                    thrust::raw_pointer_cast(temp_vals.data()),
                    thrust::raw_pointer_cast(temp_idx.data()),
                    outer_size, dim_size, inner_size, outer, inner
                );
            }
        }

        if (stream) cudaStreamSynchronize(stream);
    }

} // namespace gs::tensor_ops