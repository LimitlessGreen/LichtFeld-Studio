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

    // ============= THRUST POLICY HELPER =============

    template<typename Func>
    void run_with_thrust_policy(cudaStream_t stream, Func&& func) {
        if (stream) {
            func(thrust::cuda::par.on(stream));
        } else {
            func(thrust::cuda::par);
        }
    }

    // ============= CLAMP OPERATIONS (UNIFIED) =============

    template<typename T>
    struct ClampScalarFunctor {
        T min_val;
        T max_val;

        ClampScalarFunctor(T min, T max) : min_val(min), max_val(max) {}

        __device__ T operator()(T x) const {
            if constexpr (std::is_floating_point_v<T>) {
                if (isnan(x)) return x;
                return fmax(min_val, fmin(max_val, x));
            } else {
                return max(min_val, min(max_val, x));
            }
        }
    };

    void launch_clamp_scalar(float* data, float min_val, float max_val, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        auto data_ptr = thrust::device_pointer_cast(data);
        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, data_ptr, data_ptr + n, data_ptr,
                            ClampScalarFunctor<float>(min_val, max_val));
        });
    }

    void launch_clamp_scalar_int(int* data, int min_val, int max_val, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        auto data_ptr = thrust::device_pointer_cast(data);
        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, data_ptr, data_ptr + n, data_ptr,
                            ClampScalarFunctor<int>(min_val, max_val));
        });
    }

    // ============= UNARY OPERATIONS =============

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

    // ============= COMPILE-TIME BINARY OPERATION FUNCTORS =============

    template<typename T, BinaryOp Op>
    struct binary_op_functor;

    // Arithmetic operations
    template<typename T> struct binary_op_functor<T, BinaryOp::Add> {
        __device__ T operator()(T a, T b) const { return a + b; }
    };
    template<typename T> struct binary_op_functor<T, BinaryOp::Sub> {
        __device__ T operator()(T a, T b) const { return a - b; }
    };
    template<typename T> struct binary_op_functor<T, BinaryOp::Mul> {
        __device__ T operator()(T a, T b) const { return a * b; }
    };
    template<typename T> struct binary_op_functor<T, BinaryOp::Div> {
        __device__ T operator()(T a, T b) const { return a / b; }
    };
    template<typename T> struct binary_op_functor<T, BinaryOp::Pow> {
        __device__ T operator()(T a, T b) const { return powf(a, b); }
    };
    template<typename T> struct binary_op_functor<T, BinaryOp::Mod> {
        __device__ T operator()(T a, T b) const { return fmodf(a, b); }
    };
    template<typename T> struct binary_op_functor<T, BinaryOp::Maximum> {
        __device__ T operator()(T a, T b) const { return fmaxf(a, b); }
    };
    template<typename T> struct binary_op_functor<T, BinaryOp::Minimum> {
        __device__ T operator()(T a, T b) const { return fminf(a, b); }
    };

    // Comparison operations (return bool)
    template<typename T> struct binary_op_functor<T, BinaryOp::Equal> {
        __device__ unsigned char operator()(T a, T b) const { return (a == b) ? 1 : 0; }
    };
    template<typename T> struct binary_op_functor<T, BinaryOp::NotEqual> {
        __device__ unsigned char operator()(T a, T b) const { return (a != b) ? 1 : 0; }
    };
    template<typename T> struct binary_op_functor<T, BinaryOp::Less> {
        __device__ unsigned char operator()(T a, T b) const { return (a < b) ? 1 : 0; }
    };
    template<typename T> struct binary_op_functor<T, BinaryOp::LessEqual> {
        __device__ unsigned char operator()(T a, T b) const { return (a <= b) ? 1 : 0; }
    };
    template<typename T> struct binary_op_functor<T, BinaryOp::Greater> {
        __device__ unsigned char operator()(T a, T b) const { return (a > b) ? 1 : 0; }
    };
    template<typename T> struct binary_op_functor<T, BinaryOp::GreaterEqual> {
        __device__ unsigned char operator()(T a, T b) const { return (a >= b) ? 1 : 0; }
    };

    // Logical operations (bool only)
    template<> struct binary_op_functor<unsigned char, BinaryOp::LogicalAnd> {
        __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
            return (a && b) ? 1 : 0;
        }
    };
    template<> struct binary_op_functor<unsigned char, BinaryOp::LogicalOr> {
        __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
            return (a || b) ? 1 : 0;
        }
    };
    template<> struct binary_op_functor<unsigned char, BinaryOp::LogicalXor> {
        __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
            return (a != b) ? 1 : 0;
        }
    };
    template<> struct binary_op_functor<unsigned char, BinaryOp::BitwiseOr> {
        __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
            return (a || b) ? 1 : 0;
        }
    };

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
        switch(op) {
            case BinaryOp::Add:
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, a_ptr,
                                binary_op_functor<float, BinaryOp::Add>());
                break;
            case BinaryOp::Sub:
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, a_ptr,
                                binary_op_functor<float, BinaryOp::Sub>());
                break;
            case BinaryOp::Mul:
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, a_ptr,
                                binary_op_functor<float, BinaryOp::Mul>());
                break;
            case BinaryOp::Div:
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, a_ptr,
                                binary_op_functor<float, BinaryOp::Div>());
                break;
            case BinaryOp::Pow:
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, a_ptr,
                                binary_op_functor<float, BinaryOp::Pow>());
                break;
            case BinaryOp::Mod:
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, a_ptr,
                                binary_op_functor<float, BinaryOp::Mod>());
                break;
            case BinaryOp::Maximum:
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, a_ptr,
                                binary_op_functor<float, BinaryOp::Maximum>());
                break;
            case BinaryOp::Minimum:
                thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, a_ptr,
                                binary_op_functor<float, BinaryOp::Minimum>());
                break;
            default:
                break;
        }
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

    // ============= TYPE CONVERSIONS (UNIFIED) =============

    template<typename SrcT, typename DstT>
    struct ConvertFunctor {
        __device__ DstT operator()(SrcT x) const {
            return static_cast<DstT>(x);
        }
    };

    // Specializations for clamping conversions
    template<>
    struct ConvertFunctor<float, uint8_t> {
        __device__ uint8_t operator()(float x) const {
            return static_cast<uint8_t>(fminf(fmaxf(roundf(x), 0.0f), 255.0f));
        }
    };

    template<>
    struct ConvertFunctor<int, uint8_t> {
        __device__ uint8_t operator()(int x) const {
            return static_cast<uint8_t>(max(0, min(255, x)));
        }
    };

    template<>
    struct ConvertFunctor<int64_t, uint8_t> {
        __device__ uint8_t operator()(int64_t x) const {
            return static_cast<uint8_t>(max(static_cast<int64_t>(0),
                                            min(static_cast<int64_t>(255), x)));
        }
    };

    template<typename SrcT, typename DstT>
    void launch_convert_type(const SrcT* src, DstT* dst, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        auto src_ptr = thrust::device_pointer_cast(src);
        auto dst_ptr = thrust::device_pointer_cast(dst);
        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, src_ptr, src_ptr + n, dst_ptr, ConvertFunctor<SrcT, DstT>());
        });
    }

    // ============= BROADCASTING BINARY OPERATIONS =============

    template<typename SrcT, typename DstT, typename Op>
    __global__ void broadcast_binary_kernel(
        const SrcT* __restrict__ a,
        const SrcT* __restrict__ b,
        DstT* __restrict__ c,
        const size_t* __restrict__ a_shape,
        const size_t* __restrict__ b_shape,
        const size_t* __restrict__ c_shape,
        size_t a_rank, size_t b_rank, size_t c_rank,
        size_t total_elements,
        Op op)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements) return;

        // Compute c coordinates
        size_t c_coords[10];
        size_t temp = idx;
        for (int i = c_rank - 1; i >= 0; --i) {
            c_coords[i] = temp % c_shape[i];
            temp /= c_shape[i];
        }

        // Map to a coordinates
        size_t a_idx = 0;
        size_t a_stride = 1;
        int a_offset = static_cast<int>(c_rank) - static_cast<int>(a_rank);
        for (int i = a_rank - 1; i >= 0; --i) {
            int c_dim_idx = i + a_offset;
            size_t coord = (c_dim_idx >= 0 && c_dim_idx < static_cast<int>(c_rank)) ? c_coords[c_dim_idx] : 0;
            if (a_shape[i] == 1) {
                coord = 0;
            }
            a_idx += coord * a_stride;
            a_stride *= a_shape[i];
        }

        // Map to b coordinates
        size_t b_idx = 0;
        size_t b_stride = 1;
        int b_offset = static_cast<int>(c_rank) - static_cast<int>(b_rank);
        for (int i = b_rank - 1; i >= 0; --i) {
            int c_dim_idx = i + b_offset;
            size_t coord = (c_dim_idx >= 0 && c_dim_idx < static_cast<int>(c_rank)) ? c_coords[c_dim_idx] : 0;
            if (b_shape[i] == 1) {
                coord = 0;
            }
            b_idx += coord * b_stride;
            b_stride *= b_shape[i];
        }

        // Apply operation
        c[idx] = op(a[a_idx], b[b_idx]);
    }

    template<typename SrcT, typename DstT, typename Op>
    void launch_broadcast_binary(
        const SrcT* a, const SrcT* b, DstT* c,
        const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
        size_t a_rank, size_t b_rank, size_t c_rank,
        size_t total_elements,
        Op op,
        cudaStream_t stream)
    {
        if (total_elements == 0) return;

        thrust::device_vector<size_t> d_a_shape(a_shape, a_shape + a_rank);
        thrust::device_vector<size_t> d_b_shape(b_shape, b_shape + b_rank);
        thrust::device_vector<size_t> d_c_shape(c_shape, c_shape + c_rank);

        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        broadcast_binary_kernel<<<grid_size, block_size, 0, stream>>>(
            a, b, c,
            thrust::raw_pointer_cast(d_a_shape.data()),
            thrust::raw_pointer_cast(d_b_shape.data()),
            thrust::raw_pointer_cast(d_c_shape.data()),
            a_rank, b_rank, c_rank,
            total_elements,
            op
        );
    }

    // Runtime dispatcher with compile-time operation
    template<typename SrcT, typename DstT>
    void launch_binary_operation(
        const void* a, const void* b, void* c,
        const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
        size_t a_rank, size_t b_rank, size_t c_rank,
        size_t c_elements, BinaryOp op,
        cudaStream_t stream)
    {
        if (c_elements == 0) return;

        const SrcT* a_ptr = static_cast<const SrcT*>(a);
        const SrcT* b_ptr = static_cast<const SrcT*>(b);
        DstT* c_ptr = static_cast<DstT*>(c);

        // Dispatch based on operation
        switch(op) {
            #define DISPATCH_OP(OP_ENUM) \
            case BinaryOp::OP_ENUM: \
                launch_broadcast_binary(a_ptr, b_ptr, c_ptr, a_shape, b_shape, c_shape, \
                                       a_rank, b_rank, c_rank, c_elements, \
                                       binary_op_functor<SrcT, BinaryOp::OP_ENUM>(), stream); \
                break;

            DISPATCH_OP(Add)
            DISPATCH_OP(Sub)
            DISPATCH_OP(Mul)
            DISPATCH_OP(Div)
            DISPATCH_OP(Pow)
            DISPATCH_OP(Maximum)
            DISPATCH_OP(Minimum)
            DISPATCH_OP(Equal)
            DISPATCH_OP(NotEqual)
            DISPATCH_OP(Less)
            DISPATCH_OP(LessEqual)
            DISPATCH_OP(Greater)
            DISPATCH_OP(GreaterEqual)

            #undef DISPATCH_OP

            case BinaryOp::LogicalAnd:
            case BinaryOp::LogicalOr:
            case BinaryOp::LogicalXor:
            case BinaryOp::BitwiseOr:
                if constexpr (std::is_same_v<SrcT, unsigned char>) {
                    switch(op) {
                        case BinaryOp::LogicalAnd:
                            launch_broadcast_binary(a_ptr, b_ptr, c_ptr, a_shape, b_shape, c_shape,
                                                   a_rank, b_rank, c_rank, c_elements,
                                                   binary_op_functor<unsigned char, BinaryOp::LogicalAnd>(), stream);
                            break;
                        case BinaryOp::LogicalOr:
                        case BinaryOp::BitwiseOr:
                            launch_broadcast_binary(a_ptr, b_ptr, c_ptr, a_shape, b_shape, c_shape,
                                                   a_rank, b_rank, c_rank, c_elements,
                                                   binary_op_functor<unsigned char, BinaryOp::LogicalOr>(), stream);
                            break;
                        case BinaryOp::LogicalXor:
                            launch_broadcast_binary(a_ptr, b_ptr, c_ptr, a_shape, b_shape, c_shape,
                                                   a_rank, b_rank, c_rank, c_elements,
                                                   binary_op_functor<unsigned char, BinaryOp::LogicalXor>(), stream);
                            break;
                        default:
                            break;
                    }
                }
                break;

            default:
                break;
        }
    }

    // ============= ELEMENT-WISE BINARY OPERATIONS (LEGACY WRAPPER) =============

    void launch_binary_op(const void* a, const void* b, void* c, size_t n, BinaryOp op,
                          DataType a_dtype, DataType b_dtype, DataType c_dtype,
                          cudaStream_t stream) {
        if (n == 0) return;

        // Element-wise operation - all tensors have the same 1D shape [n]
        size_t shape[1] = {n};

        // Dispatch based on dtype combinations
        if (c_dtype == DataType::Bool && a_dtype == DataType::Int32 && b_dtype == DataType::Int32) {
            launch_binary_operation<int, unsigned char>(
                a, b, c, shape, shape, shape, 1, 1, 1, n, op, stream);
        }
        else if (c_dtype == DataType::Bool && a_dtype == DataType::Float32 && b_dtype == DataType::Float32) {
            launch_binary_operation<float, unsigned char>(
                a, b, c, shape, shape, shape, 1, 1, 1, n, op, stream);
        }
        else if (a_dtype == DataType::Bool && b_dtype == DataType::Bool && c_dtype == DataType::Bool) {
            launch_binary_operation<unsigned char, unsigned char>(
                a, b, c, shape, shape, shape, 1, 1, 1, n, op, stream);
        }
        else if (a_dtype == DataType::Float32 && b_dtype == DataType::Float32 && c_dtype == DataType::Float32) {
            launch_binary_operation<float, float>(
                a, b, c, shape, shape, shape, 1, 1, 1, n, op, stream);
        }
        else if (a_dtype == DataType::Int32 && b_dtype == DataType::Int32 && c_dtype == DataType::Int32) {
            launch_binary_operation<int, int>(
                a, b, c, shape, shape, shape, 1, 1, 1, n, op, stream);
        }
    }

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

    // ============= MULTI-AXIS REDUCTION =============

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
                    result = thrust::reduce(policy, input_ptr, input_ptr + n, 0.0f, thrust::plus<float>());
                    break;
                case ReduceOp::Mean:
                    result = thrust::reduce(policy, input_ptr, input_ptr + n, 0.0f, thrust::plus<float>()) / static_cast<float>(n);
                    break;
                case ReduceOp::Max:
                    result = thrust::reduce(policy, input_ptr, input_ptr + n, -std::numeric_limits<float>::infinity(), thrust::maximum<float>());
                    break;
                case ReduceOp::Min:
                    result = thrust::reduce(policy, input_ptr, input_ptr + n, std::numeric_limits<float>::infinity(), thrust::minimum<float>());
                    break;
                case ReduceOp::Prod:
                    result = thrust::reduce(policy, input_ptr, input_ptr + n, 1.0f, thrust::multiplies<float>());
                    break;
                default:
                    result = 0.0f;
                }
            });
            cudaMemcpyAsync(output, &result, sizeof(float), cudaMemcpyHostToDevice, stream);
            return;
        }

        // Single-axis reduction
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
            // Non-contiguous - use custom kernel
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

    // ============= OPTIMIZED PAIRWISE DISTANCE =============

    template<int BLOCK_SIZE = 16>
    __global__ void cdist_l2_optimized_kernel(
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

        for (size_t d = 0; d < vec_d; ++d) {
            float4 va = a_vec[d];
            float4 vb = b_vec[d];

            float diff_x = va.x - vb.x;
            float diff_y = va.y - vb.y;
            float diff_z = va.z - vb.z;
            float diff_w = va.w - vb.w;

            sum += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w;
        }

        for (size_t d = vec_d * 4; d < D; ++d) {
            float diff = a[i * D + d] - b[j * D + d];
            sum += diff * diff;
        }

        out[i * M + j] = sqrtf(sum);
    }

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
            if (D >= 128 && D % 4 == 0) {
                dim3 block(BLOCK_SIZE, BLOCK_SIZE);
                dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
                cdist_l2_vectorized_kernel<<<grid, block, 0, stream>>>(a, b, out, N, M, D);
            } else {
                dim3 block(BLOCK_SIZE, BLOCK_SIZE);
                dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
                cdist_l2_optimized_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(a, b, out, N, M, D);
            }
        } else if (p == 1.0f) {
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
            cdist_l1_optimized_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(a, b, out, N, M, D);
        } else {
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

    // ============= CONCATENATION OPERATIONS =============

    template<typename T>
    __global__ void cat_last_dim_kernel(
        T* output,
        const T** input_ptrs,
        const size_t* input_sizes,
        size_t num_tensors,
        size_t num_rows,
        size_t row_size)
    {
        size_t row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= num_rows) return;

        size_t result_offset = 0;
        for (size_t t = 0; t < num_tensors; ++t) {
            size_t tensor_dim_size = input_sizes[t];

            const T* src = input_ptrs[t] + row * tensor_dim_size;
            T* dst = output + row * row_size + result_offset;

            for (size_t i = 0; i < tensor_dim_size; ++i) {
                dst[i] = src[i];
            }

            result_offset += tensor_dim_size;
        }
    }

    void launch_cat_last_dim(
        void* output,
        const std::vector<Tensor>& tensors,
        size_t num_rows,
        size_t row_size,
        size_t element_size,
        cudaStream_t stream)
    {
        size_t num_tensors = tensors.size();

        thrust::device_vector<const float*> d_input_ptrs(num_tensors);
        thrust::device_vector<size_t> d_input_sizes(num_tensors);

        for (size_t i = 0; i < num_tensors; ++i) {
            d_input_ptrs[i] = static_cast<const float*>(tensors[i].raw_ptr());
            d_input_sizes[i] = tensors[i].shape()[tensors[i].shape().rank() - 1];
        }

        int block_size = 256;
        int grid_size = (num_rows + block_size - 1) / block_size;

        cat_last_dim_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<float*>(output),
            thrust::raw_pointer_cast(d_input_ptrs.data()),
            thrust::raw_pointer_cast(d_input_sizes.data()),
            num_tensors,
            num_rows,
            row_size);
    }

    template<typename T>
    __global__ void cat_middle_dim_kernel(
        T* output,
        const T** input_ptrs,
        const size_t* input_sizes,
        size_t num_tensors,
        size_t outer_size,
        size_t inner_size,
        size_t total_dim_size)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total = outer_size * total_dim_size * inner_size;

        if (idx >= total) return;

        size_t outer_idx = idx / (total_dim_size * inner_size);
        size_t remainder = idx % (total_dim_size * inner_size);
        size_t dim_idx = remainder / inner_size;
        size_t inner_idx = remainder % inner_size;

        size_t accumulated = 0;
        for (size_t t = 0; t < num_tensors; ++t) {
            if (dim_idx < accumulated + input_sizes[t]) {
                size_t tensor_dim_idx = dim_idx - accumulated;
                size_t src_idx = outer_idx * input_sizes[t] * inner_size +
                               tensor_dim_idx * inner_size + inner_idx;
                output[idx] = input_ptrs[t][src_idx];
                return;
            }
            accumulated += input_sizes[t];
        }
    }

    void launch_cat_middle_dim(
        void* output,
        const std::vector<Tensor>& tensors,
        size_t outer_size,
        size_t inner_size,
        int resolved_dim,
        size_t element_size,
        cudaStream_t stream)
    {
        size_t num_tensors = tensors.size();
        size_t total_dim_size = 0;
        for (const auto& t : tensors) {
            total_dim_size += t.shape()[resolved_dim];
        }

        size_t total_elements = outer_size * total_dim_size * inner_size;

        thrust::device_vector<const float*> d_input_ptrs(num_tensors);
        thrust::device_vector<size_t> d_input_sizes(num_tensors);

        for (size_t i = 0; i < num_tensors; ++i) {
            d_input_ptrs[i] = static_cast<const float*>(tensors[i].raw_ptr());
            d_input_sizes[i] = tensors[i].shape()[resolved_dim];
        }

        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        cat_middle_dim_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<float*>(output),
            thrust::raw_pointer_cast(d_input_ptrs.data()),
            thrust::raw_pointer_cast(d_input_sizes.data()),
            num_tensors,
            outer_size,
            inner_size,
            total_dim_size);
    }

    // ============= EXPLICIT TEMPLATE INSTANTIATIONS =============

    // Type conversions
    template void launch_convert_type<float, uint8_t>(const float*, uint8_t*, size_t, cudaStream_t);
    template void launch_convert_type<uint8_t, float>(const uint8_t*, float*, size_t, cudaStream_t);
    template void launch_convert_type<int, uint8_t>(const int*, uint8_t*, size_t, cudaStream_t);
    template void launch_convert_type<uint8_t, int>(const uint8_t*, int*, size_t, cudaStream_t);
    template void launch_convert_type<int64_t, float>(const int64_t*, float*, size_t, cudaStream_t);
    template void launch_convert_type<float, int64_t>(const float*, int64_t*, size_t, cudaStream_t);
    template void launch_convert_type<int, int64_t>(const int*, int64_t*, size_t, cudaStream_t);
    template void launch_convert_type<int64_t, int>(const int64_t*, int*, size_t, cudaStream_t);
    template void launch_convert_type<uint8_t, int64_t>(const uint8_t*, int64_t*, size_t, cudaStream_t);
    template void launch_convert_type<int64_t, uint8_t>(const int64_t*, uint8_t*, size_t, cudaStream_t);
    template void launch_convert_type<int, float>(const int*, float*, size_t, cudaStream_t);
    template void launch_convert_type<float, int>(const float*, int*, size_t, cudaStream_t);

    // Binary operations
    template void launch_binary_operation<float, float>(
        const void*, const void*, void*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, BinaryOp, cudaStream_t);
    template void launch_binary_operation<float, unsigned char>(
        const void*, const void*, void*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, BinaryOp, cudaStream_t);
    template void launch_binary_operation<int, int>(
        const void*, const void*, void*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, BinaryOp, cudaStream_t);
    template void launch_binary_operation<int, unsigned char>(
        const void*, const void*, void*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, BinaryOp, cudaStream_t);
    template void launch_binary_operation<unsigned char, unsigned char>(
        const void*, const void*, void*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, BinaryOp, cudaStream_t);

} // namespace gs::tensor_ops