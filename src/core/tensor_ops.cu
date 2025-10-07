/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>
#include <limits>

// Thrust headers
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace gs::tensor_ops {

    struct ClampScalarFunctor {
        float min_val;
        float max_val;

        ClampScalarFunctor(float min, float max) : min_val(min), max_val(max) {}

        __device__ float operator()(float x) const {
            return fmax(min_val, fmin(max_val, x));
        }
    };

    void launch_clamp_scalar(float* data, float min_val, float max_val, size_t n, cudaStream_t stream) {
        if (n == 0) return;

        auto data_ptr = thrust::device_pointer_cast(data);
        thrust::transform(
            thrust::cuda::par.on(stream),
            data_ptr, data_ptr + n,
            data_ptr,
            ClampScalarFunctor(min_val, max_val)
        );
    }
// ============= UNARY OPERATION FUNCTORS =============

template<typename T>
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

        thrust::transform(
            thrust::cuda::par.on(stream),
            in_ptr, in_ptr + n,
            out_ptr,
            UnaryOpFunctor<float>(op)
        );
    } else if (dtype == DataType::Bool && op == UnaryOp::LogicalNot) {
        auto in_ptr = thrust::device_pointer_cast(static_cast<const unsigned char*>(input));
        auto out_ptr = thrust::device_pointer_cast(static_cast<unsigned char*>(output));

        thrust::transform(
            thrust::cuda::par.on(stream),
            in_ptr, in_ptr + n,
            out_ptr,
            LogicalNotFunctor()
        );
    }
}

void launch_unary_op_inplace(void* data, size_t n, UnaryOp op, DataType dtype, cudaStream_t stream) {
    launch_unary_op(data, data, n, op, dtype, stream);
}

// ============= BINARY OPERATION FUNCTORS =============

template<typename T>
struct BinaryOpFunctor {
    BinaryOp op;

    BinaryOpFunctor(BinaryOp op_) : op(op_) {}

    __device__ T operator()(T a, T b) const {
        switch (op) {
            case BinaryOp::Add: return a + b;
            case BinaryOp::Sub: return a - b;
            case BinaryOp::Mul: return a * b;
            case BinaryOp::Div: return a / (b + T(1e-8));
            case BinaryOp::Pow: return pow(a, b);
            case BinaryOp::Mod: return fmod(a, b);
            case BinaryOp::Maximum: return fmax(a, b);
            case BinaryOp::Minimum: return fmin(a, b);
            default: return T(0);
        }
    }
};

template<typename T>
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

    if (c_dtype == DataType::Bool && a_dtype == DataType::Float32) {
        // Comparison operations
        auto a_ptr = thrust::device_pointer_cast(static_cast<const float*>(a));
        auto b_ptr = thrust::device_pointer_cast(static_cast<const float*>(b));
        auto c_ptr = thrust::device_pointer_cast(static_cast<unsigned char*>(c));

        thrust::transform(
            thrust::cuda::par.on(stream),
            a_ptr, a_ptr + n,
            b_ptr,
            c_ptr,
            ComparisonOpFunctor<float>(op)
        );
    } else if (a_dtype == DataType::Bool && b_dtype == DataType::Bool) {
        // Logical operations
        auto a_ptr = thrust::device_pointer_cast(static_cast<const unsigned char*>(a));
        auto b_ptr = thrust::device_pointer_cast(static_cast<const unsigned char*>(b));
        auto c_ptr = thrust::device_pointer_cast(static_cast<unsigned char*>(c));

        thrust::transform(
            thrust::cuda::par.on(stream),
            a_ptr, a_ptr + n,
            b_ptr,
            c_ptr,
            LogicalOpFunctor(op)
        );
    } else if (a_dtype == DataType::Float32 && b_dtype == DataType::Float32) {
        // Arithmetic operations
        auto a_ptr = thrust::device_pointer_cast(static_cast<const float*>(a));
        auto b_ptr = thrust::device_pointer_cast(static_cast<const float*>(b));
        auto c_ptr = thrust::device_pointer_cast(static_cast<float*>(c));

        thrust::transform(
            thrust::cuda::par.on(stream),
            a_ptr, a_ptr + n,
            b_ptr,
            c_ptr,
            BinaryOpFunctor<float>(op)
        );
    }
}

template<typename T>
struct BinaryScalarOpFunctor {
    T scalar;
    BinaryOp op;

    BinaryScalarOpFunctor(T scalar_, BinaryOp op_) : scalar(scalar_), op(op_) {}

    __device__ T operator()(T a) const {
        switch (op) {
            case BinaryOp::Add: return a + scalar;
            case BinaryOp::Sub: return a - scalar;
            case BinaryOp::Mul: return a * scalar;
            case BinaryOp::Div: return a / (scalar + T(1e-8));
            case BinaryOp::Pow: return pow(a, scalar);
            case BinaryOp::Mod: return fmod(a, scalar);
            case BinaryOp::Maximum: return fmax(a, scalar);
            case BinaryOp::Minimum: return fmin(a, scalar);
            default: return T(0);
        }
    }
};

template<typename T>
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

    if (dst_dtype == DataType::Bool && src_dtype == DataType::Float32) {
        // Comparison with scalar
        auto data_ptr = thrust::device_pointer_cast(static_cast<const float*>(data));
        auto result_ptr = thrust::device_pointer_cast(static_cast<unsigned char*>(result));

        thrust::transform(
            thrust::cuda::par.on(stream),
            data_ptr, data_ptr + n,
            result_ptr,
            ComparisonScalarOpFunctor<float>(scalar, op)
        );
    } else if (src_dtype == DataType::Float32 && dst_dtype == DataType::Float32) {
        // Arithmetic with scalar
        auto data_ptr = thrust::device_pointer_cast(static_cast<const float*>(data));
        auto result_ptr = thrust::device_pointer_cast(static_cast<float*>(result));

        thrust::transform(
            thrust::cuda::par.on(stream),
            data_ptr, data_ptr + n,
            result_ptr,
            BinaryScalarOpFunctor<float>(scalar, op)
        );
    }
}

void launch_binary_op_inplace(void* a, const void* b, size_t n, BinaryOp op, cudaStream_t stream) {
    if (n == 0) return;

    auto a_ptr = thrust::device_pointer_cast(static_cast<float*>(a));
    auto b_ptr = thrust::device_pointer_cast(static_cast<const float*>(b));

    thrust::transform(
        thrust::cuda::par.on(stream),
        a_ptr, a_ptr + n,
        b_ptr,
        a_ptr,
        BinaryOpFunctor<float>(op)
    );
}

void launch_binary_scalar_inplace(void* data, float scalar, size_t n, BinaryOp op, cudaStream_t stream) {
    if (n == 0) return;

    auto data_ptr = thrust::device_pointer_cast(static_cast<float*>(data));

    thrust::transform(
        thrust::cuda::par.on(stream),
        data_ptr, data_ptr + n,
        data_ptr,
        BinaryScalarOpFunctor<float>(scalar, op)
    );
}

// ============= REDUCE OPERATION FUNCTORS =============

struct ReduceSumFunctor {
    __device__ float operator()(float a, float b) const {
        return a + b;
    }
};

struct ReduceMaxFunctor {
    __device__ float operator()(float a, float b) const {
        return fmax(a, b);
    }
};

struct ReduceMinFunctor {
    __device__ float operator()(float a, float b) const {
        return fmin(a, b);
    }
};

struct ReduceProdFunctor {
    __device__ float operator()(float a, float b) const {
        return a * b;
    }
};

void launch_reduce_op(const void* input, void* output, const size_t* shape, size_t rank,
                      const int* axes, size_t num_axes, bool keepdim, ReduceOp op,
                      DataType dtype, cudaStream_t stream) {
    size_t n = 1;
    for (size_t i = 0; i < rank; ++i) {
        n *= shape[i];
    }

    if (n == 0 || dtype != DataType::Float32) return;

    auto input_ptr = thrust::device_pointer_cast(static_cast<const float*>(input));
    auto output_ptr = thrust::device_pointer_cast(static_cast<float*>(output));

    float result = 0.0f;

    switch (op) {
        case ReduceOp::Sum:
            result = thrust::reduce(
                thrust::cuda::par.on(stream),
                input_ptr, input_ptr + n,
                0.0f,
                ReduceSumFunctor()
            );
            break;

        case ReduceOp::Mean:
            result = thrust::reduce(
                thrust::cuda::par.on(stream),
                input_ptr, input_ptr + n,
                0.0f,
                ReduceSumFunctor()
            ) / static_cast<float>(n);
            break;

        case ReduceOp::Max:
            result = thrust::reduce(
                thrust::cuda::par.on(stream),
                input_ptr, input_ptr + n,
                std::numeric_limits<float>::lowest(),
                ReduceMaxFunctor()
            );
            break;

        case ReduceOp::Min:
            result = thrust::reduce(
                thrust::cuda::par.on(stream),
                input_ptr, input_ptr + n,
                std::numeric_limits<float>::max(),
                ReduceMinFunctor()
            );
            break;

        case ReduceOp::Prod:
            result = thrust::reduce(
                thrust::cuda::par.on(stream),
                input_ptr, input_ptr + n,
                1.0f,
                ReduceProdFunctor()
            );
            break;

        default:
            result = 0.0f;
            break;
    }

    // Write result back
    cudaMemcpyAsync(output, &result, sizeof(float), cudaMemcpyHostToDevice, stream);
}

// ============= TERNARY OPERATION FUNCTORS =============

struct WhereFunctor {
    __device__ float operator()(const thrust::tuple<unsigned char, float, float>& t) const {
        return thrust::get<0>(t) ? thrust::get<1>(t) : thrust::get<2>(t);
    }
};

struct ClampFunctor {
    __device__ float operator()(const thrust::tuple<float, float, float>& t) const {
        float val = thrust::get<0>(t);
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

        thrust::transform(
            thrust::cuda::par.on(stream),
            begin, end,
            out_ptr,
            ClampFunctor()
        );
    }
}

// ============= TYPE CONVERSION FUNCTORS =============

struct BoolToFloatFunctor {
    __device__ float operator()(unsigned char x) const {
        return x ? 1.0f : 0.0f;
    }
};

struct FloatToBoolFunctor {
    __device__ unsigned char operator()(float x) const {
        return (x != 0.0f) ? 1 : 0;
    }
};

struct FloatToIntFunctor {
    __device__ int operator()(float x) const {
        return static_cast<int>(x);
    }
};

struct IntToFloatFunctor {
    __device__ float operator()(int x) const {
        return static_cast<float>(x);
    }
};

void launch_bool_to_float(const unsigned char* src, float* dst, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    auto src_ptr = thrust::device_pointer_cast(src);
    auto dst_ptr = thrust::device_pointer_cast(dst);

    thrust::transform(
        thrust::cuda::par.on(stream),
        src_ptr, src_ptr + n,
        dst_ptr,
        BoolToFloatFunctor()
    );
}

void launch_float_to_bool(const float* src, unsigned char* dst, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    auto src_ptr = thrust::device_pointer_cast(src);
    auto dst_ptr = thrust::device_pointer_cast(dst);

    thrust::transform(
        thrust::cuda::par.on(stream),
        src_ptr, src_ptr + n,
        dst_ptr,
        FloatToBoolFunctor()
    );
}

void launch_float_to_int(const float* src, int* dst, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    auto src_ptr = thrust::device_pointer_cast(src);
    auto dst_ptr = thrust::device_pointer_cast(dst);

    thrust::transform(
        thrust::cuda::par.on(stream),
        src_ptr, src_ptr + n,
        dst_ptr,
        FloatToIntFunctor()
    );
}

void launch_int_to_float(const int* src, float* dst, size_t n, cudaStream_t stream) {
    if (n == 0) return;

    auto src_ptr = thrust::device_pointer_cast(src);
    auto dst_ptr = thrust::device_pointer_cast(dst);

    thrust::transform(
        thrust::cuda::par.on(stream),
        src_ptr, src_ptr + n,
        dst_ptr,
        IntToFloatFunctor()
    );
}

// ============= LOAD OPERATION IMPLEMENTATIONS =============

void launch_load_op(void* output, const size_t* shape, size_t rank, LoadOp op,
                    const void* args, DataType dtype, cudaStream_t stream) {
    if (dtype != DataType::Float32) return;

    size_t n = 1;
    for (size_t i = 0; i < rank; ++i) {
        n *= shape[i];
    }

    if (n == 0) return;

    if (op == LoadOp::Const && args) {
        float value = *static_cast<const float*>(args);
        auto output_ptr = thrust::device_pointer_cast(static_cast<float*>(output));

        thrust::fill(
            thrust::cuda::par.on(stream),
            output_ptr, output_ptr + n,
            value
        );
    }
}

// ============= CUMULATIVE SUM KERNEL =============

template<typename T>
__global__ void cumsum_strided_kernel(T* data, size_t stride, size_t outer_size, size_t dim_size) {
    size_t o = blockIdx.x;
    size_t s = threadIdx.x;

    if (o >= outer_size || s >= stride) return;

    size_t base = o * dim_size * stride + s;

    for (size_t d = 1; d < dim_size; ++d) {
        data[base + d * stride] += data[base + (d-1) * stride];
    }
}

void launch_cumsum(void* data, const size_t* shape, size_t rank,
                  int dim, DataType dtype, cudaStream_t stream) {
    if (dtype != DataType::Float32 && dtype != DataType::Int32) {
        return;
    }

    size_t stride = 1;
    for (size_t i = dim + 1; i < rank; ++i) {
        stride *= shape[i];
    }

    size_t outer_size = 1;
    for (size_t i = 0; i < static_cast<size_t>(dim); ++i) {
        outer_size *= shape[i];
    }

    size_t dim_size = shape[dim];

    // For simple 1D case or when stride == 1, use Thrust directly
    if (rank == 1 || (dim == static_cast<int>(rank) - 1)) {
        size_t total = outer_size * dim_size;

        if (dtype == DataType::Float32) {
            auto data_ptr = thrust::device_pointer_cast(static_cast<float*>(data));
            thrust::inclusive_scan(
                thrust::cuda::par.on(stream),
                data_ptr, data_ptr + total,
                data_ptr
            );
        } else if (dtype == DataType::Int32) {
            auto data_ptr = thrust::device_pointer_cast(static_cast<int*>(data));
            thrust::inclusive_scan(
                thrust::cuda::par.on(stream),
                data_ptr, data_ptr + total,
                data_ptr
            );
        }
        return;
    }

    // For complex multi-dimensional case, use custom kernel
    dim3 grid(outer_size);
    dim3 block(min(stride, size_t(256)));

    if (dtype == DataType::Float32) {
        cumsum_strided_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<float*>(data), stride, outer_size, dim_size);
    } else if (dtype == DataType::Int32) {
        cumsum_strided_kernel<int><<<grid, block, 0, stream>>>(
            static_cast<int*>(data), stride, outer_size, dim_size);
    }
}

// ============= MOVEMENT OPERATION STUB =============

void launch_movement_op(const void* input, void* output,
                       const size_t* input_shape, const size_t* output_shape,
                       size_t input_rank, size_t output_rank,
                       MovementOp op, const void* args,
                       DataType dtype, cudaStream_t stream) {
    // Movement operations are typically handled in CPU code
    // with shape manipulation - no specific kernel needed
}

// ============= LEGACY OPERATIONS =============
// These are kept for backward compatibility but use Thrust internally

struct AbsFunctor {
    __device__ float operator()(float x) const {
        return fabsf(x);
    }
};

struct SqrtFunctor {
    __device__ float operator()(float x) const {
        return sqrtf(fmaxf(x, 0.0f));
    }
};

struct ExpFunctor {
    __device__ float operator()(float x) const {
        return expf(x);
    }
};

struct LogFunctor {
    __device__ float operator()(float x) const {
        return logf(fmaxf(x, 1e-10f));
    }
};

struct SigmoidFunctor {
    __device__ float operator()(float x) const {
        return 1.0f / (1.0f + expf(-x));
    }
};

struct ReluFunctor {
    __device__ float operator()(float x) const {
        return fmaxf(x, 0.0f);
    }
};

struct ClampLegacyFunctor {
    float min_val, max_val;

    ClampLegacyFunctor(float min_, float max_) : min_val(min_), max_val(max_) {}

    __device__ float operator()(float x) const {
        return fmaxf(min_val, fminf(max_val, x));
    }
};

struct LogitFunctor {
    float eps;

    LogitFunctor(float eps_) : eps(eps_) {}

    __device__ float operator()(float x) const {
        float clamped = fmaxf(eps, fminf(1.0f - eps, x));
        return logf(clamped / (1.0f - clamped));
    }
};

struct PowScalarFunctor {
    float exponent;

    PowScalarFunctor(float exp_) : exponent(exp_) {}

    __device__ float operator()(float x) const {
        return powf(x, exponent);
    }
};

void launch_abs(float* data, size_t n, cudaStream_t stream) {
    auto ptr = thrust::device_pointer_cast(data);
    thrust::transform(
        thrust::cuda::par.on(stream),
        ptr, ptr + n,
        ptr,
        AbsFunctor()
    );
}

void launch_sqrt(float* data, size_t n, cudaStream_t stream) {
    auto ptr = thrust::device_pointer_cast(data);
    thrust::transform(
        thrust::cuda::par.on(stream),
        ptr, ptr + n,
        ptr,
        SqrtFunctor()
    );
}

void launch_exp(float* data, size_t n, cudaStream_t stream) {
    auto ptr = thrust::device_pointer_cast(data);
    thrust::transform(
        thrust::cuda::par.on(stream),
        ptr, ptr + n,
        ptr,
        ExpFunctor()
    );
}

void launch_log(float* data, size_t n, cudaStream_t stream) {
    auto ptr = thrust::device_pointer_cast(data);
    thrust::transform(
        thrust::cuda::par.on(stream),
        ptr, ptr + n,
        ptr,
        LogFunctor()
    );
}

void launch_sigmoid(float* data, size_t n, cudaStream_t stream) {
    auto ptr = thrust::device_pointer_cast(data);
    thrust::transform(
        thrust::cuda::par.on(stream),
        ptr, ptr + n,
        ptr,
        SigmoidFunctor()
    );
}

void launch_relu(float* data, size_t n, cudaStream_t stream) {
    auto ptr = thrust::device_pointer_cast(data);
    thrust::transform(
        thrust::cuda::par.on(stream),
        ptr, ptr + n,
        ptr,
        ReluFunctor()
    );
}

void launch_clamp(float* data, float min_val, float max_val, size_t n, cudaStream_t stream) {
    auto ptr = thrust::device_pointer_cast(data);
    thrust::transform(
        thrust::cuda::par.on(stream),
        ptr, ptr + n,
        ptr,
        ClampLegacyFunctor(min_val, max_val)
    );
}

void launch_logit(const float* input, float* output, size_t n, float eps, cudaStream_t stream) {
    auto in_ptr = thrust::device_pointer_cast(input);
    auto out_ptr = thrust::device_pointer_cast(output);

    thrust::transform(
        thrust::cuda::par.on(stream),
        in_ptr, in_ptr + n,
        out_ptr,
        LogitFunctor(eps)
    );
}

void launch_pow_scalar(float* data, float exponent, size_t n, cudaStream_t stream) {
    auto ptr = thrust::device_pointer_cast(data);
    thrust::transform(
        thrust::cuda::par.on(stream),
        ptr, ptr + n,
        ptr,
        PowScalarFunctor(exponent)
    );
}

void launch_reduce_sum(const float* data, float* result, size_t n, cudaStream_t stream) {
    auto ptr = thrust::device_pointer_cast(data);
    float sum = thrust::reduce(
        thrust::cuda::par.on(stream),
        ptr, ptr + n,
        0.0f,
        thrust::plus<float>()
    );
    cudaMemcpyAsync(result, &sum, sizeof(float), cudaMemcpyHostToDevice, stream);
}

void launch_reduce_mean(const float* data, float* result, size_t n, cudaStream_t stream) {
    auto ptr = thrust::device_pointer_cast(data);
    float sum = thrust::reduce(
        thrust::cuda::par.on(stream),
        ptr, ptr + n,
        0.0f,
        thrust::plus<float>()
    );
    float mean = sum / static_cast<float>(n);
    cudaMemcpyAsync(result, &mean, sizeof(float), cudaMemcpyHostToDevice, stream);
}

void launch_reduce_min(const float* data, float* result, size_t n, cudaStream_t stream) {
    auto ptr = thrust::device_pointer_cast(data);
    float min_val = thrust::reduce(
        thrust::cuda::par.on(stream),
        ptr, ptr + n,
        std::numeric_limits<float>::max(),
        thrust::minimum<float>()
    );
    cudaMemcpyAsync(result, &min_val, sizeof(float), cudaMemcpyHostToDevice, stream);
}

void launch_reduce_max(const float* data, float* result, size_t n, cudaStream_t stream) {
    auto ptr = thrust::device_pointer_cast(data);
    float max_val = thrust::reduce(
        thrust::cuda::par.on(stream),
        ptr, ptr + n,
        std::numeric_limits<float>::lowest(),
        thrust::maximum<float>()
    );
    cudaMemcpyAsync(result, &max_val, sizeof(float), cudaMemcpyHostToDevice, stream);
}

} // namespace gs::tensor_ops