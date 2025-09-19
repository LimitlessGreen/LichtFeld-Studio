/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_ops.hpp"
#include "core/tensor.hpp"
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <numeric>
#include <execution>

namespace gs::tensor_ops {

// CPU implementation of unified operations

namespace cpu {

// ============= Unary Operations =============
template<typename T>
inline T unary_op_impl(T x, UnaryOp op) {
    switch (op) {
        case UnaryOp::Neg: return -x;
        case UnaryOp::Abs: return std::abs(x);
        case UnaryOp::Sign: return (x > 0) - (x < 0);
        case UnaryOp::Reciprocal: return T(1) / (x + T(1e-8));

        case UnaryOp::Exp: return std::exp(x);
        case UnaryOp::Exp2: return std::exp2(x);
        case UnaryOp::Log: return std::log(std::max(T(1e-10), x));
        case UnaryOp::Log2: return std::log2(std::max(T(1e-10), x));
        case UnaryOp::Log10: return std::log10(std::max(T(1e-10), x));
        case UnaryOp::Log1p: return std::log1p(x);

        case UnaryOp::Sqrt: return std::sqrt(std::max(T(0), x));
        case UnaryOp::Rsqrt: return T(1) / std::sqrt(std::max(T(1e-10), x));
        case UnaryOp::Square: return x * x;

        case UnaryOp::Sin: return std::sin(x);
        case UnaryOp::Cos: return std::cos(x);
        case UnaryOp::Tan: return std::tan(x);
        case UnaryOp::Asin: return std::asin(std::clamp(x, T(-1), T(1)));
        case UnaryOp::Acos: return std::acos(std::clamp(x, T(-1), T(1)));
        case UnaryOp::Atan: return std::atan(x);

        case UnaryOp::Sinh: return std::sinh(x);
        case UnaryOp::Cosh: return std::cosh(x);
        case UnaryOp::Tanh: return std::tanh(x);

        case UnaryOp::Sigmoid: return T(1) / (T(1) + std::exp(-x));
        case UnaryOp::Relu: return std::max(T(0), x);
        case UnaryOp::Gelu: {
            T inner = std::sqrt(T(2) / T(M_PI)) * (x + T(0.044715) * x * x * x);
            return T(0.5) * x * (T(1) + std::tanh(inner));
        }
        case UnaryOp::Swish: return x / (T(1) + std::exp(-x));

        case UnaryOp::Floor: return std::floor(x);
        case UnaryOp::Ceil: return std::ceil(x);
        case UnaryOp::Round: return std::round(x);
        case UnaryOp::Trunc: return std::trunc(x);

        default: return x;
    }
}

template<typename T>
void apply_unary_op(const T* input, T* output, size_t n, UnaryOp op) {
    #pragma omp parallel for if(n > 1024)
    for (size_t i = 0; i < n; ++i) {
        output[i] = unary_op_impl(input[i], op);
    }
}

template<typename T>
void apply_unary_op_inplace(T* data, size_t n, UnaryOp op) {
    #pragma omp parallel for if(n > 1024)
    for (size_t i = 0; i < n; ++i) {
        data[i] = unary_op_impl(data[i], op);
    }
}

// ============= Binary Operations =============
template<typename T>
inline T binary_op_impl(T a, T b, BinaryOp op) {
    switch (op) {
        case BinaryOp::Add: return a + b;
        case BinaryOp::Sub: return a - b;
        case BinaryOp::Mul: return a * b;
        case BinaryOp::Div: return a / (b + T(1e-8));
        case BinaryOp::Pow: return std::pow(a, b);
        case BinaryOp::Mod: return std::fmod(a, b);
        case BinaryOp::Maximum: return std::max(a, b);
        case BinaryOp::Minimum: return std::min(a, b);
        default: return T(0);
    }
}

template<typename T>
inline bool comparison_op_impl(T a, T b, BinaryOp op) {
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

inline bool logical_op_impl(bool a, bool b, BinaryOp op) {
    switch (op) {
        case BinaryOp::LogicalAnd: return a && b;
        case BinaryOp::LogicalOr: return a || b;
        case BinaryOp::LogicalXor: return a != b;
        default: return false;
    }
}

template<typename T>
void apply_binary_op(const T* a, const T* b, T* c, size_t n, BinaryOp op) {
    #pragma omp parallel for if(n > 1024)
    for (size_t i = 0; i < n; ++i) {
        c[i] = binary_op_impl(a[i], b[i], op);
    }
}

template<typename T>
void apply_binary_scalar_op(const T* data, T scalar, T* result, size_t n, BinaryOp op) {
    #pragma omp parallel for if(n > 1024)
    for (size_t i = 0; i < n; ++i) {
        result[i] = binary_op_impl(data[i], scalar, op);
    }
}

template<typename T>
void apply_binary_op_inplace(T* a, const T* b, size_t n, BinaryOp op) {
    #pragma omp parallel for if(n > 1024)
    for (size_t i = 0; i < n; ++i) {
        a[i] = binary_op_impl(a[i], b[i], op);
    }
}

template<typename T>
void apply_binary_scalar_op_inplace(T* data, T scalar, size_t n, BinaryOp op) {
    #pragma omp parallel for if(n > 1024)
    for (size_t i = 0; i < n; ++i) {
        data[i] = binary_op_impl(data[i], scalar, op);
    }
}

// ============= Reduce Operations =============
template<typename T, typename ReduceFunc>
T reduce_op(const T* data, size_t n, T init, ReduceFunc func) {
    if (n == 0) return init;

    T result = init;

    #pragma omp parallel reduction(+:result) if(n > 1024)
    {
        T local_result = init;
        #pragma omp for nowait
        for (size_t i = 0; i < n; ++i) {
            local_result = func(local_result, data[i]);
        }
        result = func(result, local_result);
    }

    return result;
}

float reduce_sum(const float* data, size_t n) {
    return std::reduce(std::execution::par_unseq, data, data + n, 0.0f);
}

float reduce_mean(const float* data, size_t n) {
    return n > 0 ? reduce_sum(data, n) / n : 0.0f;
}

float reduce_max(const float* data, size_t n) {
    return n > 0 ? *std::max_element(std::execution::par_unseq, data, data + n) : -FLT_MAX;
}

float reduce_min(const float* data, size_t n) {
    return n > 0 ? *std::min_element(std::execution::par_unseq, data, data + n) : FLT_MAX;
}

float reduce_prod(const float* data, size_t n) {
    return std::reduce(std::execution::par_unseq, data, data + n, 1.0f, std::multiplies<float>());
}

// ============= Ternary Operations =============
void muladd(const float* a, const float* b, const float* c, float* output, size_t n) {
    #pragma omp parallel for if(n > 1024)
    for (size_t i = 0; i < n; ++i) {
        output[i] = a[i] * b[i] + c[i];
    }
}

void clamp(const float* x, float min_val, float max_val, float* output, size_t n) {
    #pragma omp parallel for if(n > 1024)
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::min(std::max(x[i], min_val), max_val);
    }
}

void clamp_inplace(float* data, float min_val, float max_val, size_t n) {
    #pragma omp parallel for if(n > 1024)
    for (size_t i = 0; i < n; ++i) {
        data[i] = std::min(std::max(data[i], min_val), max_val);
    }
}

} // namespace cpu

// ============= CPU Launch Functions =============
// These would be called when device == Device::CPU

void launch_unary_op_cpu(const void* input, void* output,
                        size_t n, UnaryOp op, DataType dtype) {
    if (dtype == DataType::Float32) {
        cpu::apply_unary_op(static_cast<const float*>(input),
                           static_cast<float*>(output), n, op);
    } else if (dtype == DataType::Bool && op == UnaryOp::LogicalNot) {
        const unsigned char* src = static_cast<const unsigned char*>(input);
        unsigned char* dst = static_cast<unsigned char*>(output);
        #pragma omp parallel for if(n > 1024)
        for (size_t i = 0; i < n; ++i) {
            dst[i] = !src[i];
        }
    }
}

void launch_unary_op_inplace_cpu(void* data, size_t n,
                                 UnaryOp op, DataType dtype) {
    if (dtype == DataType::Float32) {
        cpu::apply_unary_op_inplace(static_cast<float*>(data), n, op);
    }
}

void launch_binary_op_cpu(const void* a, const void* b, void* c,
                          size_t n, BinaryOp op,
                          DataType a_dtype, DataType b_dtype, DataType c_dtype) {
    if (c_dtype == DataType::Bool && a_dtype == DataType::Float32) {
        // Comparison operations
        const float* fa = static_cast<const float*>(a);
        const float* fb = static_cast<const float*>(b);
        unsigned char* uc = static_cast<unsigned char*>(c);

        #pragma omp parallel for if(n > 1024)
        for (size_t i = 0; i < n; ++i) {
            uc[i] = cpu::comparison_op_impl(fa[i], fb[i], op) ? 1 : 0;
        }
    } else if (a_dtype == DataType::Bool && b_dtype == DataType::Bool && c_dtype == DataType::Bool) {
        // Logical operations
        const unsigned char* ua = static_cast<const unsigned char*>(a);
        const unsigned char* ub = static_cast<const unsigned char*>(b);
        unsigned char* uc = static_cast<unsigned char*>(c);

        #pragma omp parallel for if(n > 1024)
        for (size_t i = 0; i < n; ++i) {
            uc[i] = cpu::logical_op_impl(ua[i] != 0, ub[i] != 0, op) ? 1 : 0;
        }
    } else if (a_dtype == DataType::Float32 && b_dtype == DataType::Float32 && c_dtype == DataType::Float32) {
        cpu::apply_binary_op(static_cast<const float*>(a),
                            static_cast<const float*>(b),
                            static_cast<float*>(c), n, op);
    }
}

void launch_binary_scalar_cpu(const void* data, float scalar, void* result,
                              size_t n, BinaryOp op,
                              DataType src_dtype, DataType dst_dtype) {
    if (dst_dtype == DataType::Bool && src_dtype == DataType::Float32) {
        const float* fdata = static_cast<const float*>(data);
        unsigned char* uresult = static_cast<unsigned char*>(result);

        #pragma omp parallel for if(n > 1024)
        for (size_t i = 0; i < n; ++i) {
            uresult[i] = cpu::comparison_op_impl(fdata[i], scalar, op) ? 1 : 0;
        }
    } else if (src_dtype == DataType::Float32 && dst_dtype == DataType::Float32) {
        cpu::apply_binary_scalar_op(static_cast<const float*>(data),
                                    scalar,
                                    static_cast<float*>(result), n, op);
    }
}

void launch_binary_op_inplace_cpu(void* a, const void* b, size_t n, BinaryOp op) {
    cpu::apply_binary_op_inplace(static_cast<float*>(a),
                                 static_cast<const float*>(b), n, op);
}

void launch_binary_scalar_inplace_cpu(void* data, float scalar, size_t n, BinaryOp op) {
    cpu::apply_binary_scalar_op_inplace(static_cast<float*>(data), scalar, n, op);
}

void launch_reduce_op_cpu(const void* input, void* output,
                         const size_t* shape, size_t rank,
                         const int* axes, size_t num_axes,
                         bool keepdim, ReduceOp op, DataType dtype) {
    // Simplified - just handle full reduction for now
    size_t n = 1;
    for (size_t i = 0; i < rank; ++i) {
        n *= shape[i];
    }

    if (n == 0) return;

    if (dtype == DataType::Float32) {
        const float* fdata = static_cast<const float*>(input);
        float* foutput = static_cast<float*>(output);

        float result = 0.0f;
        switch (op) {
            case ReduceOp::Sum:
                result = cpu::reduce_sum(fdata, n);
                break;
            case ReduceOp::Mean:
                result = cpu::reduce_mean(fdata, n);
                break;
            case ReduceOp::Max:
                result = cpu::reduce_max(fdata, n);
                break;
            case ReduceOp::Min:
                result = cpu::reduce_min(fdata, n);
                break;
            case ReduceOp::Prod:
                result = cpu::reduce_prod(fdata, n);
                break;
            default:
                result = 0.0f;
        }

        *foutput = result;
    }
}

void launch_ternary_op_cpu(const void* a, const void* b, const void* c, void* output,
                          size_t n, TernaryOp op, DataType dtype) {
    if (dtype == DataType::Float32) {
        const float* fa = static_cast<const float*>(a);
        const float* fb = static_cast<const float*>(b);
        const float* fc = static_cast<const float*>(c);
        float* foutput = static_cast<float*>(output);

        switch (op) {
            case TernaryOp::MulAdd:
                cpu::muladd(fa, fb, fc, foutput, n);
                break;
            case TernaryOp::Clamp:
                // For clamp, b and c should be scalars
                if (n > 0) {
                    float min_val = fb[0];
                    float max_val = fc[0];
                    cpu::clamp(fa, min_val, max_val, foutput, n);
                }
                break;
            default:
                break;
        }
    }
}

} // namespace gs::tensor_ops