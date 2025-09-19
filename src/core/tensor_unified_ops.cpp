/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include "core/tensor_broadcast.hpp"
#include "core/logger.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <numeric>
#include <algorithm>

namespace gs {

    // ============= CORE UNIFIED OPERATIONS =============

Tensor Tensor::load(LoadOp op, const LoadArgs& args) {
    Tensor result;

    switch (op) {
        case LoadOp::Empty: {
            result.shape_ = args.shape;
            result.device_ = args.device;
            result.dtype_ = args.dtype;
            result.owns_memory_ = true;
            result.initialized_ = true;
            result.id_ = next_id_++;

            size_t bytes = result.numel() * dtype_size(result.dtype_);
            if (bytes == 0) {
                return result;
            }

            if (result.device_ == Device::CUDA) {
                cudaError_t err = cudaMalloc(&result.data_, bytes);
                if (err != cudaSuccess) {
                    LOG_ERROR("Failed to allocate {} bytes on CUDA: {}", bytes, cudaGetErrorString(err));
                    result.initialized_ = false;
                    return result;
                }
            } else {
                result.data_ = std::malloc(bytes);
                if (!result.data_) {
                    LOG_ERROR("Failed to allocate {} bytes on CPU", bytes);
                    result.initialized_ = false;
                    return result;
                }
            }
            break;
        }

        case LoadOp::Const: {
            float value = std::get<float>(args.args);
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0) return result;

            if (result.device_ == Device::CUDA) {
                if (result.dtype_ == DataType::Float32) {
                    if (value == 0.0f) {
                        cudaMemset(result.data_, 0, result.bytes());
                    } else {
                        tensor_ops::launch_load_op(
                            result.data_,
                            result.shape_.dims().data(),
                            result.shape_.rank(),
                            LoadOp::Const,
                            &value,
                            result.dtype_,
                            nullptr
                        );
                        cudaDeviceSynchronize();
                    }
                } else if (result.dtype_ == DataType::Bool) {
                    unsigned char fill_val = (value != 0.0f) ? 1 : 0;
                    cudaMemset(result.data_, fill_val, result.bytes());
                } else if (result.dtype_ == DataType::Int32) {
                    if (value == 0.0f) {
                        cudaMemset(result.data_, 0, result.bytes());
                    } else {
                        std::vector<int> temp(result.numel(), static_cast<int>(value));
                        cudaMemcpy(result.data_, temp.data(), result.bytes(), cudaMemcpyHostToDevice);
                    }
                }
            } else {
                if (result.dtype_ == DataType::Float32) {
                    float* ptr = static_cast<float*>(result.data_);
                    std::fill_n(ptr, result.numel(), value);
                } else if (result.dtype_ == DataType::Bool) {
                    unsigned char* ptr = static_cast<unsigned char*>(result.data_);
                    std::fill_n(ptr, result.numel(), value != 0 ? 1 : 0);
                } else if (result.dtype_ == DataType::Int32) {
                    int* ptr = static_cast<int*>(result.data_);
                    std::fill_n(ptr, result.numel(), static_cast<int>(value));
                }
            }
            break;
        }

        case LoadOp::Arange: {
            auto [start, end, step] = std::get<std::tuple<float, float, float>>(args.args);

            if (step == 0) {
                LOG_ERROR("Step cannot be zero");
                return Tensor();
            }

            if ((end - start) * step < 0) {
                LOG_ERROR("Invalid range: start={}, end={}, step={}", start, end, step);
                return Tensor();
            }

            size_t count = static_cast<size_t>(std::ceil((end - start) / step));

            result.shape_ = TensorShape{count};
            result.device_ = args.device;
            result.dtype_ = args.dtype;
            result.owns_memory_ = true;
            result.initialized_ = true;
            result.id_ = next_id_++;

            size_t bytes = count * dtype_size(result.dtype_);

            if (result.device_ == Device::CUDA) {
                cudaMalloc(&result.data_, bytes);
                if (!result.data_) {
                    LOG_ERROR("Failed to allocate {} bytes on CUDA", bytes);
                    result.initialized_ = false;
                    return result;
                }

                // Generate on CPU and copy
                if (result.dtype_ == DataType::Float32) {
                    std::vector<float> data(count);
                    for (size_t i = 0; i < count; ++i) {
                        data[i] = start + i * step;
                    }
                    cudaMemcpy(result.data_, data.data(), bytes, cudaMemcpyHostToDevice);
                } else if (result.dtype_ == DataType::Int32) {
                    std::vector<int> data(count);
                    for (size_t i = 0; i < count; ++i) {
                        data[i] = static_cast<int>(start + i * step);
                    }
                    cudaMemcpy(result.data_, data.data(), bytes, cudaMemcpyHostToDevice);
                }
            } else {
                result.data_ = std::malloc(bytes);
                if (!result.data_) {
                    LOG_ERROR("Failed to allocate {} bytes on CPU", bytes);
                    result.initialized_ = false;
                    return result;
                }

                if (result.dtype_ == DataType::Float32) {
                    float* ptr = static_cast<float*>(result.data_);
                    for (size_t i = 0; i < count; ++i) {
                        ptr[i] = start + i * step;
                    }
                } else if (result.dtype_ == DataType::Int32) {
                    int* ptr = static_cast<int*>(result.data_);
                    for (size_t i = 0; i < count; ++i) {
                        ptr[i] = static_cast<int>(start + i * step);
                    }
                }
            }
            break;
        }

        case LoadOp::Random: {
            auto [low, high] = std::get<std::pair<float, float>>(args.args);
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0) return result;

            if (result.device_ == Device::CUDA) {
                if (result.dtype_ == DataType::Float32) {
                    tensor_ops::launch_uniform(result.ptr<float>(), result.numel(), low, high,
                                             RandomGenerator::instance().get_seed(), 0);
                    cudaDeviceSynchronize();
                } else if (result.dtype_ == DataType::Int32) {
                    tensor_ops::launch_randint(result.ptr<int>(), result.numel(),
                                              static_cast<int>(low), static_cast<int>(high),
                                              RandomGenerator::instance().get_seed(), 0);
                    cudaDeviceSynchronize();
                }
            } else {
                auto& gen = *static_cast<std::mt19937_64*>(
                    RandomGenerator::instance().get_generator(Device::CPU));

                if (result.dtype_ == DataType::Float32) {
                    std::uniform_real_distribution<float> dist(low, high);
                    float* data = result.ptr<float>();
                    for (size_t i = 0; i < result.numel(); ++i) {
                        data[i] = dist(gen);
                    }
                } else if (result.dtype_ == DataType::Int32) {
                    std::uniform_int_distribution<int> dist(static_cast<int>(low),
                                                           static_cast<int>(high) - 1);
                    int* data = result.ptr<int>();
                    for (size_t i = 0; i < result.numel(); ++i) {
                        data[i] = dist(gen);
                    }
                }
            }
            break;
        }

        case LoadOp::Normal: {
            auto [mean, std] = std::get<std::pair<float, float>>(args.args);
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0) return result;

            if (result.device_ == Device::CUDA) {
                tensor_ops::launch_normal(result.ptr<float>(), result.numel(), mean, std,
                                        RandomGenerator::instance().get_seed(), 0);
                cudaDeviceSynchronize();
            } else {
                auto& gen = *static_cast<std::mt19937_64*>(
                    RandomGenerator::instance().get_generator(Device::CPU));
                std::normal_distribution<float> dist(mean, std);
                float* data = result.ptr<float>();
                for (size_t i = 0; i < result.numel(); ++i) {
                    data[i] = dist(gen);
                }
            }
            break;
        }
        case LoadOp::Randint: {
            auto [low, high] = std::get<std::pair<int, int>>(args.args);
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0) return result;

            if (result.device_ == Device::CUDA) {
                if (result.dtype_ == DataType::Int32) {
                    tensor_ops::launch_randint(result.ptr<int>(), result.numel(), low, high,
                                              RandomGenerator::instance().get_seed(), 0);
                    cudaDeviceSynchronize();
                } else if (result.dtype_ == DataType::Float32) {
                    // Generate ints in a temporary buffer, then convert to float
                    int* temp_buffer;
                    cudaMalloc(&temp_buffer, result.numel() * sizeof(int));

                    tensor_ops::launch_randint(temp_buffer, result.numel(), low, high,
                                              RandomGenerator::instance().get_seed(), 0);

                    // Convert int to float using a kernel
                    tensor_ops::launch_int_to_float(temp_buffer, result.ptr<float>(),
                                                   result.numel(), 0);
                    cudaDeviceSynchronize();

                    cudaFree(temp_buffer);
                }
            } else {
                auto& gen = *static_cast<std::mt19937_64*>(
                    RandomGenerator::instance().get_generator(Device::CPU));
                std::uniform_int_distribution<int> dist(low, high - 1);

                if (result.dtype_ == DataType::Int32) {
                    int* data = result.ptr<int>();
                    for (size_t i = 0; i < result.numel(); ++i) {
                        data[i] = dist(gen);
                    }
                } else if (result.dtype_ == DataType::Float32) {
                    float* data = result.ptr<float>();
                    for (size_t i = 0; i < result.numel(); ++i) {
                        data[i] = static_cast<float>(dist(gen));
                    }
                }
            }
            break;
        }


        case LoadOp::Bernoulli: {
            float p = std::get<float>(args.args);
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0) return result;

            if (result.device_ == Device::CUDA) {
                tensor_ops::launch_bernoulli(result.ptr<float>(), result.numel(), p,
                                            RandomGenerator::instance().get_seed(), 0);
                cudaDeviceSynchronize();
            } else {
                auto& gen = *static_cast<std::mt19937_64*>(
                    RandomGenerator::instance().get_generator(Device::CPU));
                std::bernoulli_distribution dist(p);
                float* data = result.ptr<float>();
                for (size_t i = 0; i < result.numel(); ++i) {
                    data[i] = dist(gen) ? 1.0f : 0.0f;
                }
            }
            break;
        }

        case LoadOp::Multinomial: {
            auto [weights_ptr, replacement] = std::get<std::pair<void*, bool>>(args.args);
            const Tensor* weights = static_cast<const Tensor*>(weights_ptr);

            if (!weights->is_valid() || weights->ndim() != 1) {
                LOG_ERROR("Multinomial requires 1D weight tensor");
                return Tensor();
            }

            size_t n = weights->numel();
            size_t num_samples = args.shape.elements();

            result = load(LoadOp::Empty, args);
            if (!result.is_valid()) return result;

            if (weights->device() == Device::CUDA) {
                tensor_ops::launch_multinomial(weights->ptr<float>(), result.ptr<int>(),
                                              n, num_samples, replacement,
                                              RandomGenerator::instance().get_seed(), 0);
                cudaDeviceSynchronize();
            } else {
                // CPU implementation
                auto weights_data = weights->to_vector();

                // Normalize
                float sum = std::accumulate(weights_data.begin(), weights_data.end(), 0.0f);
                if (sum <= 0) {
                    LOG_ERROR("Weights must sum to positive value");
                    return Tensor();
                }

                // Create CDF
                std::vector<float> cdf(n);
                cdf[0] = weights_data[0] / sum;
                for (size_t i = 1; i < n; ++i) {
                    cdf[i] = cdf[i-1] + weights_data[i] / sum;
                }

                // Sample
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<float> dis(0.0f, 1.0f);

                int* samples = result.ptr<int>();
                for (size_t i = 0; i < num_samples; ++i) {
                    float u = dis(gen);
                    auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
                    samples[i] = static_cast<int>(std::distance(cdf.begin(), it));
                }
            }
            break;
        }

        case LoadOp::Eye: {
            result = load(LoadOp::Const, {args.shape, args.device, args.dtype, 0.0f});
            if (!result.is_valid() || args.shape.rank() != 2) return result;

            size_t m = args.shape[0];
            size_t n = args.shape[1];
            size_t min_dim = std::min(m, n);

            if (result.device_ == Device::CUDA) {
                tensor_ops::launch_eye(result.ptr<float>(), m, n, 0);
                cudaDeviceSynchronize();
            } else {
                float* data = result.ptr<float>();
                for (size_t i = 0; i < min_dim; ++i) {
                    data[i * n + i] = 1.0f;
                }
            }
            break;
        }

        case LoadOp::FromCPU: {
            void* src_ptr = std::get<void*>(args.args);
            if (!src_ptr) {
                LOG_ERROR("FromCPU requires valid source pointer");
                return Tensor();
            }

            result = Tensor(src_ptr, args.shape, Device::CPU, args.dtype);
            result.initialized_ = true;

            if (args.device == Device::CUDA) {
                result = result.to(Device::CUDA);
            }
            break;
        }

        case LoadOp::FromCUDA: {
            void* src_ptr = std::get<void*>(args.args);
            if (!src_ptr) {
                LOG_ERROR("FromCUDA requires valid source pointer");
                return Tensor();
            }

            result = Tensor(src_ptr, args.shape, Device::CUDA, args.dtype);
            result.initialized_ = true;

            if (args.device == Device::CPU) {
                result = result.to(Device::CPU);
            }
            break;
        }

        default:
            LOG_ERROR("Unknown load operation");
            break;
    }

    return result;
}

// Static method for multinomial
Tensor Tensor::multinomial(const Tensor& weights, int num_samples, bool replacement) {
    LoadArgs args;
    args.shape = TensorShape({static_cast<size_t>(num_samples)});
    args.device = weights.device();
    args.dtype = DataType::Int32;
    args.args = std::pair<void*, bool>{const_cast<void*>(static_cast<const void*>(&weights)), replacement};
    return load(LoadOp::Multinomial, args);
}

Tensor Tensor::unary(UnaryOp op, const UnaryArgs& args) const {
    if (!is_valid()) {
        LOG_ERROR("Unary operation on invalid tensor");
        return Tensor();
    }

    auto result = Tensor::empty(shape_, device_,
                               (op >= UnaryOp::IsNan && op <= UnaryOp::LogicalNot)
                               ? DataType::Bool : dtype_);

    if (device_ == Device::CUDA) {
        tensor_ops::launch_unary_op(
            data_, result.data_,
            numel(), op,
            dtype_, nullptr
        );
        cudaDeviceSynchronize();
    } else {
        // CPU implementation
        const float* src = static_cast<const float*>(data_);
        float* dst = static_cast<float*>(result.data_);

        for (size_t i = 0; i < numel(); ++i) {
            switch (op) {
            case UnaryOp::Neg: dst[i] = -src[i]; break;
            case UnaryOp::Abs: dst[i] = std::abs(src[i]); break;
            case UnaryOp::Exp: dst[i] = std::exp(src[i]); break;
            case UnaryOp::Log: dst[i] = std::log(std::max(src[i], 1e-45f)); break;
            case UnaryOp::Sqrt: dst[i] = std::sqrt(std::max(src[i], 0.0f)); break;
            case UnaryOp::Square: dst[i] = src[i] * src[i]; break;
            case UnaryOp::Sigmoid: dst[i] = 1.0f / (1.0f + std::exp(-src[i])); break;
            case UnaryOp::Relu: dst[i] = std::max(src[i], 0.0f); break;
            case UnaryOp::Sin: dst[i] = std::sin(src[i]); break;
            case UnaryOp::Cos: dst[i] = std::cos(src[i]); break;
            case UnaryOp::Tanh: dst[i] = std::tanh(src[i]); break;
            case UnaryOp::Floor: dst[i] = std::floor(src[i]); break;
            case UnaryOp::Ceil: dst[i] = std::ceil(src[i]); break;
            case UnaryOp::Round: dst[i] = std::round(src[i]); break;
            default: dst[i] = src[i]; break;
            }
        }
    }

    return result;
}

// Binary operations - NO TEMPLATES!
Tensor Tensor::binary_op_impl(const Tensor& other, BinaryOp op) const {
    if (!is_valid() || !other.is_valid()) {
        LOG_ERROR("Binary operation on invalid tensor");
        return Tensor();
    }

    // Get broadcast shape
    auto broadcast_shape = this->broadcast_shape(other.shape());
    if (broadcast_shape.rank() == 0 || broadcast_shape.elements() == 0) {
        LOG_ERROR("Incompatible shapes for broadcasting: {} vs {}",
                  shape_.str(), other.shape().str());
        return Tensor();
    }

    // Determine output dtype
    DataType out_dtype = dtype_;
    if (op >= BinaryOp::Equal && op <= BinaryOp::GreaterEqual) {
        out_dtype = DataType::Bool;
    } else if (op >= BinaryOp::LogicalAnd && op <= BinaryOp::LogicalXor) {
        out_dtype = DataType::Bool;
    } else {
        out_dtype = promote_types(dtype_, other.dtype());
    }

    // Create result with broadcast shape
    auto result = Tensor::empty(broadcast_shape, device_, out_dtype);

    // Check if we need broadcasting
    bool a_needs_broadcast = (shape_ != broadcast_shape);
    bool b_needs_broadcast = (other.shape() != broadcast_shape);

    if (!a_needs_broadcast && !b_needs_broadcast) {
        // No broadcasting needed - direct operation
        if (device_ == Device::CUDA) {
            tensor_ops::launch_binary_op(
                raw_ptr(), other.raw_ptr(), result.raw_ptr(),
                result.numel(), op,
                dtype_, other.dtype(), result.dtype(),
                nullptr
            );
            cudaDeviceSynchronize();
        } else {
            // CPU implementation for same-shape tensors
            const float* src_a = static_cast<const float*>(raw_ptr());
            const float* src_b = static_cast<const float*>(other.raw_ptr());

            if (out_dtype == DataType::Bool) {
                unsigned char* dst = static_cast<unsigned char*>(result.raw_ptr());
                for (size_t i = 0; i < result.numel(); ++i) {
                    bool res = false;
                    switch (op) {
                    case BinaryOp::Equal: res = (src_a[i] == src_b[i]); break;
                    case BinaryOp::NotEqual: res = (src_a[i] != src_b[i]); break;
                    case BinaryOp::Less: res = (src_a[i] < src_b[i]); break;
                    case BinaryOp::LessEqual: res = (src_a[i] <= src_b[i]); break;
                    case BinaryOp::Greater: res = (src_a[i] > src_b[i]); break;
                    case BinaryOp::GreaterEqual: res = (src_a[i] >= src_b[i]); break;
                    case BinaryOp::LogicalAnd: res = (src_a[i] != 0) && (src_b[i] != 0); break;
                    case BinaryOp::LogicalOr: res = (src_a[i] != 0) || (src_b[i] != 0); break;
                    case BinaryOp::LogicalXor: res = (src_a[i] != 0) != (src_b[i] != 0); break;
                    default: break;
                    }
                    dst[i] = res ? 1 : 0;
                }
            } else {
                float* dst = static_cast<float*>(result.raw_ptr());
                for (size_t i = 0; i < result.numel(); ++i) {
                    switch (op) {
                    case BinaryOp::Add: dst[i] = src_a[i] + src_b[i]; break;
                    case BinaryOp::Sub: dst[i] = src_a[i] - src_b[i]; break;
                    case BinaryOp::Mul: dst[i] = src_a[i] * src_b[i]; break;
                    case BinaryOp::Div:
                        dst[i] = src_a[i] / (std::abs(src_b[i]) < 1e-7f ?
                                             (src_b[i] < 0 ? -1e-7f : 1e-7f) : src_b[i]);
                        break;
                    case BinaryOp::Pow: dst[i] = std::pow(src_a[i], src_b[i]); break;
                    case BinaryOp::Maximum: dst[i] = std::max(src_a[i], src_b[i]); break;
                    case BinaryOp::Minimum: dst[i] = std::min(src_a[i], src_b[i]); break;
                    default: dst[i] = src_a[i]; break;
                    }
                }
            }
        }
    } else {
        // Broadcasting needed - we need to handle this specially
        if (device_ == Device::CUDA) {
            // For CUDA, we need a broadcasting-aware kernel
            // This should be implemented in tensor_ops.cu to handle broadcasting directly

            // For now, let's broadcast first then operate
            if (a_needs_broadcast) {
                auto a_broadcast = broadcast_to(broadcast_shape);
                if (b_needs_broadcast) {
                    auto b_broadcast = other.broadcast_to(broadcast_shape);
                    tensor_ops::launch_binary_op(
                        a_broadcast.raw_ptr(), b_broadcast.raw_ptr(), result.raw_ptr(),
                        result.numel(), op,
                        a_broadcast.dtype(), b_broadcast.dtype(), result.dtype(),
                        nullptr
                    );
                } else {
                    tensor_ops::launch_binary_op(
                        a_broadcast.raw_ptr(), other.raw_ptr(), result.raw_ptr(),
                        result.numel(), op,
                        a_broadcast.dtype(), other.dtype(), result.dtype(),
                        nullptr
                    );
                }
            } else {
                auto b_broadcast = other.broadcast_to(broadcast_shape);
                tensor_ops::launch_binary_op(
                    raw_ptr(), b_broadcast.raw_ptr(), result.raw_ptr(),
                    result.numel(), op,
                    dtype_, b_broadcast.dtype(), result.dtype(),
                    nullptr
                );
            }
            cudaDeviceSynchronize();
        } else {
            // CPU implementation with broadcasting
            const float* src_a = static_cast<const float*>(raw_ptr());
            const float* src_b = static_cast<const float*>(other.raw_ptr());

            auto a_shape = shape_.dims();
            auto b_shape = other.shape().dims();
            auto c_shape = broadcast_shape.dims();

            if (out_dtype == DataType::Bool) {
                unsigned char* dst = static_cast<unsigned char*>(result.raw_ptr());

                for (size_t i = 0; i < result.numel(); ++i) {
                    // Calculate broadcast indices
                    size_t a_idx = broadcast::index(i, c_shape, a_shape);
                    size_t b_idx = broadcast::index(i, c_shape, b_shape);

                    bool res = false;
                    switch (op) {
                    case BinaryOp::Equal: res = (src_a[a_idx] == src_b[b_idx]); break;
                    case BinaryOp::NotEqual: res = (src_a[a_idx] != src_b[b_idx]); break;
                    case BinaryOp::Less: res = (src_a[a_idx] < src_b[b_idx]); break;
                    case BinaryOp::LessEqual: res = (src_a[a_idx] <= src_b[b_idx]); break;
                    case BinaryOp::Greater: res = (src_a[a_idx] > src_b[b_idx]); break;
                    case BinaryOp::GreaterEqual: res = (src_a[a_idx] >= src_b[b_idx]); break;
                    case BinaryOp::LogicalAnd: res = (src_a[a_idx] != 0) && (src_b[b_idx] != 0); break;
                    case BinaryOp::LogicalOr: res = (src_a[a_idx] != 0) || (src_b[b_idx] != 0); break;
                    case BinaryOp::LogicalXor: res = (src_a[a_idx] != 0) != (src_b[b_idx] != 0); break;
                    default: break;
                    }
                    dst[i] = res ? 1 : 0;
                }
            } else {
                float* dst = static_cast<float*>(result.raw_ptr());

                for (size_t i = 0; i < result.numel(); ++i) {
                    // Calculate broadcast indices
                    size_t a_idx = broadcast::index(i, c_shape, a_shape);
                    size_t b_idx = broadcast::index(i, c_shape, b_shape);

                    switch (op) {
                    case BinaryOp::Add: dst[i] = src_a[a_idx] + src_b[b_idx]; break;
                    case BinaryOp::Sub: dst[i] = src_a[a_idx] - src_b[b_idx]; break;
                    case BinaryOp::Mul: dst[i] = src_a[a_idx] * src_b[b_idx]; break;
                    case BinaryOp::Div:
                        dst[i] = src_a[a_idx] / (std::abs(src_b[b_idx]) < 1e-7f ?
                                                 (src_b[b_idx] < 0 ? -1e-7f : 1e-7f) : src_b[b_idx]);
                        break;
                    case BinaryOp::Pow: dst[i] = std::pow(src_a[a_idx], src_b[b_idx]); break;
                    case BinaryOp::Maximum: dst[i] = std::max(src_a[a_idx], src_b[b_idx]); break;
                    case BinaryOp::Minimum: dst[i] = std::min(src_a[a_idx], src_b[b_idx]); break;
                    default: dst[i] = src_a[a_idx]; break;
                    }
                }
            }
        }
    }

    return result;
}

Tensor Tensor::binary_op_scalar(float scalar, BinaryOp op) const {
    if (!is_valid()) {
        LOG_ERROR("Binary operation on invalid tensor");
        return Tensor();
    }

    // Determine output dtype
    DataType out_dtype = dtype_;
    if (op >= BinaryOp::Equal && op <= BinaryOp::GreaterEqual) {
        out_dtype = DataType::Bool;
    }

    auto result = Tensor::empty(shape_, device_, out_dtype);

    if (device_ == Device::CUDA) {
        tensor_ops::launch_binary_scalar(
            raw_ptr(), scalar, result.raw_ptr(),
            numel(), op,
            dtype_, out_dtype,
            nullptr
        );
        cudaDeviceSynchronize();
    } else {
        // CPU implementation
        const float* src = static_cast<const float*>(raw_ptr());

        if (out_dtype == DataType::Bool) {
            unsigned char* dst = static_cast<unsigned char*>(result.raw_ptr());
            for (size_t i = 0; i < numel(); ++i) {
                bool res = false;
                switch (op) {
                case BinaryOp::Equal: res = (src[i] == scalar); break;
                case BinaryOp::NotEqual: res = (src[i] != scalar); break;
                case BinaryOp::Less: res = (src[i] < scalar); break;
                case BinaryOp::LessEqual: res = (src[i] <= scalar); break;
                case BinaryOp::Greater: res = (src[i] > scalar); break;
                case BinaryOp::GreaterEqual: res = (src[i] >= scalar); break;
                default: break;
                }
                dst[i] = res ? 1 : 0;
            }
        } else {
            float* dst = static_cast<float*>(result.raw_ptr());
            for (size_t i = 0; i < numel(); ++i) {
                switch (op) {
                case BinaryOp::Add: dst[i] = src[i] + scalar; break;
                case BinaryOp::Sub: dst[i] = src[i] - scalar; break;
                case BinaryOp::Mul: dst[i] = src[i] * scalar; break;
                case BinaryOp::Div:
                    dst[i] = src[i] / (std::abs(scalar) < 1e-7f ?
                                      (scalar < 0 ? -1e-7f : 1e-7f) : scalar);
                    break;
                case BinaryOp::Pow: dst[i] = std::pow(src[i], scalar); break;
                case BinaryOp::Maximum: dst[i] = std::max(src[i], scalar); break;
                case BinaryOp::Minimum: dst[i] = std::min(src[i], scalar); break;
                default: dst[i] = src[i]; break;
                }
            }
        }
    }

    return result;
}

Tensor& Tensor::binary_op_inplace_impl(const Tensor& other, BinaryOp op) {
    if (!is_valid() || !other.is_valid()) {
        LOG_ERROR("In-place binary operation on invalid tensor");
        return *this;
    }

    if (shape_ != other.shape()) {
        LOG_ERROR("In-place operations require same shapes");
        return *this;
    }

    if (device_ == Device::CUDA) {
        tensor_ops::launch_binary_op_inplace(
            raw_ptr(),
            other.raw_ptr(),
            numel(),
            op,
            nullptr
        );
        cudaDeviceSynchronize();
    } else {
        // CPU implementation
        float* dst = static_cast<float*>(raw_ptr());
        const float* src = static_cast<const float*>(other.raw_ptr());

        for (size_t i = 0; i < numel(); ++i) {
            switch (op) {
            case BinaryOp::Add: dst[i] += src[i]; break;
            case BinaryOp::Sub: dst[i] -= src[i]; break;
            case BinaryOp::Mul: dst[i] *= src[i]; break;
            case BinaryOp::Div:
                dst[i] /= (std::abs(src[i]) < 1e-7f ?
                          (src[i] < 0 ? -1e-7f : 1e-7f) : src[i]);
                break;
            default: break;
            }
        }
    }

    return *this;
}

Tensor& Tensor::binary_op_inplace_scalar(float scalar, BinaryOp op) {
    if (!is_valid()) {
        LOG_ERROR("In-place binary operation on invalid tensor");
        return *this;
    }

    if (device_ == Device::CUDA) {
        tensor_ops::launch_binary_scalar_inplace(
            raw_ptr(),
            scalar,
            numel(),
            op,
            nullptr
        );
        cudaDeviceSynchronize();
    } else {
        // CPU implementation
        float* dst = static_cast<float*>(raw_ptr());

        for (size_t i = 0; i < numel(); ++i) {
            switch (op) {
            case BinaryOp::Add: dst[i] += scalar; break;
            case BinaryOp::Sub: dst[i] -= scalar; break;
            case BinaryOp::Mul: dst[i] *= scalar; break;
            case BinaryOp::Div:
                dst[i] /= (std::abs(scalar) < 1e-7f ?
                          (scalar < 0 ? -1e-7f : 1e-7f) : scalar);
                break;
            default: break;
            }
        }
    }

    return *this;
}

    Tensor Tensor::reduce(ReduceOp op, const ReduceArgs& args) const {
    if (!is_valid()) {
        LOG_ERROR("Reduce operation on invalid tensor");
        return Tensor();
    }

    // Special handling for Std and Var
    if (op == ReduceOp::Std || op == ReduceOp::Var) {
        // First compute the mean
        ReduceArgs mean_args = args;
        auto mean_tensor = reduce(ReduceOp::Mean, mean_args);

        // Broadcast mean to original shape if needed
        Tensor mean_broadcast = (mean_tensor.shape() == shape_)
            ? mean_tensor.clone()
            : mean_tensor.broadcast_to(shape_);

        // Compute (x - mean)^2
        auto diff = this->sub(mean_broadcast);
        auto squared = diff.mul(diff);

        // Compute mean of squared differences (variance)
        auto variance = squared.reduce(ReduceOp::Mean, mean_args);

        if (op == ReduceOp::Var) {
            return variance;
        } else {
            // Std is sqrt of variance
            return variance.sqrt();
        }
    }

    // Handle empty axes (reduce all) - existing code
    std::vector<int> axes = args.axes;
    if (axes.empty()) {
        axes.resize(shape_.rank());
        std::iota(axes.begin(), axes.end(), 0);
    }


    // Calculate output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < shape_.rank(); ++i) {
        bool is_reduced = std::find(axes.begin(), axes.end(), i) != axes.end();
        if (!is_reduced || args.keepdim) {
            out_shape.push_back(is_reduced ? 1 : shape_[i]);
        }
    }

    if (out_shape.empty()) {
        out_shape.push_back(1);  // Scalar result
    }

    // Determine output dtype
    DataType out_dtype = dtype_;
    if (op == ReduceOp::Any || op == ReduceOp::All) {
        out_dtype = DataType::Bool;
    } else if (op == ReduceOp::Argmax || op == ReduceOp::Argmin) {
        out_dtype = DataType::Int64;
    }

    auto result = Tensor::empty(TensorShape(out_shape), device_, out_dtype);

    if (device_ == Device::CUDA) {
        tensor_ops::launch_reduce_op(
            raw_ptr(), result.raw_ptr(),
            shape_.dims().data(), shape_.rank(),
            axes.data(), axes.size(),
            args.keepdim, op,
            dtype_, nullptr
        );
        cudaDeviceSynchronize();
    } else {
        // CPU implementation - simplified
        if (op == ReduceOp::Sum && axes.size() == shape_.rank()) {
            // Full reduction to scalar
            const float* src = static_cast<const float*>(raw_ptr());
            float sum = 0;
            for (size_t i = 0; i < numel(); ++i) {
                sum += src[i];
            }
            *static_cast<float*>(result.raw_ptr()) = sum;
        } else if (op == ReduceOp::Mean && axes.size() == shape_.rank()) {
            // Full reduction to scalar
            const float* src = static_cast<const float*>(raw_ptr());
            float sum = 0;
            for (size_t i = 0; i < numel(); ++i) {
                sum += src[i];
            }
            *static_cast<float*>(result.raw_ptr()) = sum / numel();
        } else if (op == ReduceOp::Max && axes.size() == shape_.rank()) {
            // Full reduction to scalar
            const float* src = static_cast<const float*>(raw_ptr());
            float max_val = src[0];
            for (size_t i = 1; i < numel(); ++i) {
                max_val = std::max(max_val, src[i]);
            }
            *static_cast<float*>(result.raw_ptr()) = max_val;
        } else if (op == ReduceOp::Min && axes.size() == shape_.rank()) {
            // Full reduction to scalar
            const float* src = static_cast<const float*>(raw_ptr());
            float min_val = src[0];
            for (size_t i = 1; i < numel(); ++i) {
                min_val = std::min(min_val, src[i]);
            }
            *static_cast<float*>(result.raw_ptr()) = min_val;
        } else {
            LOG_WARN("Complex CPU reduce not fully implemented");
            std::memcpy(result.raw_ptr(), raw_ptr(),
                       std::min(bytes(), result.bytes()));
        }
    }

    return result;
}

    // In tensor_unified_ops.cpp, find the Tensor::ternary implementation and fix it:

Tensor Tensor::ternary(const Tensor& b, const Tensor& c, TernaryOp op) const {
    if (!is_valid() || !b.is_valid() || !c.is_valid()) {
        LOG_ERROR("Ternary operation on invalid tensor");
        return Tensor();
    }

    // For Where operation, first tensor should be boolean
    if (op == TernaryOp::Where && dtype_ != DataType::Bool) {
        LOG_ERROR("Where operation requires boolean condition tensor");
        return Tensor();
    }

    // Get broadcast shape for all three tensors
    auto shape_ab = this->broadcast_shape(b.shape());
    if (shape_ab.rank() == 0) {
        LOG_ERROR("Incompatible shapes for first two tensors in ternary operation");
        return Tensor();
    }

    // Now get broadcast shape with third tensor
    auto temp = Tensor::empty(shape_ab, device_, dtype_);
    auto shape_abc = temp.broadcast_shape(c.shape());

    if (shape_abc.rank() == 0) {
        LOG_ERROR("Incompatible shapes for ternary operation");
        return Tensor();
    }

    // Determine output dtype based on operation
    DataType out_dtype = dtype_;
    if (op == TernaryOp::Where) {
        // For where, output type should match the true/false value types
        out_dtype = promote_types(b.dtype(), c.dtype());
    }

    auto result = Tensor::empty(shape_abc, device_, out_dtype);

    // Broadcast all tensors to the final shape
    Tensor a_broadcast, b_broadcast, c_broadcast;

    if (shape_ == shape_abc) {
        a_broadcast = clone();
    } else {
        a_broadcast = broadcast_to(shape_abc);
    }

    if (b.shape() == shape_abc) {
        b_broadcast = b.clone();
    } else {
        b_broadcast = b.broadcast_to(shape_abc);
    }

    if (c.shape() == shape_abc) {
        c_broadcast = c.clone();
    } else {
        c_broadcast = c.broadcast_to(shape_abc);
    }

    if (device_ == Device::CUDA) {
        // Use the where kernel for CUDA
        if (op == TernaryOp::Where) {
            tensor_ops::launch_where(
                a_broadcast.ptr<unsigned char>(),  // condition (bool)
                b_broadcast.ptr<float>(),          // true values
                c_broadcast.ptr<float>(),          // false values
                result.ptr<float>(),
                a_broadcast.shape().dims().data(),
                b_broadcast.shape().dims().data(),
                c_broadcast.shape().dims().data(),
                result.shape().dims().data(),
                a_broadcast.shape().rank(),
                b_broadcast.shape().rank(),
                c_broadcast.shape().rank(),
                result.shape().rank(),
                result.numel(),
                0
            );
        } else {
            tensor_ops::launch_ternary_op(
                a_broadcast.raw_ptr(),
                b_broadcast.raw_ptr(),
                c_broadcast.raw_ptr(),
                result.raw_ptr(),
                result.numel(),
                op,
                result.dtype(),
                nullptr
            );
        }
        cudaDeviceSynchronize();
    } else {
        // CPU implementation
        if (op == TernaryOp::Where) {
            const unsigned char* cond = static_cast<const unsigned char*>(a_broadcast.raw_ptr());
            const float* x = static_cast<const float*>(b_broadcast.raw_ptr());
            const float* y = static_cast<const float*>(c_broadcast.raw_ptr());
            float* dst = static_cast<float*>(result.raw_ptr());

            for (size_t i = 0; i < result.numel(); ++i) {
                dst[i] = cond[i] ? x[i] : y[i];
            }
        } else if (op == TernaryOp::Clamp) {
            const float* src = static_cast<const float*>(a_broadcast.raw_ptr());
            const float* min_vals = static_cast<const float*>(b_broadcast.raw_ptr());
            const float* max_vals = static_cast<const float*>(c_broadcast.raw_ptr());
            float* dst = static_cast<float*>(result.raw_ptr());

            for (size_t i = 0; i < result.numel(); ++i) {
                dst[i] = std::max(min_vals[i], std::min(max_vals[i], src[i]));
            }
        }
    }

    return result;
}

Tensor Tensor::where(const Tensor& condition, const Tensor& x, const Tensor& y) {
    return condition.ternary(x, y, TernaryOp::Where);
}

float Tensor::norm(float p) const {
    if (!is_valid()) return 0.0f;

    if (p == 2.0f) {
        // L2 norm: sqrt(sum(x^2))
        auto squared = this->mul(*this);
        return std::sqrt(squared.sum_scalar());
    } else if (p == 1.0f) {
        // L1 norm: sum(|x|)
        return this->abs().sum_scalar();
    } else {
        // Lp norm: (sum(|x|^p))^(1/p)
        auto abs_vals = this->abs();
        auto powered = abs_vals.pow(p);
        auto sum = powered.sum_scalar();
        return std::pow(sum, 1.0f / p);
    }
}


// Broadcasting helper
std::pair<Tensor, Tensor> Tensor::_broadcasted(const Tensor& other, bool match_dtype) const {
    if (!is_valid() || !other.is_valid()) {
        return {Tensor(), Tensor()};
    }

    // Get broadcast shape
    auto bcast_shape = this->broadcast_shape(other.shape());
    if (bcast_shape.rank() == 0) {
        LOG_ERROR("Incompatible shapes for broadcasting");
        return {Tensor(), Tensor()};
    }

    // Broadcast both tensors (clone if already right shape to avoid copy issues)
    Tensor a_broadcast = (shape_ == bcast_shape) ? this->clone() : broadcast_to(bcast_shape);
    Tensor b_broadcast = (other.shape() == bcast_shape) ? other.clone() : other.broadcast_to(bcast_shape);

    // Match dtypes if requested
    if (match_dtype && dtype_ != other.dtype()) {
        auto common_dtype = promote_types(dtype_, other.dtype());
        if (a_broadcast.dtype() != common_dtype) {
            a_broadcast = a_broadcast.to(common_dtype);
        }
        if (b_broadcast.dtype() != common_dtype) {
            b_broadcast = b_broadcast.to(common_dtype);
        }
    }

    return {std::move(a_broadcast), std::move(b_broadcast)};
}

} // namespace gs