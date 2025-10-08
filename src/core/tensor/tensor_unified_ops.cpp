/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "core/tensor_broadcast.hpp"
#include "core/tensor_ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>

namespace gs {

    // ============= CORE UNIFIED OPERATIONS =============

    Tensor Tensor::load(LoadOp op, const LoadArgs& args) {
        Tensor result;

        switch (op) {
        case LoadOp::Empty: {
            result.shape_ = args.shape;
            result.device_ = args.device;
            result.dtype_ = args.dtype;
            result.initialized_ = true;
            result.id_ = next_id_++;

            size_t bytes = result.numel() * dtype_size(result.dtype_);
            if (bytes == 0) {
                return result;
            }

            if (result.device_ == Device::CUDA) {
                void* ptr = nullptr;
                cudaError_t err = cudaMalloc(&ptr, bytes);
                if (err != cudaSuccess) {
                    LOG_ERROR("Failed to allocate {} bytes on CUDA: {}", bytes, cudaGetErrorString(err));
                    result.initialized_ = false;
                    return result;
                }
                result.data_ = ptr;
                result.data_owner_ = std::shared_ptr<void>(ptr, [](void* p) {
                    if (p)
                        cudaFree(p);
                });
            } else {
                void* ptr = std::malloc(bytes);
                if (!ptr) {
                    LOG_ERROR("Failed to allocate {} bytes on CPU", bytes);
                    result.initialized_ = false;
                    return result;
                }
                result.data_ = ptr;
                result.data_owner_ = std::shared_ptr<void>(ptr, [](void* p) {
                    if (p)
                        std::free(p);
                });
            }
            break;
        }

        case LoadOp::Const: {
            float value = std::get<float>(args.args);
            result = load(LoadOp::Empty, args);
            if (!result.is_valid() || result.numel() == 0)
                return result;

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
                            nullptr);
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
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                if (result.dtype_ == DataType::Float32) {
                    tensor_ops::launch_uniform(result.ptr<float>(), result.numel(), low, high,
                                               RandomGenerator::instance().get_next_cuda_seed(), 0);
                    cudaDeviceSynchronize();
                } else if (result.dtype_ == DataType::Int32) {
                    tensor_ops::launch_randint(result.ptr<int>(), result.numel(),
                                               static_cast<int>(low), static_cast<int>(high),
                                               RandomGenerator::instance().get_next_cuda_seed(), 0);
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
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                tensor_ops::launch_normal(result.ptr<float>(), result.numel(), mean, std,
                                          RandomGenerator::instance().get_next_cuda_seed(), 0);
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
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                if (result.dtype_ == DataType::Int32) {
                    tensor_ops::launch_randint(result.ptr<int>(), result.numel(), low, high,
                                               RandomGenerator::instance().get_next_cuda_seed(), 0);
                    cudaDeviceSynchronize();
                } else if (result.dtype_ == DataType::Float32) {
                    int* temp_buffer;
                    cudaMalloc(&temp_buffer, result.numel() * sizeof(int));

                    tensor_ops::launch_randint(temp_buffer, result.numel(), low, high,
                                               RandomGenerator::instance().get_next_cuda_seed(), 0);

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
            if (!result.is_valid() || result.numel() == 0)
                return result;

            if (result.device_ == Device::CUDA) {
                tensor_ops::launch_bernoulli(result.ptr<float>(), result.numel(), p,
                                             RandomGenerator::instance().get_next_cuda_seed(), 0);
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
            if (!result.is_valid())
                return result;

            if (weights->device() == Device::CUDA) {
                tensor_ops::launch_multinomial(weights->ptr<float>(), result.ptr<int>(),
                                               n, num_samples, replacement,
                                               RandomGenerator::instance().get_next_cuda_seed(), 0);
                cudaDeviceSynchronize();
            } else {
                auto weights_data = weights->to_vector();

                float sum = std::accumulate(weights_data.begin(), weights_data.end(), 0.0f);
                if (sum <= 0) {
                    LOG_ERROR("Weights must sum to positive value");
                    return Tensor();
                }

                std::vector<float> cdf(n);
                cdf[0] = weights_data[0] / sum;
                for (size_t i = 1; i < n; ++i) {
                    cdf[i] = cdf[i - 1] + weights_data[i] / sum;
                }

                auto& gen = *static_cast<std::mt19937_64*>(
                    RandomGenerator::instance().get_generator(Device::CPU));
                std::uniform_real_distribution<float> dis(0.0f, 1.0f);

                int* samples = result.ptr<int>();

                if (replacement) {
                    for (size_t i = 0; i < num_samples; ++i) {
                        float u = dis(gen);
                        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
                        samples[i] = static_cast<int>(std::distance(cdf.begin(), it));
                    }
                } else {
                    std::vector<std::pair<float, int>> keys(n);

                    for (size_t i = 0; i < n; ++i) {
                        float u = dis(gen);
                        u = std::clamp(u, 1e-10f, 1.0f - 1e-10f);
                        float gumbel = -std::log(-std::log(u));
                        float log_weight = std::log(std::max(weights_data[i], 1e-10f));
                        keys[i] = {log_weight + gumbel, static_cast<int>(i)};
                    }

                    std::sort(keys.begin(), keys.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

                    for (size_t i = 0; i < num_samples; ++i) {
                        samples[i] = keys[i].second;
                    }
                }
            }
            break;
        }

        case LoadOp::Eye: {
            result = load(LoadOp::Const, {args.shape, args.device, args.dtype, 0.0f});
            if (!result.is_valid() || args.shape.rank() != 2)
                return result;

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

    Tensor Tensor::multinomial(const Tensor& weights, int num_samples, bool replacement) {
        if (!replacement && static_cast<size_t>(num_samples) > weights.numel()) {
            num_samples = static_cast<int>(weights.numel());
        }

        LoadArgs args;
        args.shape = TensorShape({static_cast<size_t>(num_samples)});
        args.device = weights.device();
        args.dtype = DataType::Int32;
        args.args = std::pair<void*, bool>{const_cast<void*>(static_cast<const void*>(&weights)), replacement};
        return load(LoadOp::Multinomial, args);
    }

    Tensor Tensor::unary(UnaryOp op, const UnaryArgs& args) const {
        if (!validate_unary_op()) {
            return Tensor();
        }

        auto result = Tensor::empty(shape_, device_,
                                    (op >= UnaryOp::IsNan && op <= UnaryOp::LogicalNot)
                                        ? DataType::Bool
                                        : dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_unary_op(
                data_, result.data_,
                numel(), op,
                dtype_, nullptr);
            cudaDeviceSynchronize();
        } else {
            // CPU implementation using compile-time operation table
            const float* src = static_cast<const float*>(data_);
            float* dst = static_cast<float*>(result.data_);

            // Use compile-time op table - single line!
            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = op_tables::float_unary_ops[int(op)](src[i]);
            }
        }

        return result;
    }

    // Binary operations - NO TEMPLATES!
     Tensor Tensor::binary_op_impl(const Tensor& other, BinaryOp op) const {
        if (!validate_binary_op(other, false, true)) {
            return Tensor();
        }

        auto broadcast_shape = this->broadcast_shape(other.shape());
        if (broadcast_shape.rank() == 0) {
            LOG_ERROR("Incompatible shapes for broadcasting: {} vs {}",
                      shape_.str(), other.shape_.str());
            return Tensor();
        }

        // Determine output dtype
        DataType out_dtype = dtype_;
        if (op >= BinaryOp::Equal && op <= BinaryOp::GreaterEqual) {
            out_dtype = DataType::Bool;
        } else if (op >= BinaryOp::LogicalAnd && op <= BinaryOp::LogicalXor) {
            out_dtype = DataType::Bool;
        } else if (op >= BinaryOp::BitwiseAnd && op <= BinaryOp::BitwiseXor) {
            out_dtype = DataType::Bool;
        } else {
            out_dtype = promote_types(dtype_, other.dtype());
        }

        auto result = Tensor::empty(broadcast_shape, device_, out_dtype);

        bool a_needs_broadcast = (shape_ != broadcast_shape);
        bool b_needs_broadcast = (other.shape() != broadcast_shape);

        if (!a_needs_broadcast && !b_needs_broadcast) {
            if (device_ == Device::CUDA) {
                tensor_ops::launch_binary_op(
                    raw_ptr(), other.raw_ptr(), result.raw_ptr(),
                    result.numel(), op,
                    dtype_, other.dtype(), result.dtype(),
                    nullptr);
                cudaDeviceSynchronize();
            } else {
                // CPU implementation using compile-time operation tables

                // Int32 comparison operations
                if (out_dtype == DataType::Bool && dtype_ == DataType::Int32 && other.dtype() == DataType::Int32) {
                    const int* src_a = static_cast<const int*>(raw_ptr());
                    const int* src_b = static_cast<const int*>(other.raw_ptr());
                    unsigned char* dst = static_cast<unsigned char*>(result.raw_ptr());

                    for (size_t i = 0; i < result.numel(); ++i) {
                        bool cmp_result = false;
                        switch (op) {
                            case BinaryOp::Equal: cmp_result = (src_a[i] == src_b[i]); break;
                            case BinaryOp::NotEqual: cmp_result = (src_a[i] != src_b[i]); break;
                            case BinaryOp::Less: cmp_result = (src_a[i] < src_b[i]); break;
                            case BinaryOp::LessEqual: cmp_result = (src_a[i] <= src_b[i]); break;
                            case BinaryOp::Greater: cmp_result = (src_a[i] > src_b[i]); break;
                            case BinaryOp::GreaterEqual: cmp_result = (src_a[i] >= src_b[i]); break;
                            default: cmp_result = false; break;
                        }
                        dst[i] = cmp_result ? 1 : 0;
                    }
                }
                // Float32 comparison operations
                else if (out_dtype == DataType::Bool && dtype_ == DataType::Float32 && other.dtype() == DataType::Float32) {
                    const float* src_a = static_cast<const float*>(raw_ptr());
                    const float* src_b = static_cast<const float*>(other.raw_ptr());
                    unsigned char* dst = static_cast<unsigned char*>(result.raw_ptr());

                    for (size_t i = 0; i < result.numel(); ++i) {
                        dst[i] = op_tables::float_compare_ops[int(op)](src_a[i], src_b[i]) ? 1 : 0;
                    }
                }
                // Logical operations
                else if (dtype_ == DataType::Bool && other.dtype() == DataType::Bool && out_dtype == DataType::Bool) {
                    const unsigned char* src_a = static_cast<const unsigned char*>(raw_ptr());
                    const unsigned char* src_b = static_cast<const unsigned char*>(other.raw_ptr());
                    unsigned char* dst = static_cast<unsigned char*>(result.raw_ptr());

                    for (size_t i = 0; i < result.numel(); ++i) {
                        dst[i] = op_tables::float_logical_ops[int(op)](src_a[i], src_b[i]) ? 1 : 0;
                    }
                }
                // Arithmetic operations
                else if (dtype_ == DataType::Float32 && other.dtype() == DataType::Float32 && out_dtype == DataType::Float32) {
                    const float* src_a = static_cast<const float*>(raw_ptr());
                    const float* src_b = static_cast<const float*>(other.raw_ptr());
                    float* dst = static_cast<float*>(result.raw_ptr());

                    for (size_t i = 0; i < result.numel(); ++i) {
                        dst[i] = op_tables::float_binary_ops[int(op)](src_a[i], src_b[i]);
                    }
                }
                // Int32 arithmetic operations
                else if (dtype_ == DataType::Int32 && other.dtype() == DataType::Int32 && out_dtype == DataType::Int32) {
                    const int* src_a = static_cast<const int*>(raw_ptr());
                    const int* src_b = static_cast<const int*>(other.raw_ptr());
                    int* dst = static_cast<int*>(result.raw_ptr());

                    for (size_t i = 0; i < result.numel(); ++i) {
                        dst[i] = op_tables::int_binary_ops[int(op)](src_a[i], src_b[i]);
                    }
                }
            }
        } else {
            // Broadcasting needed
            if (device_ == Device::CUDA) {
                if (a_needs_broadcast) {
                    auto a_broadcast = broadcast_to(broadcast_shape);
                    if (b_needs_broadcast) {
                        auto b_broadcast = other.broadcast_to(broadcast_shape);
                        tensor_ops::launch_binary_op(
                            a_broadcast.raw_ptr(), b_broadcast.raw_ptr(), result.raw_ptr(),
                            result.numel(), op,
                            a_broadcast.dtype(), b_broadcast.dtype(), result.dtype(),
                            nullptr);
                    } else {
                        tensor_ops::launch_binary_op(
                            a_broadcast.raw_ptr(), other.raw_ptr(), result.raw_ptr(),
                            result.numel(), op,
                            a_broadcast.dtype(), other.dtype(), result.dtype(),
                            nullptr);
                    }
                } else {
                    auto b_broadcast = other.broadcast_to(broadcast_shape);
                    tensor_ops::launch_binary_op(
                        raw_ptr(), b_broadcast.raw_ptr(), result.raw_ptr(),
                        result.numel(), op,
                        dtype_, b_broadcast.dtype(), result.dtype(),
                        nullptr);
                }
                cudaDeviceSynchronize();
            } else {
                // CPU implementation with broadcasting
                auto a_shape = shape_.dims();
                auto b_shape = other.shape().dims();
                auto c_shape = broadcast_shape.dims();

                // Int32 comparison with broadcasting
                if (out_dtype == DataType::Bool && dtype_ == DataType::Int32 && other.dtype() == DataType::Int32) {
                    const int* src_a = static_cast<const int*>(raw_ptr());
                    const int* src_b = static_cast<const int*>(other.raw_ptr());
                    unsigned char* dst = static_cast<unsigned char*>(result.raw_ptr());

                    for (size_t i = 0; i < result.numel(); ++i) {
                        size_t a_idx = broadcast::index(i, c_shape, a_shape);
                        size_t b_idx = broadcast::index(i, c_shape, b_shape);

                        bool cmp_result = false;
                        switch (op) {
                            case BinaryOp::Equal: cmp_result = (src_a[a_idx] == src_b[b_idx]); break;
                            case BinaryOp::NotEqual: cmp_result = (src_a[a_idx] != src_b[b_idx]); break;
                            case BinaryOp::Less: cmp_result = (src_a[a_idx] < src_b[b_idx]); break;
                            case BinaryOp::LessEqual: cmp_result = (src_a[a_idx] <= src_b[b_idx]); break;
                            case BinaryOp::Greater: cmp_result = (src_a[a_idx] > src_b[b_idx]); break;
                            case BinaryOp::GreaterEqual: cmp_result = (src_a[a_idx] >= src_b[b_idx]); break;
                            default: cmp_result = false; break;
                        }
                        dst[i] = cmp_result ? 1 : 0;
                    }
                }
                // Float32 comparison with broadcasting
                else if (out_dtype == DataType::Bool && dtype_ == DataType::Float32 && other.dtype() == DataType::Float32) {
                    const float* src_a = static_cast<const float*>(raw_ptr());
                    const float* src_b = static_cast<const float*>(other.raw_ptr());
                    unsigned char* dst = static_cast<unsigned char*>(result.raw_ptr());

                    for (size_t i = 0; i < result.numel(); ++i) {
                        size_t a_idx = broadcast::index(i, c_shape, a_shape);
                        size_t b_idx = broadcast::index(i, c_shape, b_shape);
                        dst[i] = op_tables::float_compare_ops[int(op)](src_a[a_idx], src_b[b_idx]) ? 1 : 0;
                    }
                }
                // Bool logical with broadcasting
                else if (dtype_ == DataType::Bool && other.dtype() == DataType::Bool && out_dtype == DataType::Bool) {
                    const unsigned char* src_a = static_cast<const unsigned char*>(raw_ptr());
                    const unsigned char* src_b = static_cast<const unsigned char*>(other.raw_ptr());
                    unsigned char* dst = static_cast<unsigned char*>(result.raw_ptr());

                    for (size_t i = 0; i < result.numel(); ++i) {
                        size_t a_idx = broadcast::index(i, c_shape, a_shape);
                        size_t b_idx = broadcast::index(i, c_shape, b_shape);
                        dst[i] = op_tables::float_logical_ops[int(op)](src_a[a_idx], src_b[b_idx]) ? 1 : 0;
                    }
                }
                // Float32 arithmetic with broadcasting
                else if (dtype_ == DataType::Float32 && other.dtype() == DataType::Float32 && out_dtype == DataType::Float32) {
                    const float* src_a = static_cast<const float*>(raw_ptr());
                    const float* src_b = static_cast<const float*>(other.raw_ptr());
                    float* dst = static_cast<float*>(result.raw_ptr());

                    for (size_t i = 0; i < result.numel(); ++i) {
                        size_t a_idx = broadcast::index(i, c_shape, a_shape);
                        size_t b_idx = broadcast::index(i, c_shape, b_shape);
                        dst[i] = op_tables::float_binary_ops[int(op)](src_a[a_idx], src_b[b_idx]);
                    }
                }
            }
        }

        return result;
    }


     Tensor Tensor::binary_op_scalar(float scalar, BinaryOp op) const {
        if (!validate_unary_op()) {
            return Tensor();
        }

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
                nullptr);
            cudaDeviceSynchronize();
        } else {
            // CPU implementation

            // Int32 comparison with scalar
            if (out_dtype == DataType::Bool && dtype_ == DataType::Int32) {
                const int* src = static_cast<const int*>(raw_ptr());
                unsigned char* dst = static_cast<unsigned char*>(result.raw_ptr());
                int scalar_int = static_cast<int>(scalar);

                for (size_t i = 0; i < numel(); ++i) {
                    bool cmp_result = false;
                    switch (op) {
                        case BinaryOp::Equal: cmp_result = (src[i] == scalar_int); break;
                        case BinaryOp::NotEqual: cmp_result = (src[i] != scalar_int); break;
                        case BinaryOp::Less: cmp_result = (src[i] < scalar_int); break;
                        case BinaryOp::LessEqual: cmp_result = (src[i] <= scalar_int); break;
                        case BinaryOp::Greater: cmp_result = (src[i] > scalar_int); break;
                        case BinaryOp::GreaterEqual: cmp_result = (src[i] >= scalar_int); break;
                        default: cmp_result = false; break;
                    }
                    dst[i] = cmp_result ? 1 : 0;
                }
            }
            // Float32 comparison with scalar
            else if (out_dtype == DataType::Bool && dtype_ == DataType::Float32) {
                const float* src = static_cast<const float*>(raw_ptr());
                unsigned char* dst = static_cast<unsigned char*>(result.raw_ptr());

                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = op_tables::float_compare_ops[int(op)](src[i], scalar) ? 1 : 0;
                }
            }
            // Float32 arithmetic with scalar
            else if (dtype_ == DataType::Float32 && out_dtype == DataType::Float32) {
                const float* src = static_cast<const float*>(raw_ptr());
                float* dst = static_cast<float*>(result.raw_ptr());

                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = op_tables::float_binary_ops[int(op)](src[i], scalar);
                }
            }
            // Int32 arithmetic with scalar
            else if (dtype_ == DataType::Int32 && out_dtype == DataType::Int32) {
                const int* src = static_cast<const int*>(raw_ptr());
                int* dst = static_cast<int*>(result.raw_ptr());
                int scalar_int = static_cast<int>(scalar);

                for (size_t i = 0; i < numel(); ++i) {
                    dst[i] = op_tables::int_binary_ops[int(op)](src[i], scalar_int);
                }
            }
        }

        return result;
    }

    Tensor& Tensor::binary_op_inplace_impl(const Tensor& other, BinaryOp op) {
        if (!validate_binary_op(other, true)) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_binary_op_inplace(
                raw_ptr(),
                other.raw_ptr(),
                numel(),
                op,
                nullptr);
            cudaDeviceSynchronize();
        } else {
            // CPU implementation using compile-time tables
            float* dst = static_cast<float*>(raw_ptr());
            const float* src = static_cast<const float*>(other.raw_ptr());

            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = op_tables::float_binary_ops[int(op)](dst[i], src[i]);
            }
        }

        return *this;
    }

    Tensor& Tensor::binary_op_inplace_scalar(float scalar, BinaryOp op) {
        if (!validate_unary_op()) {
            return *this;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_binary_scalar_inplace(
                raw_ptr(),
                scalar,
                numel(),
                op,
                nullptr);
            cudaDeviceSynchronize();
        } else {
            // CPU implementation using compile-time tables
            float* dst = static_cast<float*>(raw_ptr());

            for (size_t i = 0; i < numel(); ++i) {
                dst[i] = op_tables::float_binary_ops[int(op)](dst[i], scalar);
            }
        }

        return *this;
    }

    Tensor Tensor::reduce(ReduceOp op, const ReduceArgs& args) const {
        if (!validate_unary_op()) {
            return Tensor();
        }

        // Special handling for Std and Var
        if (op == ReduceOp::Std || op == ReduceOp::Var) {
            ReduceArgs mean_args = args;
            auto mean_tensor = reduce(ReduceOp::Mean, mean_args);

            Tensor mean_broadcast = (mean_tensor.shape() == shape_)
                                        ? mean_tensor.clone()
                                        : mean_tensor.broadcast_to(shape_);

            auto diff = this->sub(mean_broadcast);
            auto squared = diff.mul(diff);

            // Compute sum of squared differences
            auto sum_sq = squared.reduce(ReduceOp::Sum, mean_args);

            // Calculate N (number of elements being reduced)
            std::vector<int> axes = args.axes;
            if (axes.empty()) {
                axes.resize(shape_.rank());
                std::iota(axes.begin(), axes.end(), 0);
            }

            size_t reduce_count = 1;
            for (int ax : axes) {
                reduce_count *= shape_[ax];
            }

            // For dimensional reductions (with keepdim or specific axes), use unbiased (N-1)
            // For scalar reductions (all axes), use biased (N)
            bool is_scalar_reduction = (axes.size() == shape_.rank() && !args.keepdim);
            float correction = static_cast<float>(reduce_count);
            if (!is_scalar_reduction && reduce_count > 1) {
                // Use unbiased estimator (N-1) for dimensional reductions
                correction = static_cast<float>(reduce_count - 1);
            }

            auto variance = sum_sq.div(correction);

            if (op == ReduceOp::Var) {
                return variance;
            } else {
                return variance.sqrt();
            }
        }

        std::vector<int> axes = args.axes;
        if (axes.empty()) {
            axes.resize(shape_.rank());
            std::iota(axes.begin(), axes.end(), 0);
        }

        std::vector<size_t> out_shape;
        for (size_t i = 0; i < shape_.rank(); ++i) {
            bool is_reduced = std::find(axes.begin(), axes.end(), i) != axes.end();
            if (!is_reduced || args.keepdim) {
                out_shape.push_back(is_reduced ? 1 : shape_[i]);
            }
        }

        DataType out_dtype = dtype_;
        if (op == ReduceOp::Any || op == ReduceOp::All) {
            out_dtype = DataType::Bool;
        } else if (op == ReduceOp::Argmax || op == ReduceOp::Argmin) {
            out_dtype = DataType::Int64;
        }

        auto result = Tensor::empty(TensorShape(out_shape), device_, out_dtype);

        if (numel() == 0) {
            float identity_value = 0.0f;
            switch (op) {
            case ReduceOp::Sum:
            case ReduceOp::Mean:
                identity_value = 0.0f;
                break;
            case ReduceOp::Prod:
                identity_value = 1.0f;
                break;
            case ReduceOp::Max:
                identity_value = -std::numeric_limits<float>::infinity();
                break;
            case ReduceOp::Min:
                identity_value = std::numeric_limits<float>::infinity();
                break;
            case ReduceOp::Any:
                identity_value = 0.0f;
                break;
            case ReduceOp::All:
                identity_value = 1.0f;
                break;
            default:
                identity_value = 0.0f;
                break;
            }

            if (device_ == Device::CUDA) {
                if (out_dtype == DataType::Float32) {
                    std::vector<float> temp(result.numel(), identity_value);
                    cudaMemcpy(result.raw_ptr(), temp.data(),
                               result.bytes(), cudaMemcpyHostToDevice);
                } else if (out_dtype == DataType::Bool) {
                    unsigned char bool_val = (identity_value != 0.0f) ? 1 : 0;
                    cudaMemset(result.raw_ptr(), bool_val, result.bytes());
                }
            } else {
                if (out_dtype == DataType::Float32) {
                    float* ptr = static_cast<float*>(result.raw_ptr());
                    std::fill_n(ptr, result.numel(), identity_value);
                } else if (out_dtype == DataType::Bool) {
                    unsigned char* ptr = static_cast<unsigned char*>(result.raw_ptr());
                    std::fill_n(ptr, result.numel(), identity_value != 0.0f ? 1 : 0);
                }
            }
            return result;
        }

        if (device_ == Device::CUDA) {
            tensor_ops::launch_reduce_op(
                raw_ptr(), result.raw_ptr(),
                shape_.dims().data(), shape_.rank(),
                axes.data(), axes.size(),
                args.keepdim, op,
                dtype_, nullptr);
            cudaDeviceSynchronize();
        } else {
            // CPU implementation
            const float* src = static_cast<const float*>(raw_ptr());
            float* dst = static_cast<float*>(result.raw_ptr());

            // Full reduction to scalar
            if (axes.size() == shape_.rank()) {
                if (op == ReduceOp::Sum) {
                    float sum = 0;
                    for (size_t i = 0; i < numel(); ++i) {
                        sum += src[i];
                    }
                    dst[0] = sum;
                } else if (op == ReduceOp::Mean) {
                    float sum = 0;
                    for (size_t i = 0; i < numel(); ++i) {
                        sum += src[i];
                    }
                    dst[0] = sum / numel();
                } else if (op == ReduceOp::Max) {
                    float max_val = src[0];
                    for (size_t i = 1; i < numel(); ++i) {
                        max_val = std::max(max_val, src[i]);
                    }
                    dst[0] = max_val;
                } else if (op == ReduceOp::Min) {
                    float min_val = src[0];
                    for (size_t i = 1; i < numel(); ++i) {
                        min_val = std::min(min_val, src[i]);
                    }
                    dst[0] = min_val;
                } else if (op == ReduceOp::Prod) {
                    float prod = 1.0f;
                    for (size_t i = 0; i < numel(); ++i) {
                        prod *= src[i];
                    }
                    dst[0] = prod;
                }
                return result;
            }

            // Axis-specific reduction - general implementation
            // Build mask of which dimensions are reduced
            std::vector<bool> is_reduced_dim(shape_.rank(), false);
            for (int ax : axes) {
                is_reduced_dim[ax] = true;
            }

            // Calculate input strides
            auto input_strides = shape_.strides();

            // Calculate output strides
            std::vector<size_t> out_shape_vec;
            for (size_t i = 0; i < shape_.rank(); ++i) {
                if (!is_reduced_dim[i]) {
                    out_shape_vec.push_back(shape_[i]);
                }
            }

            std::vector<size_t> output_strides;
            if (!out_shape_vec.empty()) {
                output_strides.resize(out_shape_vec.size());
                output_strides.back() = 1;
                for (int i = static_cast<int>(out_shape_vec.size()) - 2; i >= 0; --i) {
                    output_strides[i] = output_strides[i + 1] * out_shape_vec[i + 1];
                }
            }

            size_t output_elements = result.numel();

            // Calculate how many elements to reduce per output element
            size_t reduce_count = 1;
            std::vector<size_t> reduced_dims;
            for (size_t i = 0; i < shape_.rank(); ++i) {
                if (is_reduced_dim[i]) {
                    reduced_dims.push_back(i);
                    reduce_count *= shape_[i];
                }
            }

            // Perform reduction
            for (size_t out_idx = 0; out_idx < output_elements; ++out_idx) {
                // Convert output linear index to coordinates in output space
                std::vector<size_t> out_coords;
                if (!out_shape_vec.empty()) {
                    out_coords.resize(out_shape_vec.size());
                    size_t temp = out_idx;
                    for (size_t i = 0; i < out_shape_vec.size(); ++i) {
                        out_coords[i] = temp / output_strides[i];
                        temp %= output_strides[i];
                    }
                }

                // Map output coords back to base input coords
                std::vector<size_t> base_input_coords(shape_.rank());
                size_t out_coord_idx = 0;
                for (size_t i = 0; i < shape_.rank(); ++i) {
                    if (!is_reduced_dim[i]) {
                        base_input_coords[i] = out_coords[out_coord_idx++];
                    } else {
                        base_input_coords[i] = 0;
                    }
                }

                // Initialize result with identity value for this output element
                float result_val = 0.0f;
                if (op == ReduceOp::Max) {
                    result_val = -std::numeric_limits<float>::infinity();
                } else if (op == ReduceOp::Min) {
                    result_val = std::numeric_limits<float>::infinity();
                } else if (op == ReduceOp::Prod) {
                    result_val = 1.0f;
                }

                // Iterate through all combinations of reduced dimensions
                for (size_t r = 0; r < reduce_count; ++r) {
                    // Compute coordinates in the reduced dimensions
                    size_t temp_r = r;
                    std::vector<size_t> full_input_coords = base_input_coords;

                    // Fill in reduced dimensions - work backwards for row-major order
                    for (int rd_idx = static_cast<int>(reduced_dims.size()) - 1; rd_idx >= 0; --rd_idx) {
                        size_t dim = reduced_dims[rd_idx];
                        full_input_coords[dim] = temp_r % shape_[dim];
                        temp_r /= shape_[dim];
                    }

                    // Calculate linear input index
                    size_t in_idx = 0;
                    for (size_t i = 0; i < shape_.rank(); ++i) {
                        in_idx += full_input_coords[i] * input_strides[i];
                    }

                    // Apply reduction operation
                    float val = src[in_idx];
                    switch (op) {
                        case ReduceOp::Sum:
                        case ReduceOp::Mean:  // Mean accumulates like sum, then divides at end
                            result_val += val;
                            break;
                        case ReduceOp::Max:
                            result_val = std::max(result_val, val);
                            break;
                        case ReduceOp::Min:
                            result_val = std::min(result_val, val);
                            break;
                        case ReduceOp::Prod:
                            result_val *= val;
                            break;
                        default:
                            break;
                    }
                }

                // Store result (apply mean if needed)
                if (op == ReduceOp::Mean) {
                    dst[out_idx] = result_val / reduce_count;
                } else {
                    dst[out_idx] = result_val;
                }
            }
        }

        return result;
    }

    // ============= TERNARY OPERATIONS =============

    Tensor Tensor::ternary(const Tensor& b, const Tensor& c, TernaryOp op) const {
        if (!validate_ternary_op(b, c)) {
            return Tensor();
        }

        if (numel() == 0 || b.numel() == 0 || c.numel() == 0) {
            auto shape_ab = this->broadcast_shape(b.shape());
            if (shape_ab.rank() == 0) {
                LOG_ERROR("Incompatible shapes for first two tensors in ternary operation with empty tensors");
                return Tensor();
            }

            auto shape_abc_vec = broadcast::shape(shape_ab.dims(), c.shape().dims());
            if (shape_abc_vec.empty()) {
                LOG_ERROR("Incompatible shapes for ternary operation");
                return Tensor();
            }

            DataType out_dtype = dtype_;
            if (op == TernaryOp::Where) {
                out_dtype = promote_types(b.dtype(), c.dtype());
            }

            return empty(TensorShape(shape_abc_vec), device_, out_dtype);
        }

        if (op == TernaryOp::Where && dtype_ != DataType::Bool) {
            LOG_ERROR("Where operation requires boolean condition tensor");
            return Tensor();
        }

        auto shape_ab = this->broadcast_shape(b.shape());
        if (shape_ab.rank() == 0) {
            LOG_ERROR("Incompatible shapes for first two tensors in ternary operation");
            return Tensor();
        }

        auto shape_abc_vec = broadcast::shape(shape_ab.dims(), c.shape().dims());
        if (shape_abc_vec.empty()) {
            LOG_ERROR("Incompatible shapes for ternary operation");
            return Tensor();
        }

        TensorShape shape_abc(shape_abc_vec);

        DataType out_dtype = dtype_;
        if (op == TernaryOp::Where) {
            out_dtype = promote_types(b.dtype(), c.dtype());
        }

        auto result = Tensor::empty(shape_abc, device_, out_dtype);

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
            if (op == TernaryOp::Where) {
                tensor_ops::launch_where(
                    a_broadcast.ptr<unsigned char>(),
                    b_broadcast.ptr<float>(),
                    c_broadcast.ptr<float>(),
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
                    0);
            } else {
                tensor_ops::launch_ternary_op(
                    a_broadcast.raw_ptr(),
                    b_broadcast.raw_ptr(),
                    c_broadcast.raw_ptr(),
                    result.raw_ptr(),
                    result.numel(),
                    op,
                    result.dtype(),
                    nullptr);
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
                    if (std::isnan(src[i])) {
                        dst[i] = src[i];
                    } else {
                        dst[i] = std::max(min_vals[i], std::min(max_vals[i], src[i]));
                    }
                }
            }
        }

        return result;
    }

    Tensor Tensor::where(const Tensor& condition, const Tensor& x, const Tensor& y) {
        if (!condition.is_valid() || !x.is_valid() || !y.is_valid()) {
            LOG_ERROR("where: invalid input tensors");
            return Tensor();
        }

        if (condition.dtype() != DataType::Bool) {
            LOG_ERROR("where: condition must be boolean tensor");
            return Tensor();
        }

        // Check device compatibility
        if (condition.device() != x.device() || x.device() != y.device()) {
            LOG_ERROR("where: all tensors must be on the same device");
            return Tensor();
        }
        return condition.ternary(x, y, TernaryOp::Where);
    }

    float Tensor::norm(float p) const {
        if (!is_valid())
            return 0.0f;

        if (p == 2.0f) {
            auto squared = this->mul(*this);
            return std::sqrt(squared.sum_scalar());
        } else if (p == 1.0f) {
            return this->abs().sum_scalar();
        } else if (std::isinf(p)) {
            return this->abs().max_scalar();
        } else {
            auto abs_vals = this->abs();
            auto powered = abs_vals.pow(p);
            auto sum = powered.sum_scalar();
            return std::pow(sum, 1.0f / p);
        }
    }

    Tensor Tensor::norm(float p, std::span<const int> dims, bool keepdim) const {
        if (!is_valid()) {
            LOG_ERROR("norm() on invalid tensor");
            return Tensor();
        }

        if (numel() == 0) {
            // Return appropriate empty tensor
            std::vector<size_t> out_shape;
            for (size_t i = 0; i < shape_.rank(); ++i) {
                bool is_reduced = false;
                for (int d : dims) {
                    if (resolve_dim(d) == static_cast<int>(i)) {
                        is_reduced = true;
                        break;
                    }
                }
                if (!is_reduced || keepdim) {
                    out_shape.push_back(is_reduced ? 1 : shape_[i]);
                }
            }
            return empty(TensorShape(out_shape), device_, dtype_);
        }

        // Special cases for common norms
        if (p == 2.0f) {
            // L2 norm: sqrt(sum(x^2))
            auto squared = this->mul(*this);
            auto sum = squared.sum(dims, keepdim);
            return sum.sqrt();
        } else if (p == 1.0f) {
            // L1 norm: sum(|x|)
            return this->abs().sum(dims, keepdim);
        } else if (std::isinf(p)) {
            if (p > 0) {
                // L-infinity norm: max(|x|)
                return this->abs().max(dims, keepdim);
            } else {
                // L-negative-infinity norm: min(|x|)
                return this->abs().min(dims, keepdim);
            }
        } else if (p == 0.0f) {
            // L0 "norm": count of non-zero elements
            // This isn't a true norm, but often used
            auto nonzero_mask = this->ne(0.0f).to(DataType::Float32);
            return nonzero_mask.sum(dims, keepdim);
        } else {
            // General Lp norm: (sum(|x|^p))^(1/p)
            auto abs_vals = this->abs();
            auto powered = abs_vals.pow(p);
            auto sum = powered.sum(dims, keepdim);
            return sum.pow(1.0f / p);
        }
    }

    std::pair<Tensor, Tensor> Tensor::_broadcasted(const Tensor& other, bool match_dtype) const {
        if (!is_valid() || !other.is_valid()) {
            return {Tensor(), Tensor()};
        }

        auto bcast_shape = this->broadcast_shape(other.shape());
        if (bcast_shape.rank() == 0) {
            LOG_ERROR("Incompatible shapes for broadcasting");
            return {Tensor(), Tensor()};
        }

        Tensor a_broadcast = (shape_ == bcast_shape) ? this->clone() : broadcast_to(bcast_shape);
        Tensor b_broadcast = (other.shape() == bcast_shape) ? other.clone() : other.broadcast_to(bcast_shape);

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

    // ============= STATIC CAT OPERATION =============

    Tensor Tensor::cat(const std::vector<Tensor>& tensors, int dim) {
        if (tensors.empty()) {
            LOG_ERROR("Cannot concatenate empty vector of tensors");
            return Tensor();
        }

        if (tensors.size() == 1) {
            return tensors[0].clone();
        }

        int resolved_dim = dim;
        if (resolved_dim < 0) {
            resolved_dim = tensors[0].shape().rank() + resolved_dim;
        }

        if (resolved_dim < 0 || resolved_dim >= static_cast<int>(tensors[0].shape().rank())) {
            LOG_ERROR("Invalid dimension for cat: {}", dim);
            return Tensor();
        }

        const auto& first_shape = tensors[0].shape();
        const auto first_device = tensors[0].device();
        const auto first_dtype = tensors[0].dtype();

        size_t total_size_along_dim = first_shape[resolved_dim];

        for (size_t i = 1; i < tensors.size(); ++i) {
            const auto& shape = tensors[i].shape();

            if (shape.rank() != first_shape.rank()) {
                LOG_ERROR("All tensors must have the same number of dimensions");
                return Tensor();
            }

            for (size_t d = 0; d < shape.rank(); ++d) {
                if (d != static_cast<size_t>(resolved_dim) && shape[d] != first_shape[d]) {
                    LOG_ERROR("All dimensions except dim={} must match", dim);
                    return Tensor();
                }
            }

            if (tensors[i].device() != first_device) {
                LOG_ERROR("All tensors must be on the same device");
                return Tensor();
            }

            if (tensors[i].dtype() != first_dtype) {
                LOG_ERROR("All tensors must have the same dtype");
                return Tensor();
            }

            total_size_along_dim += shape[resolved_dim];
        }

        std::vector<size_t> result_dims = first_shape.dims();
        result_dims[resolved_dim] = total_size_along_dim;

        auto result = Tensor::empty(TensorShape(result_dims), first_device, first_dtype);

        size_t outer_size = 1;
        for (int i = 0; i < resolved_dim; ++i) {
            outer_size *= first_shape[i];
        }

        size_t inner_size = 1;
        for (size_t i = resolved_dim + 1; i < first_shape.rank(); ++i) {
            inner_size *= first_shape[i];
        }

        size_t element_size = dtype_size(first_dtype);

        if (first_device == Device::CUDA) {
            for (size_t outer = 0; outer < outer_size; ++outer) {
                size_t result_offset = outer * total_size_along_dim * inner_size;

                for (const auto& t : tensors) {
                    size_t tensor_dim_size = t.shape()[resolved_dim];
                    size_t src_offset = outer * tensor_dim_size * inner_size;
                    size_t copy_size = tensor_dim_size * inner_size * element_size;

                    void* dst = static_cast<char*>(result.raw_ptr()) + (result_offset * element_size);
                    const void* src = static_cast<const char*>(t.raw_ptr()) + (src_offset * element_size);

                    cudaMemcpy(dst, src, copy_size, cudaMemcpyDeviceToDevice);

                    result_offset += tensor_dim_size * inner_size;
                }
            }
        } else {
            for (size_t outer = 0; outer < outer_size; ++outer) {
                size_t result_offset = outer * total_size_along_dim * inner_size;

                for (const auto& t : tensors) {
                    size_t tensor_dim_size = t.shape()[resolved_dim];
                    size_t src_offset = outer * tensor_dim_size * inner_size;
                    size_t copy_size = tensor_dim_size * inner_size * element_size;

                    void* dst = static_cast<char*>(result.raw_ptr()) + (result_offset * element_size);
                    const void* src = static_cast<const char*>(t.raw_ptr()) + (src_offset * element_size);

                    std::memcpy(dst, src, copy_size);

                    result_offset += tensor_dim_size * inner_size;
                }
            }
        }

        return result;
    }

    // ============= STATIC STACK OPERATION =============

    Tensor Tensor::stack(const std::vector<Tensor>& tensors, int dim) {
        if (tensors.empty()) {
            LOG_ERROR("Cannot stack empty vector of tensors");
            return Tensor();
        }

        const auto& first_shape = tensors[0].shape();
        const auto first_device = tensors[0].device();
        const auto first_dtype = tensors[0].dtype();

        for (size_t i = 1; i < tensors.size(); ++i) {
            if (tensors[i].shape() != first_shape) {
                LOG_ERROR("All tensors must have the same shape for stack");
                return Tensor();
            }
            if (tensors[i].device() != first_device) {
                LOG_ERROR("All tensors must be on the same device");
                return Tensor();
            }
            if (tensors[i].dtype() != first_dtype) {
                LOG_ERROR("All tensors must have the same dtype");
                return Tensor();
            }
        }

        std::vector<size_t> new_dims = first_shape.dims();

        if (dim < 0) {
            dim = first_shape.rank() + dim + 1;
        }

        if (dim < 0 || dim > static_cast<int>(first_shape.rank())) {
            LOG_ERROR("Invalid dimension for stack: {}", dim);
            return Tensor();
        }

        new_dims.insert(new_dims.begin() + dim, tensors.size());

        auto result = Tensor::empty(TensorShape(new_dims), first_device, first_dtype);

        size_t elements_per_tensor = first_shape.elements();
        size_t bytes_per_tensor = elements_per_tensor * dtype_size(first_dtype);

        if (dim == 0) {
            for (size_t i = 0; i < tensors.size(); ++i) {
                void* dst = static_cast<char*>(result.raw_ptr()) + i * bytes_per_tensor;
                if (first_device == Device::CUDA) {
                    cudaMemcpy(dst, tensors[i].raw_ptr(), bytes_per_tensor,
                               cudaMemcpyDeviceToDevice);
                } else {
                    std::memcpy(dst, tensors[i].raw_ptr(), bytes_per_tensor);
                }
            }
        } else {
            LOG_ERROR("Stack along dimension {} not fully implemented", dim);
            return Tensor();
        }

        return result;
    }

} // namespace gs