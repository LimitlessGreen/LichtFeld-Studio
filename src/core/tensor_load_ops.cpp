/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include <cstring>
#include <cuda_runtime.h>

namespace gs {

// ============= Unified Load Operation =============
Tensor Tensor::load(LoadOp op, const LoadArgs& args) {
    switch (op) {
        case LoadOp::Empty: {
            // Allocate uninitialized memory
            size_t n_bytes = args.shape.elements() * dtype_size(args.dtype);
            if (n_bytes == 0) {
                Tensor t;
                t.shape_ = args.shape;
                t.device_ = args.device;
                t.dtype_ = args.dtype;
                t.owns_memory_ = false;
                t.initialized_ = true;
                t.id_ = next_id_++;
                return t;
            }

            void* data = nullptr;
            if (args.device == Device::CUDA) {
                cudaError_t err = cudaMalloc(&data, n_bytes);
                if (err != cudaSuccess) {
                    LOG_ERROR("Failed to allocate {} bytes on CUDA: {}",
                             n_bytes, cudaGetErrorString(err));
                    return Tensor();
                }
            } else {
                data = new char[n_bytes];
                if (!data) {
                    LOG_ERROR("Failed to allocate {} bytes on CPU", n_bytes);
                    return Tensor();
                }
            }

            Tensor t;
            t.data_ = data;
            t.shape_ = args.shape;
            t.device_ = args.device;
            t.dtype_ = args.dtype;
            t.owns_memory_ = true;
            t.initialized_ = true;
            t.id_ = next_id_++;

            return t;
        }

        case LoadOp::Const: {
            // Fill with constant value
            auto t = load(LoadOp::Empty, args);
            if (!t.is_valid() || t.numel() == 0) return t;

            if (auto* val = std::get_if<float>(&args.args)) {
                if (args.device == Device::CUDA) {
                    if (args.dtype == DataType::Float32) {
                        // Use CUDA kernel for filling
                        if (*val == 0.0f) {
                            cudaMemset(t.data_, 0, t.bytes());
                        } else {
                            std::vector<float> temp(t.numel(), *val);
                            cudaMemcpy(t.data_, temp.data(), t.bytes(), cudaMemcpyHostToDevice);
                        }
                    } else if (args.dtype == DataType::Bool) {
                        unsigned char fill = (*val != 0) ? 1 : 0;
                        cudaMemset(t.data_, fill, t.bytes());
                    } else if (args.dtype == DataType::Int32) {
                        if (*val == 0.0f) {
                            cudaMemset(t.data_, 0, t.bytes());
                        } else {
                            std::vector<int> temp(t.numel(), static_cast<int>(*val));
                            cudaMemcpy(t.data_, temp.data(), t.bytes(), cudaMemcpyHostToDevice);
                        }
                    }
                } else {
                    if (args.dtype == DataType::Float32) {
                        float* data = t.ptr<float>();
                        std::fill(data, data + t.numel(), *val);
                    } else if (args.dtype == DataType::Bool) {
                        unsigned char* data = t.ptr<unsigned char>();
                        unsigned char fill = (*val != 0) ? 1 : 0;
                        std::fill(data, data + t.numel(), fill);
                    } else if (args.dtype == DataType::Int32) {
                        int* data = t.ptr<int>();
                        std::fill(data, data + t.numel(), static_cast<int>(*val));
                    }
                }
            }

            return t;
        }

        case LoadOp::Arange: {
            if (auto* range = std::get_if<std::tuple<float, float, float>>(&args.args)) {
                float start = std::get<0>(*range);
                float stop = std::get<1>(*range);
                float step = std::get<2>(*range);

                if (step == 0) {
                    LOG_ERROR("Step cannot be zero");
                    return Tensor();
                }

                if ((stop - start) * step < 0) {
                    LOG_ERROR("Invalid range: start={}, stop={}, step={}", start, stop, step);
                    return Tensor();
                }

                size_t n = static_cast<size_t>(std::ceil((stop - start) / step));
                LoadArgs new_args = args;
                new_args.shape = TensorShape({n});

                auto t = load(LoadOp::Empty, new_args);
                if (!t.is_valid()) return t;

                // Generate on CPU and copy
                if (args.dtype == DataType::Float32) {
                    std::vector<float> data(n);
                    for (size_t i = 0; i < n; ++i) {
                        data[i] = start + i * step;
                    }

                    if (args.device == Device::CUDA) {
                        cudaMemcpy(t.ptr<float>(), data.data(), n * sizeof(float),
                                  cudaMemcpyHostToDevice);
                    } else {
                        std::memcpy(t.ptr<float>(), data.data(), n * sizeof(float));
                    }
                } else if (args.dtype == DataType::Int32) {
                    std::vector<int> data(n);
                    for (size_t i = 0; i < n; ++i) {
                        data[i] = static_cast<int>(start + i * step);
                    }

                    if (args.device == Device::CUDA) {
                        cudaMemcpy(t.ptr<int>(), data.data(), n * sizeof(int),
                                  cudaMemcpyHostToDevice);
                    } else {
                        std::memcpy(t.ptr<int>(), data.data(), n * sizeof(int));
                    }
                }

                return t;
            }
            LOG_ERROR("Arange requires (start, stop, step) tuple");
            return {};
        }

        case LoadOp::Random: {
            if (auto* range = std::get_if<std::pair<float, float>>(&args.args)) {
                auto t = load(LoadOp::Empty, args);
                if (!t.is_valid() || t.numel() == 0) return t;

                float low = range->first;
                float high = range->second;

                if (args.device == Device::CUDA) {
                    if (args.dtype == DataType::Float32) {
                        tensor_ops::launch_uniform(t.ptr<float>(), t.numel(), low, high,
                                                 RandomGenerator::instance().get_seed(), 0);
                        cudaDeviceSynchronize();
                    } else if (args.dtype == DataType::Int32) {
                        tensor_ops::launch_randint(t.ptr<int>(), t.numel(),
                                                  static_cast<int>(low), static_cast<int>(high),
                                                  RandomGenerator::instance().get_seed(), 0);
                        cudaDeviceSynchronize();
                    }
                } else {
                    auto& gen = *static_cast<std::mt19937_64*>(
                        RandomGenerator::instance().get_generator(Device::CPU));

                    if (args.dtype == DataType::Float32) {
                        std::uniform_real_distribution<float> dist(low, high);
                        float* data = t.ptr<float>();
                        for (size_t i = 0; i < t.numel(); ++i) {
                            data[i] = dist(gen);
                        }
                    } else if (args.dtype == DataType::Int32) {
                        std::uniform_int_distribution<int> dist(static_cast<int>(low),
                                                               static_cast<int>(high) - 1);
                        int* data = t.ptr<int>();
                        for (size_t i = 0; i < t.numel(); ++i) {
                            data[i] = dist(gen);
                        }
                    }
                }

                return t;
            }
            LOG_ERROR("Random requires (low, high) pair");
            return {};
        }

        case LoadOp::Normal: {
            if (auto* params = std::get_if<std::pair<float, float>>(&args.args)) {
                auto t = load(LoadOp::Empty, args);
                if (!t.is_valid() || t.numel() == 0) return t;

                float mean = params->first;
                float std = params->second;

                if (args.device == Device::CUDA) {
                    tensor_ops::launch_normal(t.ptr<float>(), t.numel(), mean, std,
                                            RandomGenerator::instance().get_seed(), 0);
                    cudaDeviceSynchronize();
                } else {
                    auto& gen = *static_cast<std::mt19937_64*>(
                        RandomGenerator::instance().get_generator(Device::CPU));
                    std::normal_distribution<float> dist(mean, std);

                    float* data = t.ptr<float>();
                    for (size_t i = 0; i < t.numel(); ++i) {
                        data[i] = dist(gen);
                    }
                }

                return t;
            }
            LOG_ERROR("Normal requires (mean, std) pair");
            return {};
        }

        case LoadOp::Randint: {
            if (auto* range = std::get_if<std::pair<int, int>>(&args.args)) {
                auto t = load(LoadOp::Empty, args);
                if (!t.is_valid() || t.numel() == 0) return t;

                int low = range->first;
                int high = range->second;

                if (args.device == Device::CUDA) {
                    if (args.dtype == DataType::Int32) {
                        tensor_ops::launch_randint(t.ptr<int>(), t.numel(), low, high,
                                                  RandomGenerator::instance().get_seed(), 0);
                    } else if (args.dtype == DataType::Float32) {
                        // Generate ints and convert
                        auto temp = load(LoadOp::Empty, {args.shape, args.device, DataType::Int32});
                        tensor_ops::launch_randint(temp.ptr<int>(), t.numel(), low, high,
                                                  RandomGenerator::instance().get_seed(), 0);
                        cudaDeviceSynchronize();
                        t = temp.to(DataType::Float32);
                    }
                    cudaDeviceSynchronize();
                } else {
                    auto& gen = *static_cast<std::mt19937_64*>(
                        RandomGenerator::instance().get_generator(Device::CPU));
                    std::uniform_int_distribution<int> dist(low, high - 1);

                    if (args.dtype == DataType::Int32) {
                        int* data = t.ptr<int>();
                        for (size_t i = 0; i < t.numel(); ++i) {
                            data[i] = dist(gen);
                        }
                    } else if (args.dtype == DataType::Float32) {
                        float* data = t.ptr<float>();
                        for (size_t i = 0; i < t.numel(); ++i) {
                            data[i] = static_cast<float>(dist(gen));
                        }
                    }
                }

                return t;
            }
            LOG_ERROR("Randint requires (low, high) pair");
            return {};
        }

        case LoadOp::Bernoulli: {
            if (auto* p = std::get_if<float>(&args.args)) {
                auto t = load(LoadOp::Empty, args);
                if (!t.is_valid() || t.numel() == 0) return t;

                if (args.device == Device::CUDA) {
                    tensor_ops::launch_bernoulli(t.ptr<float>(), t.numel(), *p,
                                                RandomGenerator::instance().get_seed(), 0);
                    cudaDeviceSynchronize();
                } else {
                    auto& gen = *static_cast<std::mt19937_64*>(
                        RandomGenerator::instance().get_generator(Device::CPU));
                    std::bernoulli_distribution dist(*p);

                    float* data = t.ptr<float>();
                    for (size_t i = 0; i < t.numel(); ++i) {
                        data[i] = dist(gen) ? 1.0f : 0.0f;
                    }
                }

                return t;
            }
            LOG_ERROR("Bernoulli requires probability p");
            return {};
        }

        case LoadOp::Multinomial: {
            if (auto* params = std::get_if<std::pair<void*, bool>>(&args.args)) {
                const Tensor* weights = static_cast<const Tensor*>(params->first);
                bool replacement = params->second;

                if (!weights->is_valid() || weights->ndim() != 1) {
                    LOG_ERROR("Multinomial requires 1D weight tensor");
                    return Tensor();
                }

                size_t n = weights->numel();
                size_t num_samples = args.shape.elements();

                auto result = load(LoadOp::Empty, args);
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

                return result;
            }
            LOG_ERROR("Multinomial requires (weights, replacement) pair");
            return {};
        }

        case LoadOp::Eye: {
            auto t = load(LoadOp::Const, {args.shape, args.device, args.dtype, 0.0f});
            if (!t.is_valid() || args.shape.rank() != 2) return t;

            size_t m = args.shape[0];
            size_t n = args.shape[1];

            if (args.device == Device::CUDA) {
                tensor_ops::launch_eye(t.ptr<float>(), m, n, 0);
                cudaDeviceSynchronize();
            } else {
                float* data = t.ptr<float>();
                size_t min_dim = std::min(m, n);
                for (size_t i = 0; i < min_dim; ++i) {
                    data[i * n + i] = 1.0f;
                }
            }

            return t;
        }

        case LoadOp::FromCPU:
        case LoadOp::FromCUDA: {
            if (auto* ptr = std::get_if<void*>(&args.args)) {
                Tensor t(*ptr, args.shape, args.device, args.dtype);
                t.initialized_ = true;
                return t;
            }
            LOG_ERROR("FromCPU/FromCUDA requires void* pointer");
            return {};
        }

        default:
            LOG_ERROR("Unknown load operation");
            return {};
    }
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

} // namespace gs