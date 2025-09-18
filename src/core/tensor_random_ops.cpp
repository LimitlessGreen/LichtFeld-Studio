/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <random>

#define CHECK_CUDA(call)                              \
    do {                                              \
        cudaError_t error = call;                     \
        if (error != cudaSuccess) {                   \
            LOG_ERROR("CUDA error at {}:{} - {}: {}", \
                      __FILE__, __LINE__,             \
                      cudaGetErrorName(error),        \
                      cudaGetErrorString(error));     \
        }                                             \
    } while (0)

#define CHECK_CURAND(call)                             \
    do {                                               \
        curandStatus_t error = call;                   \
        if (error != CURAND_STATUS_SUCCESS) {          \
            LOG_ERROR("CURAND error at {}:{} - {}",    \
                      __FILE__, __LINE__, (int)error); \
        }                                              \
    } while (0)

namespace gs {

    // ============= RandomGenerator Implementation =============
    RandomGenerator& RandomGenerator::instance() {
        static RandomGenerator instance;
        return instance;
    }

    RandomGenerator::RandomGenerator() : seed_(42),
                                         cpu_generator_(seed_) {
        // Initialize CUDA random generator
        curandGenerator_t* gen = new curandGenerator_t;
        CHECK_CURAND(curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(*gen, seed_));
        cuda_generator_ = gen;
    }

    RandomGenerator::~RandomGenerator() {
        if (cuda_generator_) {
            curandGenerator_t* gen = static_cast<curandGenerator_t*>(cuda_generator_);
            curandDestroyGenerator(*gen);
            delete gen;
        }
    }

    void RandomGenerator::manual_seed(uint64_t seed) {
        seed_ = seed;
        cpu_generator_.seed(seed);

        if (cuda_generator_) {
            curandGenerator_t* gen = static_cast<curandGenerator_t*>(cuda_generator_);
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(*gen, seed));
        }
    }

    void* RandomGenerator::get_generator(Device device) {
        if (device == Device::CUDA) {
            return cuda_generator_;
        } else {
            return &cpu_generator_;
        }
    }

    // ============= Random Tensor Factory Methods =============
    Tensor Tensor::rand(TensorShape shape, Device device, DataType dtype) {
        return uniform(shape, 0.0f, 1.0f, device, dtype);
    }

    Tensor Tensor::randn(TensorShape shape, Device device, DataType dtype) {
        return normal(shape, 0.0f, 1.0f, device, dtype);
    }

    Tensor Tensor::uniform(TensorShape shape, float low, float high, Device device, DataType dtype) {
        if (dtype != DataType::Float32) {
            LOG_ERROR("Random operations only implemented for float32");
            return Tensor();
        }

        auto result = empty(shape, device, dtype);
        if (!result.is_valid() || result.numel() == 0) {
            return result;
        }

        result.uniform_(low, high);
        return result;
    }

    Tensor Tensor::normal(TensorShape shape, float mean, float std, Device device, DataType dtype) {
        if (dtype != DataType::Float32) {
            LOG_ERROR("Random operations only implemented for float32");
            return Tensor();
        }

        auto result = empty(shape, device, dtype);
        if (!result.is_valid() || result.numel() == 0) {
            return result;
        }

        result.normal_(mean, std);
        return result;
    }

    Tensor Tensor::randint(TensorShape shape, int low, int high, Device device, DataType dtype) {
        if (dtype != DataType::Int32 && dtype != DataType::Float32) {
            LOG_ERROR("Randint only supports int32 and float32");
            return Tensor();
        }

        auto result = empty(shape, device, dtype);
        if (!result.is_valid() || result.numel() == 0) {
            return result;
        }

        size_t n = result.numel();

        if (device == Device::CUDA) {
            if (dtype == DataType::Int32) {
                tensor_ops::launch_randint(
                    result.ptr<int>(), n, low, high,
                    RandomGenerator::instance().get_seed(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            } else {
                // For float32, generate integers and convert
                auto temp = empty(shape, device, DataType::Int32);
                tensor_ops::launch_randint(
                    temp.ptr<int>(), n, low, high,
                    RandomGenerator::instance().get_seed(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());

                // Convert to float
                int* int_data = temp.ptr<int>();
                float* float_data = result.ptr<float>();

                // For simplicity, do it on CPU
                std::vector<int> int_vals(n);
                CHECK_CUDA(cudaMemcpy(int_vals.data(), int_data, n * sizeof(int),
                                      cudaMemcpyDeviceToHost));
                std::vector<float> float_vals(n);
                for (size_t i = 0; i < n; ++i) {
                    float_vals[i] = static_cast<float>(int_vals[i]);
                }
                // FIXED: Complete the cudaMemcpy line properly
                CHECK_CUDA(cudaMemcpy(float_data, float_vals.data(), n * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
        } else {
            auto& gen = *static_cast<std::mt19937_64*>(
                RandomGenerator::instance().get_generator(Device::CPU));
            std::uniform_int_distribution<int> dist(low, high - 1);

            if (dtype == DataType::Int32) {
                int* data = result.ptr<int>();
                for (size_t i = 0; i < n; ++i) {
                    data[i] = dist(gen);
                }
            } else {
                float* data = result.ptr<float>();
                for (size_t i = 0; i < n; ++i) {
                    data[i] = static_cast<float>(dist(gen));
                }
            }
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        return result;
    }

    Tensor Tensor::bernoulli(TensorShape shape, float p, Device device, DataType dtype) {
        if (dtype != DataType::Float32) {
            LOG_ERROR("Bernoulli only implemented for float32");
            return Tensor();
        }

        auto result = empty(shape, device, dtype);
        if (!result.is_valid() || result.numel() == 0) {
            return result;
        }

        size_t n = result.numel();

        if (device == Device::CUDA) {
            tensor_ops::launch_bernoulli(
                result.ptr<float>(), n, p,
                RandomGenerator::instance().get_seed(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            auto& gen = *static_cast<std::mt19937_64*>(
                RandomGenerator::instance().get_generator(Device::CPU));
            std::bernoulli_distribution dist(p);

            float* data = result.ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                data[i] = dist(gen) ? 1.0f : 0.0f;
            }
        }

        return result;
    }

    Tensor Tensor::multinomial(const Tensor& weights, int num_samples, bool replacement) {
        if (!weights.is_valid() || weights.ndim() != 1) {
            LOG_ERROR("Multinomial requires 1D weight tensor");
            return Tensor();
        }

        size_t n = weights.numel();

        // For very large arrays (> 2^24), we need custom implementation
        if (n > (1 << 24)) {
            // Fallback to CPU implementation for huge arrays
            auto weights_cpu = weights.to(Device::CPU);
            auto weights_data = weights_cpu.to_vector();

            // Normalize weights
            float sum = 0.0f;
            for (float w : weights_data) {
                sum += w;
            }

            if (sum <= 0) {
                LOG_ERROR("Weights must sum to positive value");
                return Tensor();
            }

            // Create cumulative distribution
            std::vector<float> cumsum(n);
            cumsum[0] = weights_data[0] / sum;
            for (size_t i = 1; i < n; ++i) {
                cumsum[i] = cumsum[i-1] + weights_data[i] / sum;
            }

            // Sample
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.0f, 1.0f);

            std::vector<int> sampled_indices;
            sampled_indices.reserve(num_samples);

            for (int i = 0; i < num_samples; ++i) {
                float u = dis(gen);

                // Binary search for the index
                auto it = std::lower_bound(cumsum.begin(), cumsum.end(), u);
                size_t idx = std::distance(cumsum.begin(), it);
                if (idx >= n) idx = n - 1;

                sampled_indices.push_back(static_cast<int>(idx));

                if (!replacement) {
                    // For without replacement, we'd need to update the CDF
                    // This is complex, so for now we only support with replacement
                    if (i == 0) {
                        LOG_WARN("Multinomial without replacement for large arrays not fully implemented");
                    }
                }
            }

            // Create result tensor
            auto result = empty({static_cast<size_t>(num_samples)}, weights.device(), DataType::Int32);

            if (weights.device() == Device::CUDA) {
                CHECK_CUDA(cudaMemcpy(result.ptr<int>(), sampled_indices.data(),
                                      num_samples * sizeof(int), cudaMemcpyHostToDevice));
            } else {
                std::memcpy(result.ptr<int>(), sampled_indices.data(), num_samples * sizeof(int));
            }

            return result;
        }

        // For smaller arrays, use CUDA kernel
        auto result = empty({static_cast<size_t>(num_samples)}, weights.device(), DataType::Int32);

        if (weights.device() == Device::CUDA) {
            tensor_ops::launch_multinomial(weights.ptr<float>(), result.ptr<int>(),
                                          n, num_samples, replacement,
                                          RandomGenerator::instance().get_seed(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation
            auto weights_data = weights.to_vector();

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
            for (int i = 0; i < num_samples; ++i) {
                float u = dis(gen);
                auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
                samples[i] = static_cast<int>(std::distance(cdf.begin(), it));
            }
        }

        return result;
    }

    void Tensor::set_bool(std::initializer_list<size_t> indices, bool value) {
        if (dtype_ != DataType::Bool) {
            LOG_ERROR("set_bool only works on boolean tensors");
            return;
        }

        if (indices.size() != shape_.rank()) {
            LOG_ERROR("Number of indices must match tensor rank");
            return;
        }

        size_t linear_idx = 0;
        size_t stride = 1;
        auto it = indices.end();
        for (int i = shape_.rank() - 1; i >= 0; --i) {
            --it;
            if (*it >= shape_[i]) {
                LOG_ERROR("Index {} out of bounds for dimension {}", *it, i);
                return;
            }
            linear_idx += (*it) * stride;
            stride *= shape_[i];
        }

        if (device_ == Device::CUDA) {
            unsigned char val = value ? 1 : 0;
            CHECK_CUDA(cudaMemcpy(ptr<unsigned char>() + linear_idx, &val, sizeof(unsigned char),
                                  cudaMemcpyHostToDevice));
        } else {
            ptr<unsigned char>()[linear_idx] = value ? 1 : 0;
        }
    }

    bool Tensor::get_bool(std::initializer_list<size_t> indices) const {
        if (dtype_ != DataType::Bool) {
            LOG_ERROR("get_bool only works on boolean tensors");
            return false;
        }

        if (indices.size() != shape_.rank()) {
            LOG_ERROR("Number of indices must match tensor rank");
            return false;
        }

        size_t linear_idx = 0;
        size_t stride = 1;
        auto it = indices.end();
        for (int i = shape_.rank() - 1; i >= 0; --i) {
            --it;
            if (*it >= shape_[i]) {
                LOG_ERROR("Index {} out of bounds for dimension {}", *it, i);
                return false;
            }
            linear_idx += (*it) * stride;
            stride *= shape_[i];
        }

        if (device_ == Device::CUDA) {
            unsigned char value;
            CHECK_CUDA(cudaMemcpy(&value, ptr<unsigned char>() + linear_idx, sizeof(unsigned char),
                                  cudaMemcpyDeviceToHost));
            return value != 0;
        }

        return ptr<unsigned char>()[linear_idx] != 0;
    }

    // ============= In-place Random Operations =============
    Tensor& Tensor::uniform_(float low, float high) {
        if (dtype_ != DataType::Float32) {
            LOG_ERROR("uniform_ only implemented for float32");
            return *this;
        }

        if (!is_valid() || numel() == 0) {
            return *this;
        }

        size_t n = numel();

        if (device_ == Device::CUDA) {
            // Use kernel-based generation for reproducibility
            tensor_ops::launch_uniform(
                ptr<float>(), n, low, high,
                RandomGenerator::instance().get_seed(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            auto& gen = *static_cast<std::mt19937_64*>(
                RandomGenerator::instance().get_generator(Device::CPU));
            std::uniform_real_distribution<float> dist(low, high);

            float* data = ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                data[i] = dist(gen);
            }
        }

        return *this;
    }

    Tensor& Tensor::normal_(float mean, float std) {
        if (dtype_ != DataType::Float32) {
            LOG_ERROR("normal_ only implemented for float32");
            return *this;
        }

        if (!is_valid() || numel() == 0) {
            return *this;
        }

        size_t n = numel();

        if (device_ == Device::CUDA) {
            // Use kernel-based generation for reproducibility
            tensor_ops::launch_normal(
                ptr<float>(), n, mean, std,
                RandomGenerator::instance().get_seed(), 0);
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            auto& gen = *static_cast<std::mt19937_64*>(
                RandomGenerator::instance().get_generator(Device::CPU));
            std::normal_distribution<float> dist(mean, std);

            float* data = ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                data[i] = dist(gen);
            }
        }

        return *this;
    }

#undef CHECK_CUDA
#undef CHECK_CURAND

} // namespace gs
