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