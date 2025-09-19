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
                        std::vector<float> temp(t.numel(), *val);
                        cudaMemcpy(t.data_, temp.data(), t.bytes(), cudaMemcpyHostToDevice);
                    } else if (args.dtype == DataType::Bool) {
                        unsigned char fill = (*val != 0) ? 1 : 0;
                        cudaMemset(t.data_, fill, t.bytes());
                    }
                } else {
                    if (args.dtype == DataType::Float32) {
                        float* data = t.ptr<float>();
                        std::fill(data, data + t.numel(), *val);
                    } else if (args.dtype == DataType::Bool) {
                        unsigned char* data = t.ptr<unsigned char>();
                        unsigned char fill = (*val != 0) ? 1 : 0;
                        std::fill(data, data + t.numel(), fill);
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
                
                size_t n = static_cast<size_t>(std::ceil((stop - start) / step));
                auto t = load(LoadOp::Empty, {TensorShape({n}), args.device, args.dtype, {}});
                
                // Generate on CPU and copy
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
                    tensor_ops::launch_uniform(t.ptr<float>(), t.numel(), low, high,
                                             RandomGenerator::instance().get_seed(), 0);
                    cudaDeviceSynchronize();
                } else {
                    auto& gen = *static_cast<std::mt19937_64*>(
                        RandomGenerator::instance().get_generator(Device::CPU));
                    std::uniform_real_distribution<float> dist(low, high);
                    
                    float* data = t.ptr<float>();
                    for (size_t i = 0; i < t.numel(); ++i) {
                        data[i] = dist(gen);
                    }
                }
                
                return t;
            }
            LOG_ERROR("Random requires (low, high) pair");
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

} // namespace gs
