/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lpips_tensorrt.hpp"

#include "core/logger.hpp"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace gs::training {

// TensorRT logger implementation
class TensorRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            LOG_ERROR("[TensorRT] {}", msg);
            break;
        case Severity::kWARNING:
            LOG_WARN("[TensorRT] {}", msg);
            break;
        case Severity::kINFO:
            LOG_INFO("[TensorRT] {}", msg);
            break;
        case Severity::kVERBOSE:
            LOG_DEBUG("[TensorRT] {}", msg);
            break;
        }
    }
};

// Global logger instance
static TensorRTLogger g_trt_logger;

LPIPSTensorRT::LPIPSTensorRT(const std::string& engine_path) {
    std::string path_to_load = engine_path;

    if (path_to_load.empty()) {
        path_to_load = find_engine_file();
        if (path_to_load.empty()) {
            throw std::runtime_error(
                "TensorRT LPIPS engine not found!\n"
                "Searched paths: weights/lpips_vgg.trt, ../weights/lpips_vgg.trt\n"
                "Please run: python3 scripts/convert_lpips_to_tensorrt.py");
        }
    }

    load_engine(path_to_load);
}

LPIPSTensorRT::~LPIPSTensorRT() {
    free_buffers();

    if (context_) {
        delete context_;
        context_ = nullptr;
    }

    if (engine_) {
        delete engine_;
        engine_ = nullptr;
    }

    if (runtime_) {
        delete runtime_;
        runtime_ = nullptr;
    }
}

std::string LPIPSTensorRT::find_engine_file() const {
    const std::vector<std::string> search_paths = {
        "weights/lpips_vgg.trt",
        "../weights/lpips_vgg.trt",
        "../../weights/lpips_vgg.trt",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/.cache/LichtFeld-Studio/lpips_vgg.trt"};

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }

    return "";
}

void LPIPSTensorRT::load_engine(const std::string& engine_path) {
    LOG_INFO("Loading TensorRT LPIPS engine from: {}", engine_path);

    // Log TensorRT library version (compile-time version from headers)
    LOG_INFO("TensorRT library version: {}.{}.{}.{}",
             NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD);

    // Read engine file
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
        throw std::runtime_error("Failed to open TensorRT engine file: " + engine_path);
    }

    engine_file.seekg(0, std::ios::end);
    const size_t file_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(file_size);
    engine_file.read(engine_data.data(), file_size);
    engine_file.close();

    // Create runtime and deserialize engine
    runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), file_size);
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize TensorRT engine from: " + engine_path);
    }

    // Create execution context
    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    // Find binding indices
    const int num_bindings = engine_->getNbIOTensors();
    LOG_DEBUG("TensorRT engine has {} IO tensors", num_bindings);

    for (int i = 0; i < num_bindings; ++i) {
        const char* name = engine_->getIOTensorName(i);
        const auto mode = engine_->getTensorIOMode(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (input_index_1_ == -1) {
                input_index_1_ = i;
                LOG_DEBUG("  Input 1: '{}' at index {}", name, i);
            } else if (input_index_2_ == -1) {
                input_index_2_ = i;
                LOG_DEBUG("  Input 2: '{}' at index {}", name, i);
            }
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_index_ = i;
            LOG_DEBUG("  Output: '{}' at index {}", name, i);
        }
    }

    if (input_index_1_ == -1 || input_index_2_ == -1 || output_index_ == -1) {
        throw std::runtime_error("Failed to find all required tensor bindings in TensorRT engine");
    }

    engine_loaded_ = true;
    LOG_INFO("TensorRT LPIPS engine loaded successfully");
}

void LPIPSTensorRT::allocate_buffers(const int batch_size, const int height, const int width) {
    // Check if we can reuse existing buffers
    if (buffers_allocated_ &&
        allocated_batch_size_ == batch_size &&
        allocated_height_ == height &&
        allocated_width_ == width) {
        return; // Buffers already allocated with correct size
    }

    // Free old buffers if they exist
    free_buffers();

    // Calculate buffer sizes
    const size_t input_size = batch_size * 3 * height * width * sizeof(float);
    const size_t output_size = batch_size * sizeof(float); // Output is [B, 1, 1, 1] flattened to [B]

    // Allocate CUDA memory
    cudaError_t err;

    err = cudaMalloc(&input_buffer_1_, input_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA memory for input 1: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&input_buffer_2_, input_size);
    if (err != cudaSuccess) {
        cudaFree(input_buffer_1_);
        throw std::runtime_error("Failed to allocate CUDA memory for input 2: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&output_buffer_, output_size);
    if (err != cudaSuccess) {
        cudaFree(input_buffer_1_);
        cudaFree(input_buffer_2_);
        throw std::runtime_error("Failed to allocate CUDA memory for output: " + std::string(cudaGetErrorString(err)));
    }

    buffers_allocated_ = true;
    allocated_batch_size_ = batch_size;
    allocated_height_ = height;
    allocated_width_ = width;

    LOG_DEBUG("Allocated TensorRT buffers for batch={}, H={}, W={}", batch_size, height, width);
}

void LPIPSTensorRT::free_buffers() {
    if (input_buffer_1_) {
        cudaFree(input_buffer_1_);
        input_buffer_1_ = nullptr;
    }
    if (input_buffer_2_) {
        cudaFree(input_buffer_2_);
        input_buffer_2_ = nullptr;
    }
    if (output_buffer_) {
        cudaFree(output_buffer_);
        output_buffer_ = nullptr;
    }

    buffers_allocated_ = false;
    allocated_batch_size_ = 0;
    allocated_height_ = 0;
    allocated_width_ = 0;
}

float LPIPSTensorRT::compute(const torch::Tensor& pred, const torch::Tensor& target) {
    if (!engine_loaded_) {
        throw std::runtime_error("TensorRT engine not loaded!");
    }

    // Validate inputs
    if (pred.dim() != 4) {
        throw std::runtime_error("Expected 4D tensor [B, C, H, W], got " + std::to_string(pred.dim()) + "D");
    }

    if (pred.sizes() != target.sizes()) {
        throw std::runtime_error("Prediction and target must have the same shape");
    }

    if (pred.size(1) != 3) {
        throw std::runtime_error("Expected 3 channels (RGB), got " + std::to_string(pred.size(1)));
    }

    // Get dimensions
    const int batch_size = pred.size(0);
    const int height = pred.size(2);
    const int width = pred.size(3);

    // Allocate buffers if needed
    allocate_buffers(batch_size, height, width);

    // Normalize inputs from [0, 1] to [-1, 1] (LPIPS expects this range)
    auto pred_normalized = (2.0f * pred - 1.0f).to(torch::kCUDA).contiguous();
    auto target_normalized = (2.0f * target - 1.0f).to(torch::kCUDA).contiguous();

    // Copy data to GPU buffers
    const size_t input_size = batch_size * 3 * height * width * sizeof(float);

    cudaError_t err;
    err = cudaMemcpy(input_buffer_1_, pred_normalized.data_ptr<float>(), input_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy input 1 to device: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(input_buffer_2_, target_normalized.data_ptr<float>(), input_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy input 2 to device: " + std::string(cudaGetErrorString(err)));
    }

    // Set input shapes (for dynamic shapes)
    nvinfer1::Dims4 input_dims{batch_size, 3, height, width};

    const char* input_name_1 = engine_->getIOTensorName(input_index_1_);
    const char* input_name_2 = engine_->getIOTensorName(input_index_2_);
    const char* output_name = engine_->getIOTensorName(output_index_);

    context_->setInputShape(input_name_1, input_dims);
    context_->setInputShape(input_name_2, input_dims);

    // Set tensor addresses
    context_->setTensorAddress(input_name_1, input_buffer_1_);
    context_->setTensorAddress(input_name_2, input_buffer_2_);
    context_->setTensorAddress(output_name, output_buffer_);

    // Execute inference
    const bool success = context_->enqueueV3(nullptr); // Use default CUDA stream
    if (!success) {
        throw std::runtime_error("TensorRT inference failed");
    }

    // Synchronize to ensure inference is complete
    cudaDeviceSynchronize();

    // Copy output back to host
    std::vector<float> output_host(batch_size);
    err = cudaMemcpy(output_host.data(), output_buffer_, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy output from device: " + std::string(cudaGetErrorString(err)));
    }

    // Calculate mean across batch (to match original LPIPS behavior)
    float sum = 0.0f;
    for (const float val : output_host) {
        sum += val;
    }

    return sum / static_cast<float>(batch_size);
}

} // namespace gs::training
