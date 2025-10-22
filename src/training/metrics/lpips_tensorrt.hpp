/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <string>
#include <torch/torch.h>

// Forward declarations for TensorRT
namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

namespace gs::training {

/**
 * TensorRT-based LPIPS (Learned Perceptual Image Patch Similarity) implementation.
 *
 * This class provides a LibTorch-free implementation of LPIPS using TensorRT for inference.
 * It maintains API compatibility with the original torch::jit::Module-based implementation.
 *
 * The model expects:
 * - Input: Two images in [0, 1] range (will be normalized to [-1, 1] internally)
 * - Format: [B, C, H, W] where B=batch, C=3 (RGB), H=height, W=width
 * - Output: Perceptual distance value (lower is more similar)
 */
class LPIPSTensorRT {
public:
    /**
     * Constructor - loads TensorRT engine from file
     * @param engine_path Path to .trt engine file (default: searches common locations)
     */
    explicit LPIPSTensorRT(const std::string& engine_path = "");

    /**
     * Destructor - cleans up TensorRT resources
     */
    ~LPIPSTensorRT();

    // Delete copy/move constructors to avoid double-free
    LPIPSTensorRT(const LPIPSTensorRT&) = delete;
    LPIPSTensorRT& operator=(const LPIPSTensorRT&) = delete;
    LPIPSTensorRT(LPIPSTensorRT&&) = delete;
    LPIPSTensorRT& operator=(LPIPSTensorRT&&) = delete;

    /**
     * Compute perceptual distance between two images
     * @param pred Predicted image tensor [B, C, H, W] in range [0, 1]
     * @param target Target image tensor [B, C, H, W] in range [0, 1]
     * @return Perceptual distance (average across batch)
     */
    float compute(const torch::Tensor& pred, const torch::Tensor& target);

    /**
     * Check if the TensorRT engine was successfully loaded
     * @return true if engine is loaded and ready for inference
     */
    bool is_loaded() const { return engine_loaded_; }

private:
    /**
     * Load TensorRT engine from file
     * @param engine_path Path to .trt engine file
     */
    void load_engine(const std::string& engine_path);

    /**
     * Find engine file in common locations
     * @return Path to engine file, or empty string if not found
     */
    std::string find_engine_file() const;

    /**
     * Allocate CUDA buffers for inference
     * @param batch_size Batch size for inference
     * @param height Image height
     * @param width Image width
     */
    void allocate_buffers(int batch_size, int height, int width);

    /**
     * Free allocated CUDA buffers
     */
    void free_buffers();

    // TensorRT components
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    // CUDA buffers for inference
    void* input_buffer_1_ = nullptr;  // First image input
    void* input_buffer_2_ = nullptr;  // Second image input
    void* output_buffer_ = nullptr;   // Distance output

    // Buffer state
    bool buffers_allocated_ = false;
    int allocated_batch_size_ = 0;
    int allocated_height_ = 0;
    int allocated_width_ = 0;

    // Engine state
    bool engine_loaded_ = false;

    // Binding indices
    int input_index_1_ = -1;
    int input_index_2_ = -1;
    int output_index_ = -1;
};

} // namespace gs::training
