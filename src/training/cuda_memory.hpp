/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "kernels/training_kernels.cuh"
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace gs::training {

    // Simple CUDA memory wrapper - no LibTorch dependency
    class CUDABuffer {
    private:
        float* data_ = nullptr;
        size_t size_ = 0;

    public:
        CUDABuffer() = default;

        explicit CUDABuffer(size_t elements) {
            allocate(elements);
        }

        ~CUDABuffer() {
            free();
        }

        // Move only, no copy
        CUDABuffer(const CUDABuffer&) = delete;
        CUDABuffer& operator=(const CUDABuffer&) = delete;

        CUDABuffer(CUDABuffer&& other) noexcept
            : data_(other.data_),
              size_(other.size_) {
            other.data_ = nullptr;
            other.size_ = 0;
        }

        CUDABuffer& operator=(CUDABuffer&& other) noexcept {
            if (this != &other) {
                free();
                data_ = other.data_;
                size_ = other.size_;
                other.data_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }

        void allocate(size_t elements) {
            if (data_ && size_ == elements) {
                return; // Already allocated with same size
            }

            free();

            if (elements > 0) {
                cudaError_t err = cudaMalloc(&data_, elements * sizeof(float));
                if (err != cudaSuccess) {
                    throw std::runtime_error("CUDA allocation failed: " +
                                             std::string(cudaGetErrorString(err)));
                }
                size_ = elements;
            }
        }

        void free() {
            if (data_) {
                cudaFree(data_);
                data_ = nullptr;
                size_ = 0;
            }
        }

        // Zero the memory
        void zero() {
            if (data_ && size_ > 0) {
                cudaMemset(data_, 0, size_ * sizeof(float));
            }
        }

        // Fill with value
        void fill(float value) {
            if (data_ && size_ > 0) {
                launch_fill_tensor(data_, value, size_, 0);
            }
        }

        // Copy from another buffer
        void copy_from(const float* src, size_t elements) {
            if (elements > size_) {
                throw std::runtime_error("Copy size exceeds buffer size");
            }
            cudaMemcpy(data_, src, elements * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        // Copy from host
        void copy_from_host(const float* src, size_t elements) {
            if (elements > size_) {
                throw std::runtime_error("Copy size exceeds buffer size");
            }
            cudaMemcpy(data_, src, elements * sizeof(float), cudaMemcpyHostToDevice);
        }

        // Copy to host (for getting loss values)
        void copy_to_host(float* dst, size_t elements) const {
            if (elements > size_) {
                throw std::runtime_error("Copy size exceeds buffer size");
            }
            cudaMemcpy(dst, data_, elements * sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Accessors
        float* data() { return data_; }
        const float* data() const { return data_; }
        size_t size() const { return size_; }
        bool allocated() const { return data_ != nullptr; }
    };

    // Pre-allocated buffers for training
    class TrainingMemory {
    private:
        // Render buffers (for forward pass)
        CUDABuffer render_image_;
        CUDABuffer render_alpha_;

        // Gradient buffers (for backward pass)
        CUDABuffer grad_image_;
        CUDABuffer grad_alpha_;

        // Loss computation buffers
        CUDABuffer diff_buffer_;
        CUDABuffer abs_diff_buffer_;
        CUDABuffer l1_grad_buffer_;
        CUDABuffer ones_buffer_;

        // SSIM-specific buffers
        CUDABuffer ssim_map_;     // SSIM values per pixel
        CUDABuffer ssim_dm_dmu1_; // Partial derivatives
        CUDABuffer ssim_dm_dsigma1_sq_;
        CUDABuffer ssim_dm_dsigma12_;
        CUDABuffer ssim_grad_image_; // Final SSIM gradient
        CUDABuffer ssim_dL_dmap_;    // Gradient w.r.t SSIM map

        // Background buffers
        CUDABuffer background_;
        CUDABuffer bg_mix_buffer_;

        // Loss value buffers (single floats)
        CUDABuffer loss_value_;
        CUDABuffer ssim_loss_value_;

        // Dimensions
        int width_ = 0;
        int height_ = 0;
        int channels_ = 3;
        int batch_ = 1;

    public:
        TrainingMemory() = default;

        void initialize(int width, int height, int channels = 3, int batch = 1) {
            width_ = width;
            height_ = height;
            channels_ = channels;
            batch_ = batch;

            const size_t image_size = batch * channels * height * width;
            const size_t alpha_size = batch * height * width;

            // Allocate render buffers
            render_image_.allocate(image_size);
            render_alpha_.allocate(alpha_size);

            // Allocate gradient buffers
            grad_image_.allocate(image_size);
            grad_alpha_.allocate(alpha_size);

            // Allocate loss computation buffers
            diff_buffer_.allocate(image_size);
            abs_diff_buffer_.allocate(image_size);
            l1_grad_buffer_.allocate(image_size);
            ones_buffer_.allocate(image_size);

            // Allocate SSIM buffers
            ssim_map_.allocate(image_size);
            ssim_dm_dmu1_.allocate(image_size);
            ssim_dm_dsigma1_sq_.allocate(image_size);
            ssim_dm_dsigma12_.allocate(image_size);
            ssim_grad_image_.allocate(image_size);
            ssim_dL_dmap_.allocate(image_size);

            // Background is just 3 floats
            background_.allocate(3);
            bg_mix_buffer_.allocate(3);

            // Loss values are single floats
            loss_value_.allocate(1);
            ssim_loss_value_.allocate(1);

            // Initialize background to black
            background_.zero();

            // Initialize ones buffer
            ones_buffer_.fill(1.0f);
        }

        bool allocated() const {
            return background_.allocated() && render_image_.allocated() && grad_image_.allocated();
        }

        // Check if we need to reallocate for new dimensions
        void ensure_size(int width, int height, int channels = 3, int batch = 1) {
            if (width != width_ || height != height_ || channels != channels_ || batch != batch_) {
                initialize(width, height, channels, batch);
            }
        }

        // Render buffer accessors (for forward pass)
        float* render_image() { return render_image_.data(); }
        float* render_alpha() { return render_alpha_.data(); }
        const float* render_image() const { return render_image_.data(); }
        const float* render_alpha() const { return render_alpha_.data(); }

        // Gradient buffer accessors (for backward pass)
        float* grad_image() { return grad_image_.data(); }
        float* grad_alpha() { return grad_alpha_.data(); }

        // Loss computation buffer accessors
        float* diff_buffer() { return diff_buffer_.data(); }
        float* abs_diff_buffer() { return abs_diff_buffer_.data(); }
        float* l1_grad() { return l1_grad_buffer_.data(); }
        float* ones() { return ones_buffer_.data(); }

        // SSIM buffer accessors
        float* ssim_map() { return ssim_map_.data(); }
        float* ssim_dm_dmu1() { return ssim_dm_dmu1_.data(); }
        float* ssim_dm_dsigma1_sq() { return ssim_dm_dsigma1_sq_.data(); }
        float* ssim_dm_dsigma12() { return ssim_dm_dsigma12_.data(); }
        float* ssim_grad() { return ssim_grad_image_.data(); }
        float* ssim_dL_dmap() { return ssim_dL_dmap_.data(); }

        // Background buffer accessors
        float* background() { return background_.data(); }
        float* bg_mix() { return bg_mix_buffer_.data(); }

        // Loss value accessors
        float* loss_value() { return loss_value_.data(); }
        float* ssim_loss_value() { return ssim_loss_value_.data(); }

        // Utility functions
        void zero_gradients() {
            grad_image_.zero();
            grad_alpha_.zero();
        }

        void zero_render_buffers() {
            render_image_.zero();
            render_alpha_.zero();
        }

        void set_background(float r, float g, float b) {
            float bg[3] = {r, g, b};
            background_.copy_from_host(bg, 3);
        }

        // Get loss value from GPU
        float get_loss_value() {
            float val;
            loss_value_.copy_to_host(&val, 1);
            return val;
        }

        float get_ssim_loss_value() {
            float val;
            ssim_loss_value_.copy_to_host(&val, 1);
            return val;
        }

        // Get sizes
        size_t image_size() const { return batch_ * channels_ * height_ * width_; }
        size_t alpha_size() const { return batch_ * height_ * width_; }
        int width() const { return width_; }
        int height() const { return height_; }
        int channels() const { return channels_; }
        int batch() const { return batch_; }
    };

} // namespace gs::training