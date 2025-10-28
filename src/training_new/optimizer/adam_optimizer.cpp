/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_optimizer.hpp"
#include "adam_api.h"  // fast_lfs::optimizer::adam_step_raw
#include "core_new/logger.hpp"
#include <cmath>
#include <stdexcept>

namespace lfs::training {

    AdamOptimizer::AdamOptimizer(lfs::core::SplatData& splat_data, const AdamConfig& config)
        : splat_data_(splat_data), config_(config) {

        // Ensure gradients are allocated
        if (!splat_data_.has_gradients()) {
            splat_data_.allocate_gradients();
            LOG_DEBUG("Allocated gradients for optimizer");
        }
    }

    void AdamOptimizer::step(int iteration) {
        // Optimize each parameter
        step_param(ParamType::Means, iteration);
        step_param(ParamType::Sh0, iteration);
        step_param(ParamType::ShN, iteration);
        step_param(ParamType::Scaling, iteration);
        step_param(ParamType::Rotation, iteration);
        step_param(ParamType::Opacity, iteration);
    }

    void AdamOptimizer::zero_grad(int iteration) {
        // TODO: Optional - Skip SH gradients on certain iterations (matching old behavior)
        // For now, just zero everything
        splat_data_.zero_gradients();
    }

    lfs::core::Tensor& AdamOptimizer::get_param(ParamType type) {
        switch (type) {
            case ParamType::Means: return splat_data_.means();
            case ParamType::Sh0: return splat_data_.sh0();
            case ParamType::ShN: return splat_data_.shN();
            case ParamType::Scaling: return splat_data_.scaling_raw();
            case ParamType::Rotation: return splat_data_.rotation_raw();
            case ParamType::Opacity: return splat_data_.opacity_raw();
        }
        throw std::runtime_error("Invalid param type");
    }

    lfs::core::Tensor& AdamOptimizer::get_grad(ParamType type) {
        switch (type) {
            case ParamType::Means: return splat_data_.means_grad();
            case ParamType::Sh0: return splat_data_.sh0_grad();
            case ParamType::ShN: return splat_data_.shN_grad();
            case ParamType::Scaling: return splat_data_.scaling_grad();
            case ParamType::Rotation: return splat_data_.rotation_grad();
            case ParamType::Opacity: return splat_data_.opacity_grad();
        }
        throw std::runtime_error("Invalid param type");
    }

    std::string AdamOptimizer::param_name(ParamType type) const {
        switch (type) {
            case ParamType::Means: return "means";
            case ParamType::Sh0: return "sh0";
            case ParamType::ShN: return "shN";
            case ParamType::Scaling: return "scaling";
            case ParamType::Rotation: return "rotation";
            case ParamType::Opacity: return "opacity";
        }
        return "unknown";
    }

    void AdamOptimizer::init_state(ParamType type) {
        auto& param = get_param(type);
        auto name = param_name(type);

        auto& state = states_[name];
        size_t param_size = param.shape()[0];

        // Calculate initial capacity with pre-allocation if configured
        size_t initial_cap = compute_new_capacity(0, param_size);

        // Allocate with extra capacity if growth_factor or initial_capacity is set
        if (initial_cap > param_size) {
            auto param_shape = param.shape();
            std::vector<size_t> alloc_dims(param_shape.rank());
            for (size_t i = 0; i < param_shape.rank(); i++) {
                alloc_dims[i] = (i == 0) ? initial_cap : param_shape[i];
            }
            state.exp_avg = lfs::core::Tensor::zeros(lfs::core::TensorShape(alloc_dims), param.device());
            state.exp_avg_sq = lfs::core::Tensor::zeros(lfs::core::TensorShape(alloc_dims), param.device());
            state.capacity = initial_cap;
            state.size = param_size;

            LOG_DEBUG("Initialized optimizer state for {} with pre-allocation (size: {}, capacity: {})",
                      name, param_size, initial_cap);
        } else {
            // No pre-allocation: exact fit
            state.exp_avg = lfs::core::Tensor::zeros(param.shape(), param.device());
            state.exp_avg_sq = lfs::core::Tensor::zeros(param.shape(), param.device());
            state.capacity = param_size;
            state.size = param_size;

            LOG_DEBUG("Initialized optimizer state for {} (size: {}, no pre-allocation)", name, param_size);
        }

        state.step_count = 0;
    }

    void AdamOptimizer::step_param(ParamType type, int iteration) {
        auto& param = get_param(type);
        auto& grad = get_grad(type);

        // Skip if no gradient or if gradient is all zeros (not yet computed)
        if (!grad.is_valid() || grad.numel() == 0) {
            return;
        }

        // OPTIMIZATION: Skip if parameter doesn't exist yet (lazy initialization)
        if (!param.is_valid() || param.numel() == 0) {
            return;
        }

        auto name = param_name(type);

        // Initialize state on first call
        if (states_.find(name) == states_.end()) {
            init_state(type);
        }

        auto& state = states_[name];
        state.step_count++;

        // Compute bias correction factors
        float bias_correction1_rcp = 1.0f / (1.0f - std::pow(config_.beta1, state.step_count));
        float bias_correction2_sqrt_rcp = 1.0f / std::sqrt(1.0f - std::pow(config_.beta2, state.step_count));

        // OPTIMIZATION: Only operate on the "used" portion if we have excess capacity
        // This is safe because adam_step_raw only touches the first `numel` elements
        // The state tensors may be larger than param, but we only use state.size elements

        // Call fused CUDA kernel (operates on param.numel() elements)
        fast_lfs::optimizer::adam_step_raw(
            param.ptr<float>(),
            state.exp_avg.ptr<float>(),
            state.exp_avg_sq.ptr<float>(),
            grad.ptr<float>(),
            static_cast<int>(param.numel()),
            config_.lr,
            config_.beta1,
            config_.beta2,
            config_.eps,
            bias_correction1_rcp,
            bias_correction2_sqrt_rcp
        );
    }

    void AdamOptimizer::reset_state_at_indices(ParamType type, const std::vector<int64_t>& indices) {
        auto name = param_name(type);

        // Ensure state exists
        if (states_.find(name) == states_.end()) {
            LOG_DEBUG("State for {} not initialized yet, skipping reset", name);
            return;
        }

        if (indices.empty()) {
            return;  // Nothing to do
        }

        auto& state = states_[name];

        // Calculate row size (product of all dimensions except first)
        auto state_shape = state.exp_avg.shape();
        int row_size = 1;
        for (size_t i = 1; i < state_shape.rank(); i++) {
            row_size *= state_shape[i];
        }

        // Allocate GPU memory for indices and copy from host
        int64_t* indices_device_ptr;
        cudaMalloc(&indices_device_ptr, indices.size() * sizeof(int64_t));
        cudaMemcpy(indices_device_ptr, indices.data(), indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

        // Use batched CUDA kernel for much better performance (600x faster!)
        fast_lfs::optimizer::zero_rows_at_indices(
            state.exp_avg.template ptr<float>(),
            indices_device_ptr,
            indices.size(),
            row_size
        );

        fast_lfs::optimizer::zero_rows_at_indices(
            state.exp_avg_sq.template ptr<float>(),
            indices_device_ptr,
            indices.size(),
            row_size
        );

        cudaFree(indices_device_ptr);

        LOG_DEBUG("Reset optimizer state for {} at {} indices (batched GPU kernel)", name, indices.size());
    }

    void AdamOptimizer::extend_state_for_new_params(ParamType type, size_t n_new) {
        auto& param = get_param(type);
        auto name = param_name(type);

        // Ensure state exists
        if (states_.find(name) == states_.end()) {
            // If state doesn't exist yet, it will be initialized on first step
            LOG_DEBUG("State for {} not initialized yet, will be initialized on first step", name);
            return;
        }

        auto& state = states_[name];
        size_t new_size = state.size + n_new;

        // OPTIMIZATION: Check if we have enough capacity (avoids reallocation)
        if (new_size <= state.capacity) {
            // Fast path: Just update the size, no allocation needed!
            state.size = new_size;
            LOG_DEBUG("Extended optimizer state for {} by {} parameters using reserved capacity "
                      "(size: {} -> {}, capacity: {}, no allocation!)",
                      name, n_new, state.size - n_new, state.size, state.capacity);
            return;
        }

        // Slow path: Need to grow capacity
        auto param_shape = param.shape();
        std::vector<size_t> zeros_dims(param_shape.rank());

        // Calculate new capacity with growth factor (like std::vector)
        size_t new_capacity = compute_new_capacity(state.capacity, new_size);

        // Allocate new tensors with extra capacity
        for (size_t i = 0; i < param_shape.rank(); i++) {
            zeros_dims[i] = (i == 0) ? (new_capacity - state.size) : param_shape[i];
        }
        auto zeros_shape = lfs::core::TensorShape(zeros_dims);
        auto zeros = lfs::core::Tensor::zeros(zeros_shape, param.device());

        // Concatenate to grow
        state.exp_avg = lfs::core::Tensor::cat(std::vector<lfs::core::Tensor>{state.exp_avg, zeros}, 0);
        state.exp_avg_sq = lfs::core::Tensor::cat(std::vector<lfs::core::Tensor>{state.exp_avg_sq, zeros}, 0);

        // Update capacity and size
        state.capacity = new_capacity;
        state.size = new_size;

        // IMPORTANT: Preserve step_count (matching MCMC behavior)
        LOG_DEBUG("Extended optimizer state for {} by {} parameters with reallocation "
                  "(size: {} -> {}, capacity: {} -> {}, growth_factor: {:.2f})",
                  name, n_new, state.size - n_new, state.size,
                  state.size - n_new, state.capacity, config_.growth_factor);
    }

    size_t AdamOptimizer::compute_new_capacity(size_t current_capacity, size_t required_size) const {
        if (current_capacity == 0) {
            // First allocation: use initial_capacity if set, otherwise exact fit with some growth
            if (config_.initial_capacity > 0) {
                return std::max(config_.initial_capacity, required_size);
            }
            // Default: allocate 150% of required to avoid immediate reallocation
            return static_cast<size_t>(required_size * 1.5f);
        }

        // Grow by growth_factor (like std::vector uses 1.5x or 2x)
        size_t grown_capacity = static_cast<size_t>(current_capacity * config_.growth_factor);
        return std::max(grown_capacity, required_size);
    }

    const AdamParamState* AdamOptimizer::get_state(ParamType type) const {
        auto name = param_name(type);
        auto it = states_.find(name);
        if (it == states_.end()) {
            return nullptr;
        }
        // NOTE: Returns the state with full capacity tensors
        // Caller should use state->size to know the actual used size
        // The exp_avg/exp_avg_sq tensors may have shape[0] > size due to pre-allocation
        return &it->second;
    }

    int64_t AdamOptimizer::get_step_count(ParamType type) const {
        auto name = param_name(type);
        auto it = states_.find(name);
        if (it == states_.end()) {
            return 0;
        }
        return it->second.step_count;
    }

    void AdamOptimizer::add_new_params(ParamType type, const lfs::core::Tensor& new_values, bool validate) {
        auto& param = get_param(type);
        auto& grad = get_grad(type);

        // Validation: check that new_values has compatible shape
        if (validate) {
            if (new_values.ndim() != param.ndim()) {
                throw std::runtime_error(
                    "add_new_params: new_values rank (" + std::to_string(new_values.ndim()) +
                    ") doesn't match existing parameter rank (" + std::to_string(param.ndim()) + ")"
                );
            }

            // Check that all dimensions except first match
            for (size_t i = 1; i < param.ndim(); i++) {
                if (new_values.shape()[i] != param.shape()[i]) {
                    throw std::runtime_error(
                        "add_new_params: new_values shape mismatch at dimension " + std::to_string(i)
                    );
                }
            }

            // Check device matches
            if (new_values.device() != param.device()) {
                throw std::runtime_error(
                    "add_new_params: new_values device doesn't match existing parameter device"
                );
            }
        }

        size_t n_new = new_values.shape()[0];
        size_t n_current = param.shape()[0];

        // OPTIMIZATION: Use tensor concatenation (requires allocation)
        // NOTE: For a zero-allocation version, SplatData would need to pre-allocate
        // with excess capacity and use slicing. This is a cleaner API but allocates.
        param = lfs::core::Tensor::cat(std::vector<lfs::core::Tensor>{param, new_values}, 0);

        // Extend gradient with zeros
        std::vector<size_t> grad_dims(param.ndim());
        for (size_t i = 0; i < param.ndim(); i++) {
            grad_dims[i] = (i == 0) ? n_new : param.shape()[i];
        }
        auto zeros_grad = lfs::core::Tensor::zeros(lfs::core::TensorShape(grad_dims), param.device());
        grad = lfs::core::Tensor::cat(std::vector<lfs::core::Tensor>{grad, zeros_grad}, 0);

        // Extend optimizer state (this can be optimized with capacity tracking)
        extend_state_for_new_params(type, n_new);

        LOG_DEBUG("add_new_params: Added {} new {} parameters (total: {} -> {})",
                  n_new, param_name(type), n_current, param.shape()[0]);
    }

    void AdamOptimizer::relocate_params_at_indices(ParamType type, const std::vector<int64_t>& indices) {
        if (indices.empty()) return;

        auto& param = get_param(type);

        // Validation: check indices are in bounds
        for (auto idx : indices) {
            if (idx < 0 || static_cast<size_t>(idx) >= param.shape()[0]) {
                throw std::runtime_error(
                    "relocate_params_at_indices: index " + std::to_string(idx) +
                    " out of bounds [0, " + std::to_string(param.shape()[0]) + ")"
                );
            }
        }

        // Copy indices to GPU once, then use fast GPU version
        int64_t* indices_device_ptr;
        cudaMalloc(&indices_device_ptr, indices.size() * sizeof(int64_t));
        cudaMemcpy(indices_device_ptr, indices.data(), indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

        relocate_params_at_indices_gpu(type, indices_device_ptr, indices.size());

        cudaFree(indices_device_ptr);
    }

    void AdamOptimizer::relocate_params_at_indices_gpu(ParamType type, const int64_t* indices_device, size_t n_indices) {
        if (n_indices == 0) return;

        auto& param = get_param(type);
        auto& grad = get_grad(type);
        auto name = param_name(type);

        // Calculate row size for gradients
        auto grad_shape = grad.shape();
        int grad_row_size = 1;
        for (size_t i = 1; i < grad_shape.rank(); i++) {
            grad_row_size *= grad_shape[i];
        }

        // Zero out gradients using batched GPU kernel (FAST!)
        fast_lfs::optimizer::zero_rows_at_indices(
            grad.template ptr<float>(),
            indices_device,
            n_indices,
            grad_row_size
        );

        // Ensure optimizer state exists
        if (states_.find(name) == states_.end()) {
            LOG_DEBUG("State for {} not initialized yet, skipping reset", name);
            return;
        }

        auto& state = states_[name];

        // Calculate row size for optimizer state
        auto state_shape = state.exp_avg.shape();
        int state_row_size = 1;
        for (size_t i = 1; i < state_shape.rank(); i++) {
            state_row_size *= state_shape[i];
        }

        // Zero out optimizer state using batched GPU kernel (FAST!)
        fast_lfs::optimizer::zero_rows_at_indices(
            state.exp_avg.template ptr<float>(),
            indices_device,
            n_indices,
            state_row_size
        );

        fast_lfs::optimizer::zero_rows_at_indices(
            state.exp_avg_sq.template ptr<float>(),
            indices_device,
            n_indices,
            state_row_size
        );

        LOG_DEBUG("relocate_params_at_indices_gpu: Reset state and gradients for {} at {} indices (batched GPU kernel)",
                  name, n_indices);
    }

} // namespace lfs::training
