/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/strategies/mcmc_new.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "optimizers/fused_adam.hpp"
#include "optimizers/scheduler.hpp"
#include "strategy_utils_new.hpp"
#include "training_kernels.cuh"
#include <algorithm>
#include <chrono>
#include <random>

namespace gs::training {

    MCMCNew::MCMCNew(gs::SplatDataNew&& splat_data)
        : _splat_data(std::move(splat_data)) {
    }

    MCMCNew::~MCMCNew() = default;

    gs::Tensor MCMCNew::multinomial_sample(const gs::Tensor& weights, int n, bool replacement) {
        if (!weights.is_valid() || weights.numel() == 0) {
            LOG_ERROR("multinomial_sample: invalid or empty weights tensor");
            return gs::Tensor();
        }

        if (weights.ndim() != 1) {
            LOG_ERROR("multinomial_sample: weights must be 1D tensor, got {}D", weights.ndim());
            return gs::Tensor();
        }

        const int64_t num_elements = weights.size(0);

        if (n <= 0) {
            LOG_ERROR("multinomial_sample: n must be positive, got {}", n);
            return gs::Tensor();
        }

        if (!replacement && n > num_elements) {
            LOG_ERROR("multinomial_sample: cannot sample {} items without replacement from {} elements",
                      n, num_elements);
            return gs::Tensor();
        }

        // For now, we'll implement manual sampling using cumulative distribution
        // This works for both small and large arrays

        // Normalize weights to sum to 1
        auto weights_sum = weights.sum();
        float sum_val = weights_sum.item();

        if (sum_val <= 0.0f) {
            LOG_ERROR("multinomial_sample: weights sum to non-positive value {}", sum_val);
            return gs::Tensor();
        }

        auto weights_normalized = weights.div(sum_val);

        // Move to CPU for sampling (we'll do CPU-based sampling for now)
        auto weights_cpu = weights_normalized.cpu();

        // Compute cumulative sum
        auto cumsum = weights_cpu.cumsum(0);

        // Get accessor for efficient indexing
        const float* cumsum_ptr = cumsum.ptr<float>();
        if (!cumsum_ptr) {
            LOG_ERROR("multinomial_sample: failed to get cumsum pointer");
            return gs::Tensor();
        }

        // Sample indices
        std::vector<int> sampled_indices;
        sampled_indices.reserve(n);

        // Use the tensor library's seeded random generator for reproducibility
        // TODO: Use RandomGenerator::instance().get_next_cuda_seed() or similar to ensure
        // each call gets a unique seed. Currently using the global seed which means
        // repeated calls with the same seed setting will produce identical results.
        uint64_t seed = gs::RandomGenerator::instance().get_seed();
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        for (int i = 0; i < n; ++i) {
            float u = dis(gen);

            // Binary search for the index where cumsum[idx] >= u
            int64_t left = 0;
            int64_t right = num_elements - 1;
            int64_t idx = 0;

            while (left <= right) {
                int64_t mid = left + (right - left) / 2;
                if (cumsum_ptr[mid] < u) {
                    left = mid + 1;
                } else {
                    idx = mid;
                    right = mid - 1;
                }
            }

            sampled_indices.push_back(static_cast<int>(idx));
        }

        // Create result tensor on CPU first as Int32, then move to same device as input
        auto result = gs::Tensor::from_vector(sampled_indices,
                                              gs::TensorShape{static_cast<size_t>(n)},
                                              gs::Device::CPU);

        // Convert to Int64
        result = result.to(gs::DataType::Int64);

        // Move to same device as input weights
        if (weights.device() == gs::Device::CUDA) {
            result = result.cuda();
        }

        return result;
    }

    void MCMCNew::initialize(const gs::param::OptimizationParameters& params) {
        // Store parameters
        _params = std::make_unique<const gs::param::OptimizationParameters>(params);

        // Ensure all tensors are on CUDA
        if (_splat_data.means_raw().device() != gs::Device::CUDA) {
            _splat_data.means_raw() = _splat_data.means_raw().cuda();
        }
        if (_splat_data.scaling_raw().device() != gs::Device::CUDA) {
            _splat_data.scaling_raw() = _splat_data.scaling_raw().cuda();
        }
        if (_splat_data.rotation_raw().device() != gs::Device::CUDA) {
            _splat_data.rotation_raw() = _splat_data.rotation_raw().cuda();
        }
        if (_splat_data.opacity_raw().device() != gs::Device::CUDA) {
            _splat_data.opacity_raw() = _splat_data.opacity_raw().cuda();
        }
        if (_splat_data.sh0_raw().device() != gs::Device::CUDA) {
            _splat_data.sh0_raw() = _splat_data.sh0_raw().cuda();
        }
        if (_splat_data.shN_raw().device() != gs::Device::CUDA) {
            _splat_data.shN_raw() = _splat_data.shN_raw().cuda();
        }

        // Initialize noise buffer (N * 3 elements for position noise)
        noise_buffer_ = gs::Tensor::empty({static_cast<size_t>(_splat_data.size() * 3)}, gs::Device::CUDA);

        // Allocate gradient tensors
        _splat_data.ensure_grad_allocated();

        // Initialize binomial coefficients for relocation
        const int n_max = 51;

        // Compute binomial coefficients on CPU
        std::vector<float> binom_data(n_max * n_max, 0.0f);
        for (int n = 0; n < n_max; ++n) {
            for (int k = 0; k <= n; ++k) {
                float binom = 1.0f;
                for (int i = 0; i < k; ++i) {
                    binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
                }
                binom_data[n * n_max + k] = binom;
            }
        }

        // Create tensor from vector and reshape
        _binoms = gs::Tensor::from_vector(binom_data, {static_cast<size_t>(n_max * n_max)}, gs::Device::CPU);
        _binoms = _binoms.reshape({static_cast<size_t>(n_max), static_cast<size_t>(n_max)});
        _binoms = _binoms.cuda();

        // Create optimizer and scheduler using helper functions
        _optimizer = create_optimizer_new(_splat_data, *_params);
        _scheduler = create_scheduler_new(*_params, _optimizer.get(), 0);

        LOG_INFO("MCMCNew initialized with {} Gaussians", _splat_data.size());
    }

    void MCMCNew::post_backward(int iter, RenderOutput& render_output) {
        // Increment SH degree every sh_degree_interval iterations
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data.increment_sh_degree();
        }

        // Refine Gaussians (relocate dead + add new)
        if (is_refining(iter)) {
            // Relocate dead Gaussians
            int n_relocated = relocate_gs();
            if (n_relocated > 0) {
                LOG_DEBUG("Relocated {} dead Gaussians at iteration {}", n_relocated, iter);
            }

            // Add new Gaussians
            int n_added = add_new_gs();
            if (n_added > 0) {
                LOG_DEBUG("Added {} new Gaussians at iteration {}", n_added, iter);
            }
        }

        // Inject noise to positions
        inject_noise();

#ifdef _WIN32
        // Windows doesn't support CUDACachingAllocator expandable_segments
        if (iter % 10 == 0) {
            // Note: We're using gs::Tensor instead of torch, so we use CUDA runtime directly
            cudaError_t err = cudaDeviceSynchronize();
            if (err == cudaSuccess) {
                // Reset CUDA memory pool if needed
                // This is less critical with our custom Tensor library
            }
        }
#endif
    }

    void MCMCNew::step(int iter) {
        if (iter < _params->iterations) {
            // Perform optimization step
            _optimizer->step(iter);

            // Zero gradients manually
            _splat_data.zero_grad();

            // Update learning rate scheduler
            _scheduler->step();
        }
    }

    void MCMCNew::remove_gaussians(const gs::Tensor& mask) {
        // Check if there are any Gaussians to remove
        // Count True values by checking nonzero indices
        auto remove_indices = mask.nonzero();
        if (remove_indices.ndim() == 2 && remove_indices.size(1) == 1) {
            remove_indices = remove_indices.squeeze(-1);
        }
        int num_to_remove = remove_indices.numel();

        if (num_to_remove == 0) {
            LOG_DEBUG("No Gaussians to remove");
            return;
        }

        LOG_DEBUG("MCMC: Removing {} Gaussians", num_to_remove);

        // Get indices to keep (logical_not of mask, then nonzero)
        auto keep_mask = mask.logical_not();
        auto keep_indices = keep_mask.nonzero();

        // Handle shape: nonzero returns [N, 1] for 1D input, squeeze to [N]
        if (keep_indices.ndim() == 2 && keep_indices.size(1) == 1) {
            keep_indices = keep_indices.squeeze(-1);
        }

        // Subset all parameters using index_select
        _splat_data.means_raw() = _splat_data.means_raw().index_select(0, keep_indices);
        _splat_data.sh0_raw() = _splat_data.sh0_raw().index_select(0, keep_indices);
        _splat_data.shN_raw() = _splat_data.shN_raw().index_select(0, keep_indices);
        _splat_data.scaling_raw() = _splat_data.scaling_raw().index_select(0, keep_indices);
        _splat_data.rotation_raw() = _splat_data.rotation_raw().index_select(0, keep_indices);
        _splat_data.opacity_raw() = _splat_data.opacity_raw().index_select(0, keep_indices);

        // Reallocate gradients for new size
        _splat_data.ensure_grad_allocated();

        // Recreate optimizer with new sizes
        // We reset optimizer state since subsetting is complex
        if (_optimizer && _params) {
            // Save current learning rates
            std::vector<double> old_lrs;
            for (size_t i = 0; i < _optimizer->param_groups().size(); ++i) {
                old_lrs.push_back(_optimizer->get_lr(i));
            }

            // Recreate optimizer using helper function
            _optimizer = create_optimizer_new(_splat_data, *_params);

            // Restore learning rates
            for (size_t i = 0; i < old_lrs.size() && i < _optimizer->param_groups().size(); ++i) {
                _optimizer->set_lr(old_lrs[i], i);
            }
        }

        // Reallocate noise buffer for new size
        noise_buffer_ = gs::Tensor::empty({static_cast<size_t>(_splat_data.size() * 3)}, gs::Device::CUDA);

        LOG_DEBUG("After removal: {} Gaussians remaining", _splat_data.size());
    }

    int MCMCNew::relocate_gs() {
        if (!_params) {
            LOG_ERROR("relocate_gs: params not initialized");
            return 0;
        }

        // Get opacities (handle both [N] and [N, 1] shapes)
        auto opacities = _splat_data.get_opacity();
        if (opacities.ndim() == 2 && opacities.size(1) == 1) {
            opacities = opacities.squeeze(-1);
        }

        // Find dead Gaussians (low opacity or invalid rotation)
        auto rotation_raw = _splat_data.rotation_raw();
        auto rot_norm_sq = (rotation_raw * rotation_raw).sum(-1);
        auto dead_mask = (opacities.le(_params->min_opacity)).logical_or(rot_norm_sq.lt(1e-8f));

        auto dead_indices = dead_mask.nonzero();
        if (dead_indices.ndim() == 2 && dead_indices.size(1) == 1) {
            dead_indices = dead_indices.squeeze(-1);
        }

        int n_dead = dead_indices.numel();
        if (n_dead == 0)
            return 0;

        // Find alive Gaussians
        auto alive_mask = dead_mask.logical_not();
        auto alive_indices = alive_mask.nonzero();
        if (alive_indices.ndim() == 2 && alive_indices.size(1) == 1) {
            alive_indices = alive_indices.squeeze(-1);
        }

        if (alive_indices.numel() == 0)
            return 0;

        // Sample from alive Gaussians based on opacity
        auto probs = opacities.index_select(0, alive_indices);
        auto sampled_idxs_local = multinomial_sample(probs, n_dead, true);
        auto sampled_idxs = alive_indices.index_select(0, sampled_idxs_local);

        // Get parameters for sampled Gaussians
        auto sampled_opacities = opacities.index_select(0, sampled_idxs);
        auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

        // Count occurrences of each sampled index
        // Start with ones, then add 1 for each occurrence
        auto ratios = gs::Tensor::ones({opacities.size(0)}, opacities.device(), gs::DataType::Int32);
        auto ones_for_add = gs::Tensor::ones({sampled_idxs.size(0)}, opacities.device(), gs::DataType::Int32);
        ratios.index_add_(0, sampled_idxs, ones_for_add);
        ratios = ratios.index_select(0, sampled_idxs);

        // Clamp ratios to n_max
        const int n_max = static_cast<int>(_binoms.size(0));
        ratios = ratios.clamp_max(n_max);

        // Ensure ratios is contiguous and Int32
        if (!ratios.is_contiguous()) {
            ratios = ratios.contiguous();
        }

        // Call the CUDA relocation kernel
        auto new_opacities = gs::Tensor::empty_like(sampled_opacities);
        auto new_scales = gs::Tensor::empty_like(sampled_scales);

        launch_relocation_kernel(
            sampled_opacities.ptr<float>(),
            sampled_scales.ptr<float>(),
            ratios.ptr<int>(),
            _binoms.ptr<float>(),
            sampled_opacities.size(0),
            n_max,
            new_opacities.ptr<float>(),
            new_scales.ptr<float>(),
            nullptr // default stream
        );

        // Clamp new opacities
        new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);

        // Update parameters for sampled indices - apply new opacities/scales
        // Convert opacities to logit space for storage
        auto new_opacity_raw = gs::Tensor::empty_like(new_opacities);
        // Logit: log(x / (1 - x))
        auto one_minus_op = (new_opacities.mul(-1.0f)).add(1.0f);
        new_opacity_raw = (new_opacities.div(one_minus_op)).log();

        // Convert scales to log space
        auto new_scales_raw = new_scales.log();

        // Update sampled Gaussians with new values
        if (_splat_data.opacity_raw().ndim() == 2) {
            // [N, 1] shape
            new_opacity_raw = new_opacity_raw.unsqueeze(-1);
        }

        // Use index_put_ to update the sampled indices
        _splat_data.opacity_raw().index_copy_(0, sampled_idxs, new_opacity_raw);
        _splat_data.scaling_raw().index_copy_(0, sampled_idxs, new_scales_raw);

        // Copy parameters from sampled to dead indices
        _splat_data.means_raw().index_copy_(0, dead_indices, _splat_data.means_raw().index_select(0, sampled_idxs));
        _splat_data.sh0_raw().index_copy_(0, dead_indices, _splat_data.sh0_raw().index_select(0, sampled_idxs));
        _splat_data.shN_raw().index_copy_(0, dead_indices, _splat_data.shN_raw().index_select(0, sampled_idxs));
        _splat_data.scaling_raw().index_copy_(0, dead_indices, _splat_data.scaling_raw().index_select(0, sampled_idxs));
        _splat_data.rotation_raw().index_copy_(0, dead_indices, _splat_data.rotation_raw().index_select(0, sampled_idxs));
        _splat_data.opacity_raw().index_copy_(0, dead_indices, _splat_data.opacity_raw().index_select(0, sampled_idxs));

        // Update optimizer states for sampled indices (placeholder for now)
        if (_optimizer) {
            for (int i = 0; i < 6; ++i) {
                update_optimizer_for_relocate(_optimizer.get(), sampled_idxs, dead_indices, i);
            }
        }

        return n_dead;
    }

    int MCMCNew::add_new_gs() {
        if (!_optimizer || !_params) {
            LOG_ERROR("add_new_gs: optimizer or params not initialized");
            return 0;
        }

        const int current_n = _splat_data.size();
        const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
        const int n_new = std::max(0, n_target - current_n);

        if (n_new == 0)
            return 0;

        // Get opacities and handle both [N] and [N, 1] shapes
        auto opacities = _splat_data.get_opacity();
        if (opacities.ndim() == 2 && opacities.size(1) == 1) {
            opacities = opacities.squeeze(-1);
        }

        // Sample new Gaussians based on opacity
        auto probs = opacities.flatten();
        auto sampled_idxs = multinomial_sample(probs, n_new, true);

        // Get parameters for sampled Gaussians
        auto sampled_opacities = opacities.index_select(0, sampled_idxs);
        auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

        // Count occurrences: start with zeros, add 1 for each occurrence
        auto ratios = gs::Tensor::zeros({opacities.size(0)}, opacities.device(), gs::DataType::Float32);
        auto ones_for_add = gs::Tensor::ones({sampled_idxs.size(0)}, opacities.device(), gs::DataType::Float32);
        ratios.index_add_(0, sampled_idxs, ones_for_add);
        ratios = ratios.index_select(0, sampled_idxs).add(1.0f);

        // Clamp and convert to int
        const int n_max = static_cast<int>(_binoms.size(0));
        ratios = ratios.clamp(1.0f, static_cast<float>(n_max)).to(gs::DataType::Int32);
        if (!ratios.is_contiguous()) {
            ratios = ratios.contiguous();
        }

        // Call relocation kernel
        auto new_opacities = gs::Tensor::empty_like(sampled_opacities);
        auto new_scales = gs::Tensor::empty_like(sampled_scales);

        launch_relocation_kernel(
            sampled_opacities.ptr<float>(),
            sampled_scales.ptr<float>(),
            ratios.ptr<int>(),
            _binoms.ptr<float>(),
            sampled_opacities.size(0),
            n_max,
            new_opacities.ptr<float>(),
            new_scales.ptr<float>(),
            nullptr);

        // Clamp new opacities
        new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);

        // Update existing Gaussians FIRST (before concatenation)
        // Convert to logit space for storage
        auto new_opacity_raw = (new_opacities.div((new_opacities.mul(-1.0f)).add(1.0f))).log();
        auto new_scales_raw = new_scales.log();

        if (_splat_data.opacity_raw().ndim() == 2) {
            new_opacity_raw = new_opacity_raw.unsqueeze(-1);
        }

        _splat_data.opacity_raw().index_copy_(0, sampled_idxs, new_opacity_raw);
        _splat_data.scaling_raw().index_copy_(0, sampled_idxs, new_scales_raw);

        // Prepare new Gaussians to concatenate
        auto new_means = _splat_data.means_raw().index_select(0, sampled_idxs);
        auto new_sh0 = _splat_data.sh0_raw().index_select(0, sampled_idxs);
        auto new_shN = _splat_data.shN_raw().index_select(0, sampled_idxs);
        auto new_scaling = _splat_data.scaling_raw().index_select(0, sampled_idxs);
        auto new_rotation = _splat_data.rotation_raw().index_select(0, sampled_idxs);
        auto new_opacity = _splat_data.opacity_raw().index_select(0, sampled_idxs);

        // Concatenate all parameters
        _splat_data.means_raw() = gs::Tensor::cat({_splat_data.means_raw(), new_means}, 0);
        _splat_data.sh0_raw() = gs::Tensor::cat({_splat_data.sh0_raw(), new_sh0}, 0);
        _splat_data.shN_raw() = gs::Tensor::cat({_splat_data.shN_raw(), new_shN}, 0);
        _splat_data.scaling_raw() = gs::Tensor::cat({_splat_data.scaling_raw(), new_scaling}, 0);
        _splat_data.rotation_raw() = gs::Tensor::cat({_splat_data.rotation_raw(), new_rotation}, 0);
        _splat_data.opacity_raw() = gs::Tensor::cat({_splat_data.opacity_raw(), new_opacity}, 0);

        // Ensure gradients are allocated for new size
        _splat_data.ensure_grad_allocated();

        // Recreate optimizer with new sizes
        std::vector<double> old_lrs;
        for (size_t i = 0; i < _optimizer->param_groups().size(); ++i) {
            old_lrs.push_back(_optimizer->get_lr(i));
        }

        // Recreate optimizer using helper function
        _optimizer = create_optimizer_new(_splat_data, *_params);

        // Restore learning rates
        for (size_t i = 0; i < old_lrs.size() && i < _optimizer->param_groups().size(); ++i) {
            _optimizer->set_lr(old_lrs[i], i);
        }

        // Reallocate noise buffer for new size
        noise_buffer_ = gs::Tensor::empty({static_cast<size_t>(_splat_data.size() * 3)}, gs::Device::CUDA);

        LOG_DEBUG("Added {} new Gaussians (total: {})", n_new, _splat_data.size());
        return n_new;
    }

    void MCMCNew::inject_noise() {
        if (!_optimizer) {
            LOG_ERROR("inject_noise: optimizer not initialized");
            return;
        }

        // Get current learning rate from optimizer (after scheduler has updated it)
        double current_lr = _optimizer->get_lr(0) * _noise_lr;

        int N = _splat_data.size();
        size_t noise_elements = N * 3; // [N, 3]

        // Ensure noise buffer is allocated
        if (!noise_buffer_.is_valid() || noise_buffer_.numel() != noise_elements) {
            noise_buffer_ = gs::Tensor::empty({noise_elements}, gs::Device::CUDA);
        }

        // Generate random seed from current time
        unsigned long long seed = std::chrono::steady_clock::now().time_since_epoch().count();

        // Generate normal distribution noise
        launch_generate_normal_noise(
            noise_buffer_.ptr<float>(),
            noise_elements,
            seed,
            nullptr); // default stream

        // Inject noise to gaussians
        launch_inject_noise_to_gaussians(
            _splat_data.opacity_raw().ptr<float>(),
            _splat_data.scaling_raw().ptr<float>(),
            _splat_data.rotation_raw().ptr<float>(),
            noise_buffer_.ptr<float>(),
            _splat_data.means_raw().ptr<float>(),
            N,
            static_cast<float>(current_lr),
            nullptr); // default stream

        // Synchronize to ensure completion
        cudaDeviceSynchronize();
    }

    bool MCMCNew::is_refining(int iter) const {
        if (!_params)
            return false;
        return (iter < _params->stop_refine &&
                iter > _params->start_refine &&
                iter % _params->refine_every == 0);
    }

    void MCMCNew::update_optimizer_for_relocate(FusedAdam* optimizer,
                                                const gs::Tensor& sampled_indices,
                                                const gs::Tensor& dead_indices,
                                                int param_position) {
        // Placeholder - optimizer will adapt naturally to parameter changes
        // In a production system, you'd want to reset optimizer state for the changed indices
    }

} // namespace gs::training
