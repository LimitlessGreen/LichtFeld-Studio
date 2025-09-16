/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <numeric>

namespace gs::tensor {

    // ============= Range Operations =============
    Tensor arange(float end) {
        return arange(0, end, 1);
    }

    Tensor arange(float start, float end, float step) {
        // Handle negative step
        int n;
        if (step > 0) {
            if (start >= end) {
                LOG_ERROR("Invalid range: start={}, end={}, step={} (start >= end with positive step)", start, end, step);
                return Tensor();
            }
            n = static_cast<int>((end - start) / step);
        } else if (step < 0) {
            if (start <= end) {
                LOG_ERROR("Invalid range: start={}, end={}, step={} (start <= end with negative step)", start, end, step);
                return Tensor();
            }
            n = static_cast<int>((end - start) / step);
        } else {
            LOG_ERROR("Invalid range: step cannot be zero");
            return Tensor();
        }

        if (n <= 0) {
            LOG_ERROR("Invalid range: start={}, end={}, step={}", start, end, step);
            return Tensor();
        }

        auto t = Tensor::empty({static_cast<size_t>(n)}, Device::CUDA, DataType::Float32);

        // Fill with sequence on CPU then copy to GPU
        std::vector<float> vec(n);
        for (int i = 0; i < n; ++i) {
            vec[i] = start + i * step;
        }

        cudaMemcpy(t.ptr<float>(), vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        return t;
    }

    Tensor linspace(float start, float end, size_t steps) {
        if (steps == 0) {
            LOG_ERROR("linspace requires steps > 0");
            return Tensor();
        }

        auto t = Tensor::empty({steps}, Device::CUDA, DataType::Float32);
        float step = (steps == 1) ? 0 : (end - start) / (steps - 1);

        std::vector<float> vec(steps);
        for (size_t i = 0; i < steps; ++i) {
            vec[i] = start + i * step;
        }

        cudaMemcpy(t.ptr<float>(), vec.data(), steps * sizeof(float), cudaMemcpyHostToDevice);

        return t;
    }

    // ============= Stack/Concatenate Operations =============
    Tensor stack(std::vector<Tensor>&& tensors, int dim) {
        if (tensors.empty()) {
            LOG_ERROR("Cannot stack empty tensor list");
            return Tensor();
        }

        // Check all shapes match
        auto shape = tensors[0].shape();
        auto device = tensors[0].device();
        auto dtype = tensors[0].dtype();

        for (const auto& t : tensors) {
            if (t.shape() != shape) {
                LOG_ERROR("Cannot stack tensors with different shapes");
                return Tensor();
            }
            if (t.device() != device) {
                LOG_ERROR("Cannot stack tensors on different devices");
                return Tensor();
            }
            if (t.dtype() != dtype) {
                LOG_ERROR("Cannot stack tensors with different dtypes");
                return Tensor();
            }
        }

        // Create new shape with extra dimension
        std::vector<size_t> new_dims = shape.dims();
        if (dim < 0)
            dim = new_dims.size() + dim + 1;
        new_dims.insert(new_dims.begin() + dim, tensors.size());

        auto result = Tensor::empty(TensorShape(new_dims), device, dtype);

        // Copy each tensor
        size_t offset = 0;
        size_t stride = shape.elements() * dtype_size(dtype);

        for (const auto& t : tensors) {
            cudaMemcpy(
                static_cast<char*>(result.raw_ptr()) + offset,
                t.raw_ptr(),
                stride,
                cudaMemcpyDeviceToDevice);
            offset += stride;
        }

        return result;
    }

    Tensor cat(std::vector<Tensor>&& tensors, int dim) {
        if (tensors.empty()) {
            LOG_ERROR("Cannot concatenate empty tensor list");
            return Tensor();
        }

        // Get common properties from first tensor
        auto device = tensors[0].device();
        auto dtype = tensors[0].dtype();

        // For simplicity, implement basic concatenation along dim 0
        if (dim != 0) {
            LOG_ERROR("Concatenation only implemented for dim=0");
            return Tensor();
        }

        // Calculate total size
        size_t total_rows = 0;
        auto shape = tensors[0].shape();
        for (const auto& t : tensors) {
            total_rows += t.shape()[0];

            // Check other dimensions match
            for (size_t i = 1; i < shape.rank(); ++i) {
                if (t.shape()[i] != shape[i]) {
                    LOG_ERROR("Dimension mismatch in concatenation");
                    return Tensor();
                }
            }

            // Check device and dtype match
            if (t.device() != device) {
                LOG_ERROR("Cannot concatenate tensors on different devices");
                return Tensor();
            }
            if (t.dtype() != dtype) {
                LOG_ERROR("Cannot concatenate tensors with different dtypes");
                return Tensor();
            }
        }

        // Create output shape
        std::vector<size_t> new_dims = shape.dims();
        new_dims[0] = total_rows;

        auto result = Tensor::empty(TensorShape(new_dims), device, dtype);

        // Copy each tensor
        size_t offset = 0;
        for (const auto& t : tensors) {
            size_t bytes = t.bytes();
            cudaMemcpy(
                static_cast<char*>(result.raw_ptr()) + offset,
                t.raw_ptr(),
                bytes,
                cudaMemcpyDeviceToDevice);
            offset += bytes;
        }

        return result;
    }

    // ============= Utility Functions =============
    bool check_valid(const Tensor& t, const std::string& name) {
        if (!t.is_valid()) {
            LOG_ERROR("Invalid tensor: {}", name);
            return false;
        }
        return true;
    }

    void assert_shape(const Tensor& t, TensorShape expected, const std::string& name) {
        if (t.shape() != expected) {
            LOG_ERROR("Shape mismatch for {}: expected {}, got {}",
                      name, expected.str(), t.shape().str());
        }
    }

    // ============= Builder Pattern =============
    Tensor TensorBuilder::build() {
        Tensor t;
        if (fill_value.has_value()) {
            t = Tensor::full(shape, *fill_value, device, dtype);
        } else {
            t = Tensor::empty(shape, device, dtype);
        }

        if (check_finite) {
            t.assert_finite();
        }

        LOG_TRACE("Built tensor: {}", t.str());
        return t;
    }

    // ============= Safe Operations =============
    Tensor SafeOps::divide(const Tensor& a, const Tensor& b, float eps) {
        return a / (b + eps);
    }

    Tensor SafeOps::log(const Tensor& t, float eps) {
        return (t + eps).log();
    }

    Tensor SafeOps::sqrt(const Tensor& t, float eps) {
        return (t + eps).sqrt();
    }

    // ============= Memory Info =============
    MemoryInfo MemoryInfo::cuda() {
        MemoryInfo info{};
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        info.free_bytes = free;
        info.reserved_bytes = total;
        info.allocated_bytes = total - free;
        return info;
    }

    void MemoryInfo::log() const {
        LOG_INFO("CUDA Memory: {:.2f}GB allocated, {:.2f}GB free",
                 allocated_bytes / 1e9, free_bytes / 1e9);
    }

    // ============= Batch Processing =============
    template <typename Func>
    Tensor apply_batched(const Tensor& input, size_t batch_size, Func func) {
        auto batches = Tensor::split_batch(input, batch_size);
        std::vector<Tensor> results;
        results.reserve(batches.size());

        for (auto& batch : batches) {
            results.push_back(func(batch));
        }

        return cat(std::move(results), 0);
    }

    // Explicit instantiations for common function types
    template Tensor apply_batched<std::function<Tensor(const Tensor&)>>(
        const Tensor&, size_t, std::function<Tensor(const Tensor&)>);

} // namespace gs::tensor

// ============= Functional Programming Support =============
namespace gs::tensor::functional {

    template <typename Func>
    Tensor map(const Tensor& input, Func func) {
        // Use Tensor::empty with all three parameters
        auto output = Tensor::empty(input.shape(), input.device(), input.dtype());

        // For now, simple CPU implementation
        auto in_vec = input.to_vector();
        std::vector<float> out_vec(in_vec.size());

        std::transform(in_vec.begin(), in_vec.end(), out_vec.begin(), func);

        cudaMemcpy(output.ptr<float>(), out_vec.data(),
                   out_vec.size() * sizeof(float),
                   cudaMemcpyHostToDevice);

        return output;
    }

    template <typename Func>
    float reduce(const Tensor& input, float init, Func func) {
        auto vec = input.to_vector();
        return std::accumulate(vec.begin(), vec.end(), init, func);
    }

    template <typename Pred>
    Tensor filter(const Tensor& input, Pred predicate) {
        auto vec = input.to_vector();
        std::vector<float> mask(vec.size());

        std::transform(vec.begin(), vec.end(), mask.begin(),
                       [&](float v) { return predicate(v) ? 1.0f : 0.0f; });

        // Use Tensor::empty with all three parameters
        auto result = Tensor::empty(input.shape(), input.device(), input.dtype());
        cudaMemcpy(result.ptr<float>(), mask.data(),
                   mask.size() * sizeof(float),
                   cudaMemcpyHostToDevice);

        return result;
    }

    // ============= EXPLICIT INSTANTIATIONS FOR ALL TYPES USED IN TESTS =============

    // Basic function types
    template Tensor map<std::function<float(float)>>(const Tensor&, std::function<float(float)>);
    template float reduce<std::plus<float>>(const Tensor&, float, std::plus<float>);
    template float reduce<std::multiplies<float>>(const Tensor&, float, std::multiplies<float>);
    template Tensor filter<std::function<bool(float)>>(const Tensor&, std::function<bool(float)>);

} // namespace gs::tensor::functional

// ============= Explicit instantiations for test functors =============
// These are the function objects used in test_tensor_advanced.cpp

// Functor for squaring
struct SquareFunc {
    float operator()(float x) const { return x * x; }
};

// Functor for filtering positive values
struct PositiveFilter {
    bool operator()(float x) const { return x > 0; }
};

// Functor for profiling test - needs to be in gs namespace to use gs::Tensor
struct ProfileOp {
    gs::Tensor operator()(const gs::Tensor& t) const {
        return t.add(1.0f).mul(2.0f).sub(1.0f);
    }
};

// Now instantiate the templates for these functors
namespace gs::tensor::functional {
    template Tensor map<SquareFunc>(const Tensor&, SquareFunc);
    template Tensor filter<PositiveFilter>(const Tensor&, PositiveFilter);
} // namespace gs::tensor::functional

// ============= Template implementation and instantiation for Tensor::timed =============
namespace gs {
    // Template implementation for Tensor::timed (must be defined before instantiation)
    template <typename Func>
    auto Tensor::timed(const std::string& name, Func func) -> decltype(func(*this)) {
        if (profiling_enabled_) {
            TensorTimer timer(name);
            return func(*this);
        }
        return func(*this);
    }

    // Explicit instantiation for ProfileOp
    template auto Tensor::timed<ProfileOp>(const std::string&, ProfileOp) -> decltype(std::declval<ProfileOp>()(*std::declval<Tensor*>()));
} // namespace gs

// ============= Old test functors that might still be used =============
struct GreaterThanTwo {
    bool operator()(float x) const { return x > 2; }
};

struct AddOne {
    gs::Tensor operator()(const gs::Tensor& t) const { return t.add(1); }
};

struct MulTwo {
    gs::Tensor operator()(const gs::Tensor& t) const { return t.mul(2); }
};

struct SubThree {
    gs::Tensor operator()(const gs::Tensor& t) const { return t.sub(3); }
};

namespace gs::tensor::functional {
    // Instantiate for old test functors if they're still used
    template Tensor filter<GreaterThanTwo>(const Tensor&, GreaterThanTwo);
} // namespace gs::tensor::functional
