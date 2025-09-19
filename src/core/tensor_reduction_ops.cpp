/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
* SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include <cmath>

namespace gs {

    // These are now just convenience functions for scalar reduction
    // The tensor reduction goes through reduce()

    float Tensor::sum_scalar() const {
        return sum().item();
    }

    float Tensor::mean_scalar() const {
        return mean().item();
    }

    float Tensor::min_scalar() const {
        return min().item();
    }

    float Tensor::max_scalar() const {
        return max().item();
    }

    float Tensor::std_scalar(float eps) const {
        return std().item();
    }

    float Tensor::var_scalar(float eps) const {
        return var().item();
    }

    std::pair<float, float> Tensor::minmax() const {
        return {min_scalar(), max_scalar()};
    }

    float Tensor::norm(float p) const {
        if (p == 2.0f) {
            auto sq = mul(*this);  // square
            return std::sqrt(sq.sum_scalar());
        } else if (p == 1.0f) {
            return abs().sum_scalar();
        } else if (std::isinf(p)) {
            return abs().max_scalar();
        } else {
            // General p-norm
            auto powered = abs().pow(p);
            return std::pow(powered.sum_scalar(), 1.0f / p);
        }
    }

    float Tensor::item() const {
        if (numel() != 1) {
            LOG_ERROR("item() only works for single-element tensors");
            return 0.0f;
        }

        if (device_ == Device::CUDA) {
            float result;
            cudaMemcpy(&result, ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
            return result;
        } else {
            return *ptr<float>();
        }
    }

} // namespace gs