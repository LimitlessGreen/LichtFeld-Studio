/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_api.h"
#include "adam.h"
#include "cuda_utils.h"

namespace fast_gs::optimizer {

void adam_step_raw(
    float* param,
    float* exp_avg,
    float* exp_avg_sq,
    const float* param_grad,
    const int n_elements,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction1_rcp,
    const float bias_correction2_sqrt_rcp) {
    
    // Validate pointers
    CHECK_CUDA_PTR(param, "param");
    CHECK_CUDA_PTR(exp_avg, "exp_avg");
    CHECK_CUDA_PTR(exp_avg_sq, "exp_avg_sq");
    CHECK_CUDA_PTR(param_grad, "param_grad");
    
    // Validate parameters
    if (n_elements <= 0) {
        throw std::runtime_error("n_elements must be positive");
    }
    
    // Call the actual implementation
    adam_step(
        param,
        exp_avg,
        exp_avg_sq,
        param_grad,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1_rcp,
        bias_correction2_sqrt_rcp);
}

} // namespace fast_gs::optimizer
