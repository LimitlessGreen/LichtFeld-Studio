/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

namespace gs::tensor_ops {

    // ============================================================================
    // THRUST POLICY HELPER - Stream support for Thrust operations
    // ============================================================================

    template<typename Func>
    inline void run_with_thrust_policy(cudaStream_t stream, Func&& func) {
        if (stream) {
            func(thrust::cuda::par.on(stream));
        } else {
            func(thrust::cuda::par);
        }
    }

    // ============================================================================
    // GENERIC OPERATIONS - Header-only for optimal inlining and specialization
    // ============================================================================

    // These are in a header (not .cu) to allow:
    // 1. Compiler to inline Thrust operations across translation units
    // 2. Template specialization for specific functors (e.g., composed_unary_op)
    // 3. Better devirtualization of functors at compile-time
    // 4. No need for explicit template instantiations
    // 5. Works with expression template fusion (composed functors)

    /**
     * Binary operation: Applies binary functor element-wise to two arrays
     * Supports different input/output types (e.g., float -> bool for comparisons)
     *
     * @tparam InT Input element type
     * @tparam OutT Output element type
     * @tparam Op Binary functor type (must be __device__ callable)
     */
    template<typename InT, typename OutT, typename Op>
    void launch_binary_op_generic(const InT* a, const InT* b, OutT* c, size_t n,
                                  Op op, cudaStream_t stream = nullptr) {
        if (n == 0) return;

        auto a_ptr = thrust::device_pointer_cast(a);
        auto b_ptr = thrust::device_pointer_cast(b);
        auto c_ptr = thrust::device_pointer_cast(c);

        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, c_ptr, op);
        });
    }

    /**
     * Unary operation: Applies unary functor element-wise to an array
     * Supports different input/output types (e.g., float -> bool for predicates)
     *
     * @tparam InT Input element type
     * @tparam OutT Output element type
     * @tparam Op Unary functor type (must be __device__ callable)
     */
    template<typename InT, typename OutT, typename Op>
    void launch_unary_op_generic(const InT* input, OutT* output, size_t n,
                                 Op op, cudaStream_t stream = nullptr) {
        if (n == 0) return;

        auto in_ptr = thrust::device_pointer_cast(input);
        auto out_ptr = thrust::device_pointer_cast(output);

        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, in_ptr, in_ptr + n, out_ptr, op);
        });
    }

    /**
     * Scalar operation: Applies binary operation with a scalar value
     * OPTIMIZED: Uses constant_iterator for zero-memory scalar broadcasting
     *
     * @tparam T Element type
     * @tparam OutputT Output element type
     * @tparam Op Binary functor type (must be __device__ callable)
     */
    template<typename T, typename OutputT, typename Op>
    void launch_scalar_op_generic(const T* data, T scalar, OutputT* result, size_t n,
                                  Op op, cudaStream_t stream = nullptr) {
        if (n == 0) return;

        auto data_ptr = thrust::device_pointer_cast(data);
        auto result_ptr = thrust::device_pointer_cast(result);

        // Use constant_iterator for zero-memory scalar - generated on-the-fly!
        auto constant_scalar = thrust::make_constant_iterator(scalar);

        run_with_thrust_policy(stream, [&](auto policy) {
            // Binary transform: tensor op constant_iterator
            thrust::transform(policy, data_ptr, data_ptr + n, constant_scalar, result_ptr, op);
        });
    }

} // namespace gs::tensor_ops
