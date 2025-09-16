/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_ops.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace gs {

// Helper macro
#define CHECK_CUDA(x)                                                                          \
    do {                                                                                       \
        cudaError_t err = x;                                                                   \
        if (err != cudaSuccess) {                                                              \
            LOG_ERROR("CUDA error: {} at {}:{}", cudaGetErrorString(err), __FILE__, __LINE__); \
        }                                                                                      \
    } while (0)

#define CHECK_CUBLAS(x)                                                                        \
    do {                                                                                       \
        cublasStatus_t err = x;                                                                \
        if (err != CUBLAS_STATUS_SUCCESS) {                                                    \
            LOG_ERROR("cuBLAS error: {} at {}:{}", static_cast<int>(err), __FILE__, __LINE__); \
        }                                                                                      \
    } while (0)

    // Static cuBLAS handle management
    static cublasHandle_t& get_cublas_handle() {
        static cublasHandle_t handle = nullptr;
        if (!handle) {
            CHECK_CUBLAS(cublasCreate(&handle));
        }
        return handle;
    }

    // ============= Matrix Multiplication =============
    Tensor Tensor::matmul(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensor in matmul");
            return Tensor();
        }

        if (dtype_ != DataType::Float32 || other.dtype_ != DataType::Float32) {
            LOG_ERROR("MatMul only implemented for float32");
            return Tensor();
        }

        if (device_ != other.device_) {
            LOG_ERROR("MatMul requires tensors on same device");
            return Tensor();
        }

        // Determine dimensions for matrix multiplication
        size_t m, k, n;
        std::vector<size_t> result_shape;

        // Handle different dimension cases
        if (shape_.rank() == 1 && other.shape_.rank() == 1) {
            // Vector dot product
            if (shape_[0] != other.shape_[0]) {
                LOG_ERROR("Vector dimensions don't match for dot product");
                return Tensor();
            }
            return dot(other);
        } else if (shape_.rank() == 1 && other.shape_.rank() == 2) {
            // Vector-matrix multiplication (1D @ 2D)
            if (shape_[0] != other.shape_[0]) {
                LOG_ERROR("Dimension mismatch for vector-matrix multiplication");
                return Tensor();
            }
            m = 1;
            k = shape_[0];
            n = other.shape_[1];
            result_shape = {n};
        } else if (shape_.rank() == 2 && other.shape_.rank() == 1) {
            // Matrix-vector multiplication (2D @ 1D)
            if (shape_[1] != other.shape_[0]) {
                LOG_ERROR("Dimension mismatch for matrix-vector multiplication");
                return Tensor();
            }
            m = shape_[0];
            k = shape_[1];
            n = 1;
            result_shape = {m};
        } else if (shape_.rank() == 2 && other.shape_.rank() == 2) {
            // Matrix-matrix multiplication (2D @ 2D)
            if (shape_[1] != other.shape_[0]) {
                LOG_ERROR("Matrix dimensions don't match for multiplication: {}x{} @ {}x{}",
                          shape_[0], shape_[1], other.shape_[0], other.shape_[1]);
                return Tensor();
            }
            m = shape_[0];
            k = shape_[1];
            n = other.shape_[1];
            result_shape = {m, n};
        } else {
            LOG_ERROR("MatMul not implemented for {}D @ {}D", shape_.rank(), other.shape_.rank());
            return Tensor();
        }

        auto result = empty(TensorShape(result_shape), device_, dtype_);

        if (device_ == Device::CUDA) {
            // Use cuBLAS for GPU
            cublasHandle_t handle = get_cublas_handle();

            const float alpha = 1.0f;
            const float beta = 0.0f;

            // cuBLAS uses column-major, but we have row-major
            // So we compute C^T = B^T * A^T
            // This means we swap A and B, and swap m and n
            CHECK_CUBLAS(cublasSgemm(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, m, k,
                                     &alpha,
                                     other.ptr<float>(), n, // B (as is)
                                     ptr<float>(), k,       // A (as is)
                                     &beta,
                                     result.ptr<float>(), n));

            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation
            const float* a_data = ptr<float>();
            const float* b_data = other.ptr<float>();
            float* c_data = result.ptr<float>();

            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; ++l) {
                        sum += a_data[i * k + l] * b_data[l * n + j];
                    }
                    c_data[i * n + j] = sum;
                }
            }
        }

        return result;
    }

    Tensor Tensor::bmm(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensor in bmm");
            return Tensor();
        }

        if (shape_.rank() != 3 || other.shape_.rank() != 3) {
            LOG_ERROR("BMM requires 3D tensors");
            return Tensor();
        }

        if (shape_[0] != other.shape_[0]) {
            LOG_ERROR("Batch sizes don't match in BMM");
            return Tensor();
        }

        if (shape_[2] != other.shape_[1]) {
            LOG_ERROR("Matrix dimensions don't match in BMM");
            return Tensor();
        }

        size_t batch_size = shape_[0];
        size_t m = shape_[1];
        size_t k = shape_[2];
        size_t n = other.shape_[2];

        auto result = empty({batch_size, m, n}, device_, dtype_);

        if (device_ == Device::CUDA) {
            // Use cuBLAS batched GEMM
            cublasHandle_t handle = get_cublas_handle();

            const float alpha = 1.0f;
            const float beta = 0.0f;

            size_t stride_a = m * k;
            size_t stride_b = k * n;
            size_t stride_c = m * n;

            CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
                                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                                   n, m, k,
                                                   &alpha,
                                                   other.ptr<float>(), n, stride_b,
                                                   ptr<float>(), k, stride_a,
                                                   &beta,
                                                   result.ptr<float>(), n, stride_c,
                                                   batch_size));

            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation
            const float* a_data = ptr<float>();
            const float* b_data = other.ptr<float>();
            float* c_data = result.ptr<float>();

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        float sum = 0.0f;
                        for (size_t l = 0; l < k; ++l) {
                            sum += a_data[b * m * k + i * k + l] *
                                   b_data[b * k * n + l * n + j];
                        }
                        c_data[b * m * n + i * n + j] = sum;
                    }
                }
            }
        }

        return result;
    }

    Tensor Tensor::dot(const Tensor& other) const {
        if (!is_valid() || !other.is_valid()) {
            LOG_ERROR("Invalid tensor in dot");
            return Tensor();
        }

        if (shape_.rank() != 1 || other.shape_.rank() != 1) {
            LOG_ERROR("Dot product requires 1D tensors");
            return Tensor();
        }

        if (shape_[0] != other.shape_[0]) {
            LOG_ERROR("Vector dimensions don't match for dot product");
            return Tensor();
        }

        auto result = empty({1}, device_, dtype_);

        if (device_ == Device::CUDA) {
            cublasHandle_t handle = get_cublas_handle();

            CHECK_CUBLAS(cublasSdot(handle,
                                    shape_[0],
                                    ptr<float>(), 1,
                                    other.ptr<float>(), 1,
                                    result.ptr<float>()));

            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            const float* a_data = ptr<float>();
            const float* b_data = other.ptr<float>();
            float sum = 0.0f;

            for (size_t i = 0; i < shape_[0]; ++i) {
                sum += a_data[i] * b_data[i];
            }

            *result.ptr<float>() = sum;
        }

        return result;
    }

    // ============= Transpose =============
    Tensor Tensor::t() const {
        if (!is_valid()) {
            return Tensor();
        }

        if (shape_.rank() < 2) {
            LOG_ERROR("Transpose requires at least 2D tensor");
            return Tensor();
        }

        // Determine new shape (transpose last two dimensions)
        std::vector<size_t> new_shape = shape_.dims();
        size_t rank = shape_.rank();
        std::swap(new_shape[rank - 2], new_shape[rank - 1]);

        auto result = empty(TensorShape(new_shape), device_, dtype_);

        if (device_ == Device::CUDA) {
            // Use CUDA kernel for transpose
            if (shape_.rank() == 2) {
                tensor_ops::launch_transpose(
                    ptr<float>(), result.ptr<float>(),
                    shape_[0], shape_[1], 0);
            } else {
                // For higher dimensions, transpose last two dims for each batch
                size_t batch_size = 1;
                for (size_t i = 0; i < rank - 2; ++i) {
                    batch_size *= shape_[i];
                }

                size_t rows = shape_[rank - 2];
                size_t cols = shape_[rank - 1];
                size_t matrix_size = rows * cols;

                for (size_t b = 0; b < batch_size; ++b) {
                    tensor_ops::launch_transpose(
                        ptr<float>() + b * matrix_size,
                        result.ptr<float>() + b * matrix_size,
                        rows, cols, 0);
                }
            }
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            // CPU implementation
            const float* src = ptr<float>();
            float* dst = result.ptr<float>();

            if (shape_.rank() == 2) {
                size_t rows = shape_[0];
                size_t cols = shape_[1];

                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        dst[j * rows + i] = src[i * cols + j];
                    }
                }
            } else {
                // For higher dimensions
                size_t batch_size = 1;
                for (size_t i = 0; i < rank - 2; ++i) {
                    batch_size *= shape_[i];
                }

                size_t rows = shape_[rank - 2];
                size_t cols = shape_[rank - 1];

                for (size_t b = 0; b < batch_size; ++b) {
                    for (size_t i = 0; i < rows; ++i) {
                        for (size_t j = 0; j < cols; ++j) {
                            size_t src_idx = b * rows * cols + i * cols + j;
                            size_t dst_idx = b * rows * cols + j * rows + i;
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
            }
        }

        return result;
    }

    // ============= Matrix Creation Functions =============
    namespace tensor {

        Tensor eye(size_t n, Device device) {
            return eye(n, n, device);
        }

        Tensor eye(size_t m, size_t n, Device device) {
            auto result = Tensor::zeros({m, n}, device);

            if (device == Device::CUDA) {
                tensor_ops::launch_eye(result.ptr<float>(), m, n, 0);
                cudaDeviceSynchronize();
            } else {
                float* data = result.ptr<float>();
                size_t min_dim = std::min(m, n);
                for (size_t i = 0; i < min_dim; ++i) {
                    data[i * n + i] = 1.0f;
                }
            }

            return result;
        }

        Tensor diag(const Tensor& diagonal) {
            if (!diagonal.is_valid()) {
                return Tensor();
            }

            if (diagonal.shape().rank() != 1) {
                LOG_ERROR("diag() requires 1D tensor");
                return Tensor();
            }

            size_t n = diagonal.shape()[0];
            auto result = Tensor::zeros({n, n}, diagonal.device());

            if (diagonal.device() == Device::CUDA) {
                tensor_ops::launch_diag(diagonal.ptr<float>(), result.ptr<float>(), n, 0);
                cudaDeviceSynchronize();
            } else {
                const float* diag_data = diagonal.ptr<float>();
                float* mat_data = result.ptr<float>();

                for (size_t i = 0; i < n; ++i) {
                    mat_data[i * n + i] = diag_data[i];
                }
            }

            return result;
        }

    } // namespace tensor

#undef CHECK_CUDA
#undef CHECK_CUBLAS

} // namespace gs
