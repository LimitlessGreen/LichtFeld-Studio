/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>

using namespace gs;

class TensorMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        tensor::manual_seed(42);

        // Ensure CUDA is properly initialized
        cudaSetDevice(0);
    }

    bool compare_tensors(const Tensor& our_tensor, const torch::Tensor& torch_tensor,
                         float tolerance = 1e-4f) {
        auto our_values = our_tensor.to_vector();

        // Always work with CPU tensors for comparison
        torch::Tensor torch_cpu;
        if (torch_tensor.is_cuda()) {
            torch_cpu = torch_tensor.to(torch::kCPU).contiguous();
        } else {
            torch_cpu = torch_tensor.contiguous();
        }

        auto torch_data = torch_cpu.data_ptr<float>();

        if (our_values.size() != static_cast<size_t>(torch_cpu.numel())) {
            std::cerr << "Size mismatch: " << our_values.size()
                      << " vs " << torch_cpu.numel() << std::endl;
            return false;
        }

        for (size_t i = 0; i < our_values.size(); ++i) {
            if (std::abs(our_values[i] - torch_data[i]) > tolerance) {
                std::cerr << "Value mismatch at index " << i << ": "
                          << our_values[i] << " vs " << torch_data[i]
                          << " (diff: " << std::abs(our_values[i] - torch_data[i]) << ")"
                          << std::endl;
                return false;
            }
        }
        return true;
    }

    Tensor create_tensor_from_vector(const std::vector<float>& data,
                                     const std::vector<size_t>& shape) {
        auto tensor = Tensor::empty(TensorShape(shape), Device::CUDA);
        cudaMemcpy(tensor.ptr<float>(), data.data(),
                   data.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        return tensor;
    }

    torch::Tensor create_torch_from_vector(const std::vector<float>& data,
                                           const std::vector<int64_t>& shape) {
        // Always create on CPU to avoid CUDA issues
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto tensor = torch::from_blob(const_cast<float*>(data.data()), shape, options).clone();
        return tensor;
    }

    // Helper to perform torch operations on CPU
    torch::Tensor torch_matmul_cpu(const torch::Tensor& a, const torch::Tensor& b) {
        auto a_cpu = a.is_cuda() ? a.to(torch::kCPU) : a;
        auto b_cpu = b.is_cuda() ? b.to(torch::kCPU) : b;
        return torch::matmul(a_cpu, b_cpu);
    }
};

// ============= Matrix Multiplication Tests =============

TEST_F(TensorMatrixTest, MatMul2D) {
    // Test basic 2D matrix multiplication
    std::vector<float> data_a = {1, 2, 3,
                                 4, 5, 6}; // 2x3
    std::vector<float> data_b = {7, 8,
                                 9, 10,
                                 11, 12}; // 3x2

    auto our_a = create_tensor_from_vector(data_a, {2, 3});
    auto our_b = create_tensor_from_vector(data_b, {3, 2});

    auto torch_a = create_torch_from_vector(data_a, {2, 3});
    auto torch_b = create_torch_from_vector(data_b, {3, 2});

    auto our_result = our_a.matmul(our_b);
    auto torch_result = torch_matmul_cpu(torch_a, torch_b);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));

    // Test mm() alias
    auto our_result2 = our_a.mm(our_b);
    EXPECT_TRUE(compare_tensors(our_result2, torch_result));
}

TEST_F(TensorMatrixTest, MatMulVectorMatrix) {
    // Test vector-matrix multiplication
    std::vector<float> vec_data = {1, 2, 3};
    std::vector<float> mat_data = {4, 5,
                                   6, 7,
                                   8, 9}; // 3x2

    auto our_vec = create_tensor_from_vector(vec_data, {3});
    auto our_mat = create_tensor_from_vector(mat_data, {3, 2});

    auto torch_vec = create_torch_from_vector(vec_data, {3});
    auto torch_mat = create_torch_from_vector(mat_data, {3, 2});

    auto our_result = our_vec.matmul(our_mat);
    auto torch_result = torch_matmul_cpu(torch_vec, torch_mat);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));
}

TEST_F(TensorMatrixTest, MatMulMatrixVector) {
    // Test matrix-vector multiplication
    std::vector<float> mat_data = {1, 2, 3,
                                   4, 5, 6}; // 2x3
    std::vector<float> vec_data = {7, 8, 9};

    auto our_mat = create_tensor_from_vector(mat_data, {2, 3});
    auto our_vec = create_tensor_from_vector(vec_data, {3});

    auto torch_mat = create_torch_from_vector(mat_data, {2, 3});
    auto torch_vec = create_torch_from_vector(vec_data, {3});

    auto our_result = our_mat.matmul(our_vec);
    auto torch_result = torch_matmul_cpu(torch_mat, torch_vec);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));
}

TEST_F(TensorMatrixTest, DotProduct) {
    // Test dot product of two vectors
    std::vector<float> vec1_data = {1, 2, 3, 4};
    std::vector<float> vec2_data = {5, 6, 7, 8};

    auto our_vec1 = create_tensor_from_vector(vec1_data, {4});
    auto our_vec2 = create_tensor_from_vector(vec2_data, {4});

    auto torch_vec1 = create_torch_from_vector(vec1_data, {4});
    auto torch_vec2 = create_torch_from_vector(vec2_data, {4});

    auto our_result = our_vec1.dot(our_vec2);
    auto torch_result = torch::dot(torch_vec1, torch_vec2);

    EXPECT_FLOAT_EQ(our_result.item(), torch_result.item<float>());
}

TEST_F(TensorMatrixTest, BatchMatMul) {
    // Test batch matrix multiplication
    std::vector<float> data_a(2 * 3 * 4); // batch=2, 3x4
    std::vector<float> data_b(2 * 4 * 5); // batch=2, 4x5

    // Fill with sequential values
    std::iota(data_a.begin(), data_a.end(), 0.0f);
    std::iota(data_b.begin(), data_b.end(), 0.0f);

    auto our_a = create_tensor_from_vector(data_a, {2, 3, 4});
    auto our_b = create_tensor_from_vector(data_b, {2, 4, 5});

    auto torch_a = create_torch_from_vector(data_a, {2, 3, 4});
    auto torch_b = create_torch_from_vector(data_b, {2, 4, 5});

    auto our_result = our_a.bmm(our_b);
    auto torch_result = torch::bmm(torch_a, torch_b);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));
}

// ============= Transpose Tests =============

TEST_F(TensorMatrixTest, Transpose2D) {
    // Test 2D transpose
    std::vector<float> data = {1, 2, 3,
                               4, 5, 6}; // 2x3

    auto our_tensor = create_tensor_from_vector(data, {2, 3});
    auto torch_tensor = create_torch_from_vector(data, {2, 3});

    auto our_result = our_tensor.t();
    auto torch_result = torch_tensor.t();

    EXPECT_TRUE(compare_tensors(our_result, torch_result));

    // Check shape
    EXPECT_EQ(our_result.shape()[0], 3);
    EXPECT_EQ(our_result.shape()[1], 2);
}

TEST_F(TensorMatrixTest, Transpose3D) {
    // Test 3D transpose (transposes last two dimensions)
    std::vector<float> data(2 * 3 * 4);
    std::iota(data.begin(), data.end(), 0.0f);

    auto our_tensor = create_tensor_from_vector(data, {2, 3, 4});
    auto torch_tensor = create_torch_from_vector(data, {2, 3, 4});

    auto our_result = our_tensor.t();
    auto torch_result = torch_tensor.transpose(-2, -1);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));

    // Check shape
    EXPECT_EQ(our_result.shape()[0], 2);
    EXPECT_EQ(our_result.shape()[1], 4);
    EXPECT_EQ(our_result.shape()[2], 3);
}

TEST_F(TensorMatrixTest, TransposeSpecificDims) {
    // Test transpose with specific dimensions
    std::vector<float> data(2 * 3 * 4);
    std::iota(data.begin(), data.end(), 0.0f);

    auto our_tensor = create_tensor_from_vector(data, {2, 3, 4});
    auto torch_tensor = create_torch_from_vector(data, {2, 3, 4});

    auto our_result = our_tensor.transpose(0, 1);
    auto torch_result = torch_tensor.transpose(0, 1);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));
}

// ============= Matrix Creation Tests =============
TEST_F(TensorMatrixTest, Diag) {
    // Test creating diagonal matrix from vector
    std::vector<float> diag_data = {1, 2, 3, 4};
    auto our_diag_vec = create_tensor_from_vector(diag_data, {4});
    auto torch_diag_vec = create_torch_from_vector(diag_data, {4});

    auto our_result = tensor::diag(our_diag_vec);
    auto torch_result = torch::diag(torch_diag_vec);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));
}

// ============= Complex Matrix Operations =============

TEST_F(TensorMatrixTest, MatMulChain) {
    // Test chaining multiple matrix multiplications
    std::vector<float> a_data = {1, 2, 3, 4};    // 2x2
    std::vector<float> b_data = {5, 6, 7, 8};    // 2x2
    std::vector<float> c_data = {9, 10, 11, 12}; // 2x2

    auto our_a = create_tensor_from_vector(a_data, {2, 2});
    auto our_b = create_tensor_from_vector(b_data, {2, 2});
    auto our_c = create_tensor_from_vector(c_data, {2, 2});

    auto torch_a = create_torch_from_vector(a_data, {2, 2});
    auto torch_b = create_torch_from_vector(b_data, {2, 2});
    auto torch_c = create_torch_from_vector(c_data, {2, 2});

    auto our_result = our_a.matmul(our_b).matmul(our_c);
    auto torch_result = torch_matmul_cpu(torch_matmul_cpu(torch_a, torch_b), torch_c);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));
}

TEST_F(TensorMatrixTest, TransposeMatMul) {
    // Test A^T @ B
    std::vector<float> a_data = {1, 2, 3,
                                 4, 5, 6}; // 2x3
    std::vector<float> b_data = {7, 8,
                                 9, 10}; // 2x2

    auto our_a = create_tensor_from_vector(a_data, {2, 3});
    auto our_b = create_tensor_from_vector(b_data, {2, 2});

    auto torch_a = create_torch_from_vector(a_data, {2, 3});
    auto torch_b = create_torch_from_vector(b_data, {2, 2});

    auto our_result = our_a.t().matmul(our_b);
    auto torch_result = torch_matmul_cpu(torch_a.t(), torch_b);

    EXPECT_TRUE(compare_tensors(our_result, torch_result));
}

// ============= Edge Cases =============

TEST_F(TensorMatrixTest, SingleElementMatMul) {
    // Test 1x1 matrix multiplication
    auto our_a = Tensor::full({1, 1}, 3.0f, Device::CUDA);
    auto our_b = Tensor::full({1, 1}, 4.0f, Device::CUDA);

    auto result = our_a.matmul(our_b);
    EXPECT_FLOAT_EQ(result.item(), 12.0f);
}

TEST_F(TensorMatrixTest, LargeMatMul) {
    // Test larger matrices for performance
    const size_t n = 128;

    auto our_a = Tensor::randn({n, n}, Device::CUDA);
    auto our_b = Tensor::randn({n, n}, Device::CUDA);

    // Get data for torch tensors
    auto a_vec = our_a.to_vector();
    auto b_vec = our_b.to_vector();

    std::vector<int64_t> shape = {static_cast<int64_t>(n), static_cast<int64_t>(n)};

    // Create torch tensors on CPU only
    auto torch_a = torch::from_blob(a_vec.data(), shape,
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                       .clone();
    auto torch_b = torch::from_blob(b_vec.data(), shape,
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                       .clone();

    auto our_result = our_a.matmul(our_b);
    auto torch_result = torch_matmul_cpu(torch_a, torch_b);

    // Use slightly higher tolerance for larger matrices
    EXPECT_TRUE(compare_tensors(our_result, torch_result, 1e-3f));
}

TEST_F(TensorMatrixTest, InvalidMatMul) {
    // Test dimension mismatch
    auto a = Tensor::ones({2, 3}, Device::CUDA);
    auto b = Tensor::ones({4, 5}, Device::CUDA);

    auto result = a.matmul(b);
    EXPECT_FALSE(result.is_valid());
}

// ============= Performance Test =============
TEST_F(TensorMatrixTest, MatMulPerformance) {
    // Compare performance with PyTorch
    const size_t n = 512;

    auto our_a = Tensor::randn({n, n}, Device::CUDA);
    auto our_b = Tensor::randn({n, n}, Device::CUDA);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        auto result = our_a.matmul(our_b);
        cudaDeviceSynchronize();
    }
    auto our_time = std::chrono::high_resolution_clock::now() - start;

    // Just log the time, don't assert on performance
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(our_time).count();
    std::cout << "MatMul " << n << "x" << n << " took " << ms / 10.0 << " ms per operation" << std::endl;
}
