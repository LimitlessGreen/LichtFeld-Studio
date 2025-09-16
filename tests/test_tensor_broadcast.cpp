/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_broadcast.hpp"
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>

using namespace gs;

class TensorBroadcastTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set consistent seed for reproducibility
        torch::manual_seed(42);
        tensor::manual_seed(42);
    }

    // Helper to create a tensor with sequential values
    Tensor create_sequential_tensor(TensorShape shape, Device device = Device::CUDA) {
        auto tensor = Tensor::empty(shape, device);
        size_t n = tensor.numel();

        if (device == Device::CUDA) {
            std::vector<float> data(n);
            for (size_t i = 0; i < n; ++i) {
                data[i] = static_cast<float>(i);
            }
            cudaMemcpy(tensor.ptr<float>(), data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            float* data = tensor.ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                data[i] = static_cast<float>(i);
            }
        }

        return tensor;
    }

    // Helper to compare our tensor with PyTorch tensor
    bool compare_with_torch(const Tensor& our_tensor, const torch::Tensor& torch_tensor,
                            float tolerance = 1e-5f) {
        if (our_tensor.numel() != torch_tensor.numel()) {
            return false;
        }

        auto torch_cpu = torch_tensor.to(torch::kCPU);
        auto our_values = our_tensor.to_vector();
        auto torch_values = std::vector<float>(
            torch_cpu.data_ptr<float>(),
            torch_cpu.data_ptr<float>() + torch_cpu.numel());

        for (size_t i = 0; i < our_values.size(); ++i) {
            if (std::abs(our_values[i] - torch_values[i]) > tolerance) {
                return false;
            }
        }

        return true;
    }
};

// ============= Basic Broadcasting Tests =============

TEST_F(TensorBroadcastTest, CanBroadcast) {
    // Compatible shapes
    EXPECT_TRUE(BroadcastHelper::can_broadcast({3, 4}, {3, 4}));       // Same shape
    EXPECT_TRUE(BroadcastHelper::can_broadcast({3, 1}, {3, 4}));       // Broadcast dim 1
    EXPECT_TRUE(BroadcastHelper::can_broadcast({1, 4}, {3, 4}));       // Broadcast dim 0
    EXPECT_TRUE(BroadcastHelper::can_broadcast({1}, {3, 4}));          // Broadcast scalar
    EXPECT_TRUE(BroadcastHelper::can_broadcast({4}, {3, 4}));          // Broadcast 1D to 2D
    EXPECT_TRUE(BroadcastHelper::can_broadcast({3, 1, 5}, {3, 4, 5})); // 3D broadcast

    // Incompatible shapes
    EXPECT_FALSE(BroadcastHelper::can_broadcast({3, 4}, {3, 5})); // Mismatch in dim 1
    EXPECT_FALSE(BroadcastHelper::can_broadcast({3, 4}, {4, 4})); // Mismatch in dim 0
    EXPECT_FALSE(BroadcastHelper::can_broadcast({3}, {4}));       // Different 1D sizes
}

TEST_F(TensorBroadcastTest, BroadcastShape) {
    // Test broadcast shape computation
    auto check_broadcast_shape = [](TensorShape a, TensorShape b, TensorShape expected) {
        auto result = BroadcastHelper::broadcast_shape(a, b);
        return result == expected;
    };

    EXPECT_TRUE(check_broadcast_shape({3, 4}, {3, 4}, {3, 4}));
    EXPECT_TRUE(check_broadcast_shape({3, 1}, {3, 4}, {3, 4}));
    EXPECT_TRUE(check_broadcast_shape({1, 4}, {3, 4}, {3, 4}));
    EXPECT_TRUE(check_broadcast_shape({1}, {3, 4}, {3, 4}));
    EXPECT_TRUE(check_broadcast_shape({4}, {3, 4}, {3, 4}));
    EXPECT_TRUE(check_broadcast_shape({1, 1}, {3, 4}, {3, 4}));
    EXPECT_TRUE(check_broadcast_shape({5, 1, 4}, {5, 3, 4}, {5, 3, 4}));
    EXPECT_TRUE(check_broadcast_shape({1, 3, 1}, {5, 1, 4}, {5, 3, 4}));
}

// ============= Scalar Broadcasting Tests =============

TEST_F(TensorBroadcastTest, ScalarBroadcast) {
    // Broadcast scalar to various shapes
    auto scalar = Tensor::full({1}, 2.0f, Device::CUDA);
    auto target = Tensor::ones({3, 4}, Device::CUDA);

    auto result = scalar.add(target);
    EXPECT_EQ(result.shape(), target.shape());

    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 3.0f); // 2 + 1 = 3
    }
}

// ============= 1D to 2D Broadcasting Tests =============

TEST_F(TensorBroadcastTest, Broadcast1DRow) {
    // Broadcast row vector to matrix
    auto row = Tensor::full({4}, 2.0f, Device::CUDA);
    auto matrix = Tensor::ones({3, 4}, Device::CUDA);

    auto result = row.add(matrix);
    EXPECT_EQ(result.shape(), matrix.shape());

    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 3.0f); // 2 + 1 = 3
    }

    // Compare with PyTorch
    auto torch_row = torch::full({4}, 2.0f, torch::TensorOptions().device(torch::kCUDA));
    auto torch_matrix = torch::ones({3, 4}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_result = torch_row + torch_matrix;

    EXPECT_TRUE(compare_with_torch(result, torch_result));
}

TEST_F(TensorBroadcastTest, Broadcast1DColumn) {
    // Broadcast column vector to matrix
    auto col = Tensor::full({3, 1}, 3.0f, Device::CUDA);
    auto matrix = Tensor::full({3, 4}, 2.0f, Device::CUDA);

    auto result = col.add(matrix);
    EXPECT_EQ(result.shape(), matrix.shape());

    // Each row should have the same value added
    auto values = result.to_vector();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(values[i * 4 + j], 5.0f); // 3 + 2 = 5
        }
    }
}

// ============= 2D Broadcasting Tests =============

TEST_F(TensorBroadcastTest, BroadcastMatrixColumn) {
    // Create a column vector [0, 1, 2] and broadcast with a matrix
    auto col = create_sequential_tensor({3, 1}, Device::CUDA);
    auto matrix = Tensor::ones({3, 4}, Device::CUDA);

    auto result = col.add(matrix);
    EXPECT_EQ(result.shape(), TensorShape({3, 4}));

    // Each row should have the column value added
    auto values = result.to_vector();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(values[i * 4 + j], i + 1.0f);
        }
    }
}

TEST_F(TensorBroadcastTest, BroadcastMatrixRow) {
    // Create a row vector [0, 1, 2, 3] and broadcast with a matrix
    auto row = create_sequential_tensor({1, 4}, Device::CUDA);
    auto matrix = Tensor::ones({3, 4}, Device::CUDA);

    auto result = row.add(matrix);
    EXPECT_EQ(result.shape(), TensorShape({3, 4}));

    // Each column should have the row value added
    auto values = result.to_vector();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(values[i * 4 + j], j + 1.0f);
        }
    }
}

// ============= 3D Broadcasting Tests =============

TEST_F(TensorBroadcastTest, Broadcast3D) {
    // Test 3D broadcasting
    auto a = Tensor::full({2, 1, 3}, 2.0f, Device::CUDA);
    auto b = Tensor::full({2, 4, 3}, 1.0f, Device::CUDA);

    auto result = a.add(b);
    EXPECT_EQ(result.shape(), TensorShape({2, 4, 3}));

    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 3.0f);
    }
}

TEST_F(TensorBroadcastTest, ComplexBroadcast3D) {
    // More complex 3D broadcasting
    auto a = Tensor::full({1, 3, 1}, 5.0f, Device::CUDA);
    auto b = Tensor::full({2, 1, 4}, 3.0f, Device::CUDA);

    auto result = a.mul(b);
    EXPECT_EQ(result.shape(), TensorShape({2, 3, 4}));

    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 15.0f); // 5 * 3 = 15
    }
}

// ============= All Operations Broadcasting Tests =============

TEST_F(TensorBroadcastTest, BroadcastAddition) {
    auto a = Tensor::full({3, 1}, 2.0f, Device::CUDA);
    auto b = Tensor::full({1, 4}, 3.0f, Device::CUDA);

    auto result = a + b;
    EXPECT_EQ(result.shape(), TensorShape({3, 4}));

    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 5.0f);
    }

    // Compare with PyTorch
    auto torch_a = torch::full({3, 1}, 2.0f, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::full({1, 4}, 3.0f, torch::TensorOptions().device(torch::kCUDA));
    auto torch_result = torch_a + torch_b;

    EXPECT_TRUE(compare_with_torch(result, torch_result));
}

TEST_F(TensorBroadcastTest, BroadcastSubtraction) {
    auto a = Tensor::full({2, 3, 1}, 10.0f, Device::CUDA);
    auto b = Tensor::full({3, 4}, 3.0f, Device::CUDA);

    auto result = a - b;
    EXPECT_EQ(result.shape(), TensorShape({2, 3, 4}));

    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 7.0f);
    }
}

TEST_F(TensorBroadcastTest, BroadcastMultiplication) {
    auto a = create_sequential_tensor({3, 1}, Device::CUDA); // [0, 1, 2]^T
    auto b = create_sequential_tensor({1, 4}, Device::CUDA); // [0, 1, 2, 3]

    auto result = a * b;
    EXPECT_EQ(result.shape(), TensorShape({3, 4}));

    // Result should be outer product
    auto values = result.to_vector();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(values[i * 4 + j], i * j);
        }
    }
}

TEST_F(TensorBroadcastTest, BroadcastDivision) {
    auto a = Tensor::full({2, 3, 1}, 12.0f, Device::CUDA);
    auto b = Tensor::full({1, 4}, 3.0f, Device::CUDA);

    auto result = a / b;
    EXPECT_EQ(result.shape(), TensorShape({2, 3, 4}));

    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_NEAR(val, 4.0f, 1e-5f);
    }
}

// ============= Edge Cases =============

TEST_F(TensorBroadcastTest, BroadcastWithEmpty) {
    auto empty = Tensor::empty({0}, Device::CUDA);
    auto normal = Tensor::ones({3, 4}, Device::CUDA);

    // Broadcasting with empty should fail
    auto result = empty.add(normal);
    EXPECT_FALSE(result.is_valid());
}

TEST_F(TensorBroadcastTest, BroadcastSingleElement) {
    auto single = Tensor::full({1, 1, 1}, 5.0f, Device::CUDA);
    auto matrix = Tensor::full({3, 4, 5}, 2.0f, Device::CUDA);

    auto result = single.add(matrix);
    EXPECT_EQ(result.shape(), matrix.shape());

    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 7.0f);
    }
}

TEST_F(TensorBroadcastTest, BroadcastLargeTensors) {
    // Test with larger tensors
    auto a = Tensor::ones({100, 1}, Device::CUDA);
    auto b = Tensor::ones({1, 200}, Device::CUDA);

    auto result = a.add(b);
    EXPECT_EQ(result.shape(), TensorShape({100, 200}));
    EXPECT_EQ(result.numel(), 20000);

    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 2.0f);
    }
}

// ============= CPU Broadcasting Tests =============

TEST_F(TensorBroadcastTest, BroadcastOnCPU) {
    auto a = Tensor::full({3, 1}, 4.0f, Device::CPU);
    auto b = Tensor::full({1, 5}, 2.0f, Device::CPU);

    auto result = a.add(b);
    EXPECT_EQ(result.shape(), TensorShape({3, 5}));
    EXPECT_EQ(result.device(), Device::CPU);

    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 6.0f);
    }
}

// ============= Broadcasting Expansion Tests =============

TEST_F(TensorBroadcastTest, ExpandTensor) {
    auto tensor = Tensor::full({3, 1}, 2.0f, Device::CUDA);
    auto expanded = BroadcastHelper::expand(tensor, {3, 4});

    EXPECT_EQ(expanded.shape(), TensorShape({3, 4}));

    auto values = expanded.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 2.0f);
    }
}

TEST_F(TensorBroadcastTest, ExpandMultipleDimensions) {
    auto tensor = Tensor::full({1, 3, 1}, 3.0f, Device::CUDA);
    auto expanded = BroadcastHelper::expand(tensor, {2, 3, 4});

    EXPECT_EQ(expanded.shape(), TensorShape({2, 3, 4}));

    auto values = expanded.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 3.0f);
    }
}

// ============= Broadcasting with Mixed Shapes Tests =============

TEST_F(TensorBroadcastTest, MixedRankBroadcast) {
    // Test broadcasting between different ranks
    auto rank1 = Tensor::full({5}, 1.0f, Device::CUDA);
    auto rank2 = Tensor::full({3, 5}, 2.0f, Device::CUDA);
    auto rank3 = Tensor::full({2, 3, 5}, 3.0f, Device::CUDA);

    // 1D + 2D
    auto result12 = rank1.add(rank2);
    EXPECT_EQ(result12.shape(), TensorShape({3, 5}));

    // 1D + 3D
    auto result13 = rank1.add(rank3);
    EXPECT_EQ(result13.shape(), TensorShape({2, 3, 5}));

    // 2D + 3D
    auto result23 = rank2.add(rank3);
    EXPECT_EQ(result23.shape(), TensorShape({2, 3, 5}));
}

// ============= Performance Test =============

TEST_F(TensorBroadcastTest, BroadcastPerformance) {
    // Test performance of broadcasting large tensors
    const size_t rows = 1000;
    const size_t cols = 1000;

    auto col_vector = Tensor::randn({rows, 1}, Device::CUDA);
    auto row_vector = Tensor::randn({1, cols}, Device::CUDA);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = col_vector.add(row_vector);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    EXPECT_EQ(result.shape(), TensorShape({rows, cols}));

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Broadcast " << rows << "x1 + 1x" << cols
              << " took " << duration.count() << " ms" << std::endl;
}

// ============= Complex Expression Tests =============

TEST_F(TensorBroadcastTest, ChainedBroadcastOperations) {
    // Test chained operations with broadcasting
    auto a = Tensor::full({3, 1}, 2.0f, Device::CUDA);
    auto b = Tensor::full({1, 4}, 3.0f, Device::CUDA);
    auto c = Tensor::full({3, 4}, 1.0f, Device::CUDA);

    // (a + b) * c
    auto result = (a + b) * c;
    EXPECT_EQ(result.shape(), TensorShape({3, 4}));

    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 5.0f); // (2 + 3) * 1 = 5
    }
}

TEST_F(TensorBroadcastTest, ComplexBroadcastExpression) {
    // More complex expression with multiple broadcasts
    auto a = create_sequential_tensor({3, 1}, Device::CUDA); // [0, 1, 2]^T
    auto b = create_sequential_tensor({1, 4}, Device::CUDA); // [0, 1, 2, 3]
    auto c = Tensor::ones({3, 4}, Device::CUDA);

    // a * b + c
    auto result = a * b + c;
    EXPECT_EQ(result.shape(), TensorShape({3, 4}));

    auto values = result.to_vector();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(values[i * 4 + j], i * j + 1);
        }
    }
}

TEST_F(TensorBroadcastTest, ScalarMultiplication) {
    auto scalar = Tensor::full({1}, 3.0f, Device::CUDA);
    auto matrix = Tensor::full({2, 3}, 2.0f, Device::CUDA);

    auto result = scalar.mul(matrix);
    EXPECT_EQ(result.shape(), matrix.shape());

    auto values = result.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 6.0f); // 3 * 2 = 6
    }
}
