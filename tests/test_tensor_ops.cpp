/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>

using namespace gs;

class TensorOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        gen.seed(42);
    }

    torch::Tensor create_torch_tensor(const std::vector<size_t>& shape, bool random = false) {
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        if (random) {
            return torch::randn(torch_shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        }
        return torch::zeros(torch_shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }

    Tensor create_tensor_from_torch(const torch::Tensor& torch_tensor) {
        auto shape_vec = torch_tensor.sizes().vec();
        std::vector<size_t> shape(shape_vec.begin(), shape_vec.end());

        auto tensor = Tensor::empty(TensorShape(shape), Device::CUDA);
        cudaMemcpy(tensor.ptr<float>(), torch_tensor.data_ptr<float>(),
                   torch_tensor.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        return tensor;
    }

    bool compare_results(const Tensor& our_result, const torch::Tensor& torch_result, float tolerance = 1e-5f) {
        auto our_values = our_result.to_vector();
        auto torch_cpu = torch_result.to(torch::kCPU).contiguous();
        auto torch_data = torch_cpu.data_ptr<float>();

        if (our_values.size() != static_cast<size_t>(torch_cpu.numel())) {
            return false;
        }

        for (size_t i = 0; i < our_values.size(); ++i) {
            if (std::abs(our_values[i] - torch_data[i]) > tolerance) {
                std::cout << "Mismatch at index " << i << ": "
                          << our_values[i] << " vs " << torch_data[i] << std::endl;
                return false;
            }
        }
        return true;
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};
};

// Scalar operations
TEST_F(TensorOpsTest, ScalarAdd) {
    auto torch_tensor = create_torch_tensor({3, 4}, true);
    auto our_tensor = create_tensor_from_torch(torch_tensor);

    float scalar = 2.5f;

    auto torch_result = torch_tensor + scalar;
    auto our_result = our_tensor.add(scalar);

    EXPECT_TRUE(compare_results(our_result, torch_result));

    // Test operator overload
    auto our_result2 = our_tensor + scalar;
    EXPECT_TRUE(compare_results(our_result2, torch_result));
}

TEST_F(TensorOpsTest, ScalarSubtract) {
    auto torch_tensor = create_torch_tensor({2, 5}, true);
    auto our_tensor = create_tensor_from_torch(torch_tensor);

    float scalar = 1.5f;

    auto torch_result = torch_tensor - scalar;
    auto our_result = our_tensor.sub(scalar);

    EXPECT_TRUE(compare_results(our_result, torch_result));

    // Test operator overload
    auto our_result2 = our_tensor - scalar;
    EXPECT_TRUE(compare_results(our_result2, torch_result));
}

TEST_F(TensorOpsTest, ScalarMultiply) {
    auto torch_tensor = create_torch_tensor({4, 3}, true);
    auto our_tensor = create_tensor_from_torch(torch_tensor);

    float scalar = 3.0f;

    auto torch_result = torch_tensor * scalar;
    auto our_result = our_tensor.mul(scalar);

    EXPECT_TRUE(compare_results(our_result, torch_result));

    // Test operator overload
    auto our_result2 = our_tensor * scalar;
    EXPECT_TRUE(compare_results(our_result2, torch_result));
}

TEST_F(TensorOpsTest, ScalarDivide) {
    auto torch_tensor = create_torch_tensor({3, 3}, true);
    auto our_tensor = create_tensor_from_torch(torch_tensor);

    float scalar = 2.0f;

    auto torch_result = torch_tensor / scalar;
    auto our_result = our_tensor.div(scalar);

    EXPECT_TRUE(compare_results(our_result, torch_result));

    // Test operator overload
    auto our_result2 = our_tensor / scalar;
    EXPECT_TRUE(compare_results(our_result2, torch_result));
}

TEST_F(TensorOpsTest, Negation) {
    auto torch_tensor = create_torch_tensor({2, 4}, true);
    auto our_tensor = create_tensor_from_torch(torch_tensor);

    auto torch_result = -torch_tensor;
    auto our_result = our_tensor.neg();

    EXPECT_TRUE(compare_results(our_result, torch_result));

    // Test operator overload
    auto our_result2 = -our_tensor;
    EXPECT_TRUE(compare_results(our_result2, torch_result));
}

// Element-wise operations
TEST_F(TensorOpsTest, ElementWiseAdd) {
    auto torch_a = create_torch_tensor({3, 4}, true);
    auto torch_b = create_torch_tensor({3, 4}, true);
    auto our_a = create_tensor_from_torch(torch_a);
    auto our_b = create_tensor_from_torch(torch_b);

    auto torch_result = torch_a + torch_b;
    auto our_result = our_a.add(our_b);

    EXPECT_TRUE(compare_results(our_result, torch_result));

    // Test operator overload
    auto our_result2 = our_a + our_b;
    EXPECT_TRUE(compare_results(our_result2, torch_result));
}

TEST_F(TensorOpsTest, ElementWiseSubtract) {
    auto torch_a = create_torch_tensor({2, 5}, true);
    auto torch_b = create_torch_tensor({2, 5}, true);
    auto our_a = create_tensor_from_torch(torch_a);
    auto our_b = create_tensor_from_torch(torch_b);

    auto torch_result = torch_a - torch_b;
    auto our_result = our_a.sub(our_b);

    EXPECT_TRUE(compare_results(our_result, torch_result));

    // Test operator overload
    auto our_result2 = our_a - our_b;
    EXPECT_TRUE(compare_results(our_result2, torch_result));
}

TEST_F(TensorOpsTest, ElementWiseMultiply) {
    auto torch_a = create_torch_tensor({4, 3}, true);
    auto torch_b = create_torch_tensor({4, 3}, true);
    auto our_a = create_tensor_from_torch(torch_a);
    auto our_b = create_tensor_from_torch(torch_b);

    auto torch_result = torch_a * torch_b;
    auto our_result = our_a.mul(our_b);

    EXPECT_TRUE(compare_results(our_result, torch_result));
}

TEST_F(TensorOpsTest, ElementWiseDivide) {
    auto torch_a = create_torch_tensor({3, 3}, true);
    auto torch_b = create_torch_tensor({3, 3}, true) + 2.0f; // Avoid division by zero
    auto our_a = create_tensor_from_torch(torch_a);
    auto our_b = create_tensor_from_torch(torch_b);

    auto torch_result = torch_a / torch_b;
    auto our_result = our_a.div(our_b);

    EXPECT_TRUE(compare_results(our_result, torch_result, 1e-4f));
}

// In-place operations
TEST_F(TensorOpsTest, InPlaceScalarAdd) {
    auto torch_tensor = create_torch_tensor({3, 4}, true);
    auto our_tensor = create_tensor_from_torch(torch_tensor);

    float scalar = 2.5f;

    torch_tensor.add_(scalar);
    our_tensor.add_(scalar);

    EXPECT_TRUE(compare_results(our_tensor, torch_tensor));
}

TEST_F(TensorOpsTest, InPlaceScalarMultiply) {
    auto torch_tensor = create_torch_tensor({2, 3}, true);
    auto our_tensor = create_tensor_from_torch(torch_tensor);

    float scalar = 3.0f;

    torch_tensor.mul_(scalar);
    our_tensor.mul_(scalar);

    EXPECT_TRUE(compare_results(our_tensor, torch_tensor));
}

TEST_F(TensorOpsTest, InPlaceElementWiseAdd) {
    auto torch_a = create_torch_tensor({3, 3}, true);
    auto torch_b = create_torch_tensor({3, 3}, true);
    auto our_a = create_tensor_from_torch(torch_a);
    auto our_b = create_tensor_from_torch(torch_b);

    torch_a.add_(torch_b);
    our_a.add_(our_b);

    EXPECT_TRUE(compare_results(our_a, torch_a));
}

TEST_F(TensorOpsTest, InPlaceElementWiseSubtract) {
    auto torch_a = create_torch_tensor({2, 4}, true);
    auto torch_b = create_torch_tensor({2, 4}, true);
    auto our_a = create_tensor_from_torch(torch_a);
    auto our_b = create_tensor_from_torch(torch_b);

    torch_a.sub_(torch_b);
    our_a.sub_(our_b);

    EXPECT_TRUE(compare_results(our_a, torch_a));
}

TEST_F(TensorOpsTest, InPlaceElementWiseMultiply) {
    auto torch_a = create_torch_tensor({3, 2}, true);
    auto torch_b = create_torch_tensor({3, 2}, true);
    auto our_a = create_tensor_from_torch(torch_a);
    auto our_b = create_tensor_from_torch(torch_b);

    torch_a.mul_(torch_b);
    our_a.mul_(our_b);

    EXPECT_TRUE(compare_results(our_a, torch_a));
}

TEST_F(TensorOpsTest, InPlaceElementWiseDivide) {
    auto torch_a = create_torch_tensor({2, 2}, true);
    auto torch_b = create_torch_tensor({2, 2}, true) + 2.0f; // Avoid division by zero
    auto our_a = create_tensor_from_torch(torch_a);
    auto our_b = create_tensor_from_torch(torch_b);

    torch_a.div_(torch_b);
    our_a.div_(our_b);

    EXPECT_TRUE(compare_results(our_a, torch_a, 1e-4f));
}

// Edge cases
TEST_F(TensorOpsTest, DivisionByZeroHandling) {
    auto tensor = Tensor::ones({2, 2}, Device::CUDA);

    // Division by near-zero scalar should be handled
    auto result = tensor.div(1e-10f);
    EXPECT_TRUE(result.is_valid());

    // Element-wise division with zeros
    auto zeros = Tensor::zeros({2, 2}, Device::CUDA);
    auto result2 = tensor.div(zeros);
    EXPECT_TRUE(result2.is_valid());

    // Check that values are large but not infinite
    auto values = result2.to_vector();
    for (float val : values) {
        EXPECT_FALSE(std::isinf(val));
        EXPECT_GT(std::abs(val), 1e7f); // Should be large
    }
}

TEST_F(TensorOpsTest, ChainedOperations) {
    auto torch_tensor = create_torch_tensor({3, 3}, true);
    auto our_tensor = create_tensor_from_torch(torch_tensor);

    // Chain operations: ((t + 2) * 3) - 1
    auto torch_result = ((torch_tensor + 2.0f) * 3.0f) - 1.0f;
    auto our_result = ((our_tensor + 2.0f) * 3.0f) - 1.0f;

    EXPECT_TRUE(compare_results(our_result, torch_result));
}

TEST_F(TensorOpsTest, MixedOperations) {
    auto torch_a = create_torch_tensor({4, 4}, true);
    auto torch_b = create_torch_tensor({4, 4}, true);
    auto our_a = create_tensor_from_torch(torch_a);
    auto our_b = create_tensor_from_torch(torch_b);

    // Complex expression: (a * 2 + b) / (a - b + 1)
    auto torch_result = (torch_a * 2 + torch_b) / (torch_a - torch_b + 1);
    auto our_result = (our_a * 2 + our_b) / (our_a - our_b + 1);

    EXPECT_TRUE(compare_results(our_result, torch_result, 1e-4f));
}
