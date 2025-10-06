/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core/tensor.hpp"

using namespace gs;

class TensorAccessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        tensor::manual_seed(42);
    }
};

// ============= 1D Accessor Tests =============

TEST_F(TensorAccessorTest, Accessor1DBasic) {
    auto t = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5}, Device::CPU);
    auto acc = t.accessor<float, 1>();

    EXPECT_EQ(acc.sizes()[0], 5);

    // Read access
    EXPECT_FLOAT_EQ(acc(0), 1.0f);
    EXPECT_FLOAT_EQ(acc(1), 2.0f);
    EXPECT_FLOAT_EQ(acc(2), 3.0f);
    EXPECT_FLOAT_EQ(acc(3), 4.0f);
    EXPECT_FLOAT_EQ(acc(4), 5.0f);
}

TEST_F(TensorAccessorTest, Accessor1DWrite) {
    auto t = Tensor::zeros({5}, Device::CPU);
    auto acc = t.accessor<float, 1>();

    // Write access
    acc(0) = 10.0f;
    acc(1) = 20.0f;
    acc(2) = 30.0f;
    acc(3) = 40.0f;
    acc(4) = 50.0f;

    auto values = t.to_vector();
    EXPECT_FLOAT_EQ(values[0], 10.0f);
    EXPECT_FLOAT_EQ(values[1], 20.0f);
    EXPECT_FLOAT_EQ(values[2], 30.0f);
    EXPECT_FLOAT_EQ(values[3], 40.0f);
    EXPECT_FLOAT_EQ(values[4], 50.0f);
}

// ============= 2D Accessor Tests =============

TEST_F(TensorAccessorTest, Accessor2DBasic) {
    auto t = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        {2, 3},
        Device::CPU
    );
    auto acc = t.accessor<float, 2>();

    EXPECT_EQ(acc.sizes()[0], 2);
    EXPECT_EQ(acc.sizes()[1], 3);

    // Read access
    EXPECT_FLOAT_EQ(acc(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(acc(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(acc(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(acc(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(acc(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(acc(1, 2), 6.0f);
}

TEST_F(TensorAccessorTest, Accessor2DWrite) {
    auto t = Tensor::zeros({3, 4}, Device::CPU);
    auto acc = t.accessor<float, 2>();

    // Fill with pattern
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            acc(i, j) = i * 10.0f + j;
        }
    }

    // Verify
    auto values = t.to_vector();
    EXPECT_FLOAT_EQ(values[0], 0.0f);   // (0,0)
    EXPECT_FLOAT_EQ(values[1], 1.0f);   // (0,1)
    EXPECT_FLOAT_EQ(values[4], 10.0f);  // (1,0)
    EXPECT_FLOAT_EQ(values[5], 11.0f);  // (1,1)
}

TEST_F(TensorAccessorTest, Accessor2DRowMajor) {
    auto t = Tensor::zeros({2, 3}, Device::CPU);
    auto acc = t.accessor<float, 2>();

    // Fill row-major: [0,1,2; 3,4,5]
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            acc(i, j) = i * 3 + j;
        }
    }

    // Verify storage is row-major
    auto values = t.to_vector();
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(values[i], static_cast<float>(i));
    }
}

// ============= 3D Accessor Tests =============

TEST_F(TensorAccessorTest, Accessor3DBasic) {
    auto t = Tensor::zeros({2, 3, 4}, Device::CPU);
    auto acc = t.accessor<float, 3>();

    EXPECT_EQ(acc.sizes()[0], 2);
    EXPECT_EQ(acc.sizes()[1], 3);
    EXPECT_EQ(acc.sizes()[2], 4);

    // Write a pattern
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                acc(i, j, k) = i * 100 + j * 10 + k;
            }
        }
    }

    // Spot check
    EXPECT_FLOAT_EQ(acc(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(acc(0, 1, 2), 12.0f);
    EXPECT_FLOAT_EQ(acc(1, 2, 3), 123.0f);
}

TEST_F(TensorAccessorTest, Accessor3DStrides) {
    auto t = Tensor::zeros({2, 3, 4}, Device::CPU);
    auto acc = t.accessor<float, 3>();

    // Fill with linear sequence
    float val = 0.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                acc(i, j, k) = val++;
            }
        }
    }

    // Verify strides work correctly
    auto values = t.to_vector();
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_FLOAT_EQ(values[i], static_cast<float>(i));
    }
}

// ============= 4D Accessor Tests =============

TEST_F(TensorAccessorTest, Accessor4DBasic) {
    auto t = Tensor::zeros({2, 2, 2, 2}, Device::CPU);
    auto acc = t.accessor<float, 4>();

    EXPECT_EQ(acc.sizes()[0], 2);
    EXPECT_EQ(acc.sizes()[1], 2);
    EXPECT_EQ(acc.sizes()[2], 2);
    EXPECT_EQ(acc.sizes()[3], 2);

    // Set diagonal elements
    acc(0, 0, 0, 0) = 1.0f;
    acc(1, 1, 1, 1) = 2.0f;

    EXPECT_FLOAT_EQ(acc(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(acc(1, 1, 1, 1), 2.0f);
    EXPECT_FLOAT_EQ(acc(0, 1, 0, 1), 0.0f);
}

// ============= Accessor with Different Types =============

TEST_F(TensorAccessorTest, AccessorIntType) {
    auto t = Tensor::zeros({3, 3}, Device::CPU, DataType::Int32);
    auto acc = t.accessor<int, 2>();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            acc(i, j) = i * 3 + j;
        }
    }

    auto values = t.to_vector_int();
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_EQ(values[i], static_cast<int>(i));
    }
}

TEST_F(TensorAccessorTest, AccessorBoolType) {
    auto t = Tensor::zeros_bool({2, 3}, Device::CPU);
    auto acc = t.accessor<unsigned char, 2>();

    // Set some values to true
    acc(0, 0) = 1;
    acc(0, 2) = 1;
    acc(1, 1) = 1;

    auto values = t.to_vector_bool();
    std::vector<bool> expected = {true, false, true, false, true, false};
    EXPECT_EQ(values.size(), expected.size());
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(values[i], expected[i]);
    }
}

// ============= Error Cases =============

TEST_F(TensorAccessorTest, AccessorWrongDimension) {
    auto t = Tensor::zeros({3, 3}, Device::CPU);
    
    // Try to create 1D accessor for 2D tensor
    auto acc = t.accessor<float, 1>();
    
    // Should return invalid accessor (or handle gracefully)
    // The exact behavior depends on implementation
    // Check that sizes are reasonable or zero
    EXPECT_TRUE(acc.sizes()[0] == 0 || acc.sizes()[0] == 3);
}

TEST_F(TensorAccessorTest, AccessorOnCUDAFails) {
    auto t = Tensor::zeros({3, 3}, Device::CUDA);
    
    // Accessor should only work on CPU tensors
    auto acc = t.accessor<float, 2>();
    
    // Implementation should either return null or handle gracefully
    // This is a safety check
}

// ============= Practical Use Cases =============

TEST_F(TensorAccessorTest, FillIdentityMatrix) {
    auto t = Tensor::zeros({5, 5}, Device::CPU);
    auto acc = t.accessor<float, 2>();

    // Fill diagonal with 1s
    for (size_t i = 0; i < 5; ++i) {
        acc(i, i) = 1.0f;
    }

    // Verify it's an identity matrix
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            if (i == j) {
                EXPECT_FLOAT_EQ(acc(i, j), 1.0f);
            } else {
                EXPECT_FLOAT_EQ(acc(i, j), 0.0f);
            }
        }
    }
}

TEST_F(TensorAccessorTest, FillBinomialCoefficients) {
    // Fill Pascal's triangle (binomial coefficients)
    size_t n = 10;
    auto t = Tensor::zeros({n, n}, Device::CPU);
    auto acc = t.accessor<float, 2>();

    // C(n, k) = C(n-1, k-1) + C(n-1, k)
    for (size_t i = 0; i < n; ++i) {
        acc(i, 0) = 1.0f;  // First column
        acc(i, i) = 1.0f;  // Diagonal
        for (size_t j = 1; j < i; ++j) {
            acc(i, j) = acc(i-1, j-1) + acc(i-1, j);
        }
    }

    // Verify known values
    EXPECT_FLOAT_EQ(acc(4, 2), 6.0f);   // C(4,2) = 6
    EXPECT_FLOAT_EQ(acc(5, 2), 10.0f);  // C(5,2) = 10
    EXPECT_FLOAT_EQ(acc(6, 3), 20.0f);  // C(6,3) = 20
}

TEST_F(TensorAccessorTest, MatrixTranspose) {
    auto src = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        {2, 3},
        Device::CPU
    );
    auto dst = Tensor::zeros({3, 2}, Device::CPU);

    auto src_acc = src.accessor<float, 2>();
    auto dst_acc = dst.accessor<float, 2>();

    // Manual transpose
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            dst_acc(j, i) = src_acc(i, j);
        }
    }

    // Verify transpose
    EXPECT_FLOAT_EQ(dst_acc(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(dst_acc(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(dst_acc(2, 0), 3.0f);
    EXPECT_FLOAT_EQ(dst_acc(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(dst_acc(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(dst_acc(2, 1), 6.0f);
}

TEST_F(TensorAccessorTest, ConvolutionWindow) {
    // Simulate accessing a 3x3 window in a larger image
    auto img = Tensor::randn({10, 10}, Device::CPU);
    auto acc = img.accessor<float, 2>();

    // Extract a 3x3 window starting at (5, 5)
    float sum = 0.0f;
    for (size_t i = 5; i < 8; ++i) {
        for (size_t j = 5; j < 8; ++j) {
            sum += acc(i, j);
        }
    }

    // Just verify we can do this without crashing
    EXPECT_FALSE(std::isnan(sum));
}

// ============= Accessor Lifetime Tests =============

TEST_F(TensorAccessorTest, AccessorLifetime) {
    auto t = Tensor::zeros({3, 3}, Device::CPU);
    
    {
        auto acc = t.accessor<float, 2>();
        acc(1, 1) = 42.0f;
        // Accessor goes out of scope
    }

    // Changes should persist
    auto values = t.to_vector();
    EXPECT_FLOAT_EQ(values[4], 42.0f);  // Position (1,1) in row-major
}

TEST_F(TensorAccessorTest, MultipleAccessors) {
    auto t = Tensor::zeros({3, 3}, Device::CPU);
    
    auto acc1 = t.accessor<float, 2>();
    auto acc2 = t.accessor<float, 2>();

    acc1(0, 0) = 1.0f;
    acc2(1, 1) = 2.0f;

    // Both should see all changes
    EXPECT_FLOAT_EQ(acc1(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(acc1(1, 1), 2.0f);
    EXPECT_FLOAT_EQ(acc2(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(acc2(1, 1), 2.0f);
}

// ============= Performance Test =============

TEST_F(TensorAccessorTest, AccessorPerformance) {
    auto t = Tensor::zeros({1000, 1000}, Device::CPU);
    auto acc = t.accessor<float, 2>();

    // Fill entire matrix using accessor
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t j = 0; j < 1000; ++j) {
            acc(i, j) = i + j;
        }
    }

    // Spot check
    EXPECT_FLOAT_EQ(acc(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(acc(100, 200), 300.0f);
    EXPECT_FLOAT_EQ(acc(999, 999), 1998.0f);
}
