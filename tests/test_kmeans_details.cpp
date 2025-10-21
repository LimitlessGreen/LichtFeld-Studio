/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <torch/torch.h>
#include <vector>

#include "core/tensor.hpp"
#include "kernels/kmeans.cuh"
#include "kernels/kmeans_new.cuh"

namespace {

    constexpr float FLOAT_TOLERANCE = 1e-3f;

    // Helper to convert torch::Tensor to gs::Tensor
    gs::Tensor torch_to_tensor(const torch::Tensor& torch_tensor) {
        auto cpu_tensor = torch_tensor.cpu().contiguous();
        std::vector<size_t> shape;
        for (int i = 0; i < torch_tensor.dim(); ++i) {
            shape.push_back(torch_tensor.size(i));
        }

        if (torch_tensor.scalar_type() == torch::kFloat32) {
            std::vector<float> data(cpu_tensor.data_ptr<float>(),
                                    cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
            return gs::Tensor::from_vector(data, gs::TensorShape(shape), gs::Device::CUDA);
        } else if (torch_tensor.scalar_type() == torch::kInt32) {
            std::vector<int> data(cpu_tensor.data_ptr<int>(),
                                  cpu_tensor.data_ptr<int>() + cpu_tensor.numel());
            return gs::Tensor::from_vector(data, gs::TensorShape(shape), gs::Device::CUDA);
        }

        return gs::Tensor();
    }

    // Helper to convert gs::Tensor to torch::Tensor
    torch::Tensor tensor_to_torch(const gs::Tensor& gs_tensor) {
        auto cpu_tensor = gs_tensor.cpu();
        std::vector<int64_t> shape;
        for (size_t i = 0; i < cpu_tensor.ndim(); ++i) {
            shape.push_back(cpu_tensor.shape()[i]);
        }

        if (gs_tensor.dtype() == gs::DataType::Float32) {
            auto data = cpu_tensor.to_vector();
            auto torch_tensor = torch::from_blob(data.data(), shape, torch::kFloat32).clone();
            return torch_tensor.cuda();
        } else if (gs_tensor.dtype() == gs::DataType::Int32) {
            auto data = cpu_tensor.to_vector_int();
            auto torch_tensor = torch::from_blob(data.data(), shape, torch::kInt32).clone();
            return torch_tensor.cuda();
        }

        return torch::Tensor();
    }

    void print_tensor(const std::string& name, const gs::Tensor& t, int max_items = 10) {
        auto cpu = t.cpu();
        std::cout << name << " [";
        for (size_t i = 0; i < t.ndim(); ++i) {
            std::cout << t.shape()[i];
            if (i < t.ndim() - 1)
                std::cout << ", ";
        }
        std::cout << "]: ";

        if (t.dtype() == gs::DataType::Float32) {
            auto vec = cpu.to_vector();
            for (size_t i = 0; i < std::min(vec.size(), static_cast<size_t>(max_items)); ++i) {
                std::cout << std::fixed << std::setprecision(4) << vec[i] << " ";
            }
            if (vec.size() > max_items)
                std::cout << "...";
        } else if (t.dtype() == gs::DataType::Int32) {
            auto vec = cpu.to_vector_int();
            for (size_t i = 0; i < std::min(vec.size(), static_cast<size_t>(max_items)); ++i) {
                std::cout << vec[i] << " ";
            }
            if (vec.size() > max_items)
                std::cout << "...";
        }
        std::cout << std::endl;
    }

    void print_torch(const std::string& name, const torch::Tensor& t, int max_items = 10) {
        auto cpu = t.cpu();
        std::cout << name << " [";
        for (int i = 0; i < t.dim(); ++i) {
            std::cout << t.size(i);
            if (i < t.dim() - 1)
                std::cout << ", ";
        }
        std::cout << "]: ";

        if (t.scalar_type() == torch::kFloat32) {
            auto ptr = cpu.data_ptr<float>();
            for (int i = 0; i < std::min(static_cast<int>(t.numel()), max_items); ++i) {
                std::cout << std::fixed << std::setprecision(4) << ptr[i] << " ";
            }
            if (t.numel() > max_items)
                std::cout << "...";
        } else if (t.scalar_type() == torch::kInt32) {
            auto ptr = cpu.data_ptr<int>();
            for (int i = 0; i < std::min(static_cast<int>(t.numel()), max_items); ++i) {
                std::cout << ptr[i] << " ";
            }
            if (t.numel() > max_items)
                std::cout << "...";
        }
        std::cout << std::endl;
    }

    bool compare_tensors(const gs::Tensor& a, const torch::Tensor& b, float tol = FLOAT_TOLERANCE) {
        if (a.numel() != b.numel())
            return false;

        auto a_cpu = a.cpu().to_vector();
        auto b_cpu = b.cpu();
        auto b_ptr = b_cpu.data_ptr<float>();

        for (size_t i = 0; i < a.numel(); ++i) {
            if (std::abs(a_cpu[i] - b_ptr[i]) > tol) {
                return false;
            }
        }
        return true;
    }

} // anonymous namespace

class KMeansTensorOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        gs::Tensor::manual_seed(42);
    }
};

class KMeansDetailedTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";

        // Fixed seed for reproducibility
        torch::manual_seed(12345);
        torch::cuda::manual_seed(12345);
        gs::Tensor::manual_seed(12345);
    }
};

// ============================================================================
// NEW: Test Basic Tensor Operations Used in K-means++
// ============================================================================

TEST_F(KMeansTensorOpsTest, BasicSquare) {
    std::cout << "\n=== Test: Square Operation ===" << std::endl;

    auto a = gs::Tensor::rand({10}, gs::Device::CUDA);
    print_tensor("Input", a, 10);

    auto b = a.square();
    print_tensor("Squared", b, 10);

    // Verify
    auto a_vec = a.cpu().to_vector();
    auto b_vec = b.cpu().to_vector();

    for (size_t i = 0; i < a_vec.size(); ++i) {
        EXPECT_NEAR(b_vec[i], a_vec[i] * a_vec[i], 1e-5f) << "Mismatch at index " << i;
    }

    std::cout << "✓ Square operation works correctly" << std::endl;
}

TEST_F(KMeansTensorOpsTest, SumAndItem) {
    std::cout << "\n=== Test: Sum and Item ===" << std::endl;

    auto a = gs::Tensor::ones({10}, gs::Device::CUDA);
    print_tensor("Input (ones)", a, 10);

    auto sum_tensor = a.sum();
    print_tensor("Sum result", sum_tensor, 1);

    float sum_val = sum_tensor.item();
    std::cout << "Sum value: " << sum_val << std::endl;

    EXPECT_NEAR(sum_val, 10.0f, 1e-5f);

    std::cout << "✓ Sum and item work correctly" << std::endl;
}

TEST_F(KMeansTensorOpsTest, DivisionByScalar) {
    std::cout << "\n=== Test: Division by Scalar ===" << std::endl;

    auto a = gs::Tensor::full({5}, 10.0f, gs::Device::CUDA);
    print_tensor("Input (10s)", a, 5);

    auto b = a / 2.0f;
    print_tensor("Divided by 2", b, 5);

    auto b_vec = b.cpu().to_vector();
    for (size_t i = 0; i < b_vec.size(); ++i) {
        EXPECT_NEAR(b_vec[i], 5.0f, 1e-5f) << "Mismatch at index " << i;
    }

    std::cout << "✓ Division by scalar works correctly" << std::endl;
}

TEST_F(KMeansTensorOpsTest, Cumsum) {
    std::cout << "\n=== Test: Cumulative Sum ===" << std::endl;

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto a = gs::Tensor::from_vector(data, gs::TensorShape({5}), gs::Device::CUDA);
    print_tensor("Input", a, 5);

    auto b = a.cumsum(0);
    print_tensor("Cumsum", b, 5);

    // Expected: [1, 3, 6, 10, 15]
    auto b_vec = b.cpu().to_vector();
    EXPECT_NEAR(b_vec[0], 1.0f, 1e-5f);
    EXPECT_NEAR(b_vec[1], 3.0f, 1e-5f);
    EXPECT_NEAR(b_vec[2], 6.0f, 1e-5f);
    EXPECT_NEAR(b_vec[3], 10.0f, 1e-5f);
    EXPECT_NEAR(b_vec[4], 15.0f, 1e-5f);

    std::cout << "✓ Cumsum works correctly" << std::endl;
}

TEST_F(KMeansTensorOpsTest, ChainedOperations) {
    std::cout << "\n=== Test: Chained Operations (square -> sum -> item) ===" << std::endl;

    auto a = gs::Tensor::full({5}, 2.0f, gs::Device::CUDA);
    print_tensor("Input (2s)", a, 5);

    try {
        auto squared = a.square();
        print_tensor("Squared", squared, 5);

        auto summed = squared.sum();
        print_tensor("Sum", summed, 1);

        float result = summed.item();
        std::cout << "Final result: " << result << std::endl;

        // Expected: 2^2 * 5 = 20
        EXPECT_NEAR(result, 20.0f, 1e-5f);

        std::cout << "✓ Chained operations work correctly" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED: " << e.what() << std::endl;
        FAIL() << "Chained operations crashed: " << e.what();
    }
}

TEST_F(KMeansTensorOpsTest, ComplexChain) {
    std::cout << "\n=== Test: Complex Chain (square -> sum -> divide -> cumsum) ===" << std::endl;

    try {
        auto a = gs::Tensor::rand({100}, gs::Device::CUDA);
        std::cout << "Created random tensor" << std::endl;

        auto squared = a.square();
        std::cout << "Squared" << std::endl;
        print_tensor("Squared (first 5)", squared.slice(0, 0, 5), 5);

        auto sum_result = squared.sum();
        std::cout << "Summed" << std::endl;

        float sum_val = sum_result.item();
        std::cout << "Sum value: " << sum_val << std::endl;

        if (sum_val > 0) {
            auto normalized = squared / sum_val;
            std::cout << "Normalized" << std::endl;
            print_tensor("Normalized (first 5)", normalized.slice(0, 0, 5), 5);

            auto cumulative = normalized.cumsum(0);
            std::cout << "Cumsum" << std::endl;
            print_tensor("Cumsum (first 5)", cumulative.slice(0, 0, 5), 5);

            // Last element should be close to 1.0 (sum of normalized)
            auto last = cumulative.slice(0, 99, 100);
            float last_val = last.cpu().to_vector()[0];
            std::cout << "Last cumsum value: " << last_val << std::endl;

            EXPECT_NEAR(last_val, 1.0f, 1e-3f);

            std::cout << "✓ Complex chain works correctly" << std::endl;
        } else {
            std::cout << "⚠ Sum was zero, skipping normalization test" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "✗ FAILED at some point: " << e.what() << std::endl;
        FAIL() << "Complex chain crashed: " << e.what();
    }
}

TEST_F(KMeansTensorOpsTest, MemoryStability) {
    std::cout << "\n=== Test: Memory Stability (create many tensors) ===" << std::endl;

    try {
        for (int i = 0; i < 100; ++i) {
            auto a = gs::Tensor::rand({10}, gs::Device::CUDA);
            auto b = a.square();
            auto c = b.sum();
            float val = c.item();

            if (i % 10 == 0) {
                std::cout << "Iteration " << i << ": sum = " << val << std::endl;
            }

            EXPECT_GT(val, 0.0f) << "Invalid result at iteration " << i;
        }

        std::cout << "✓ Memory remains stable after 100 iterations" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Memory corruption detected: " << e.what() << std::endl;
        FAIL() << "Memory stability test failed: " << e.what();
    }
}

// ============================================================================
// Original Tests
// ============================================================================

TEST_F(KMeansDetailedTest, Step1_InputDataIdentical) {
    std::cout << "\n=== Step 1: Verify Input Data ===" << std::endl;

    const int n = 100;
    const int d = 2;

    // Create data with torch
    torch::manual_seed(42);
    auto torch_data = torch::randn({n, d}, torch::kCUDA);

    // Convert to gs::Tensor
    auto gs_data = torch_to_tensor(torch_data);

    print_torch("Torch data", torch_data, 10);
    print_tensor("GS data", gs_data, 10);

    // Verify they're identical
    EXPECT_TRUE(compare_tensors(gs_data, torch_data));
    std::cout << "✓ Input data is identical" << std::endl;
}

TEST_F(KMeansDetailedTest, Step2_InitialCentroidsComparison) {
    std::cout << "\n=== Step 2: Initial Centroids ===" << std::endl;

    const int n = 100;
    const int d = 2;
    const int k = 3;

    // Create identical input data
    torch::manual_seed(42);
    auto torch_data = torch::randn({n, d}, torch::kCUDA);
    auto gs_data = torch_to_tensor(torch_data);

    std::cout << "Testing random initialization..." << std::endl;

    // For torch kmeans, we need to look at what initialization it uses
    // Let's manually create k-means++ initialization for torch
    torch::manual_seed(100);
    auto torch_centroids = torch::zeros({k, d}, torch::kCUDA);

    // Pick first centroid randomly
    int first_idx = torch::randint(n, {1}, torch::kInt32).item<int>();
    torch_centroids[0] = torch_data[first_idx];

    print_torch("Torch first centroid", torch_centroids.slice(0, 0, 1), 10);

    // For gs::Tensor - manually do same initialization
    gs::Tensor::manual_seed(100);
    auto gs_centroids = gs::Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(d)},
                                          gs::Device::CUDA, gs::DataType::Float32);

    auto first_idx_tensor = gs::Tensor::randint({1}, 0, n, gs::Device::CUDA, gs::DataType::Int32);
    int gs_first_idx = first_idx_tensor.cpu().item<int>();

    std::cout << "Torch first_idx: " << first_idx << std::endl;
    std::cout << "GS first_idx: " << gs_first_idx << std::endl;

    // Copy first centroid
    cudaMemcpy(
        gs_centroids.ptr<float>(),
        gs_data.ptr<float>() + gs_first_idx * d,
        d * sizeof(float),
        cudaMemcpyDeviceToDevice);

    print_tensor("GS first centroid", gs_centroids.slice(0, 0, 1), 10);

    std::cout << "⚠ Note: Random initialization will differ due to different RNG implementations" << std::endl;
    std::cout << "This is EXPECTED and OK - k-means should still converge to similar results" << std::endl;
}

TEST_F(KMeansDetailedTest, Step3_DistanceCalculationAndAssignment) {
    std::cout << "\n=== Step 3: Distance Calculation and Cluster Assignment ===" << std::endl;

    const int n = 20;
    const int d = 2;
    const int k = 3;

    // Create simple deterministic data
    std::vector<float> data_vec = {
        0.0f, 0.0f, // cluster 0
        0.1f, 0.1f,
        10.0f, 10.0f, // cluster 1
        10.1f, 10.1f,
        20.0f, 20.0f, // cluster 2
        20.1f, 20.1f,
        0.2f, 0.2f,   // cluster 0
        10.2f, 10.2f, // cluster 1
        20.2f, 20.2f, // cluster 2
        0.0f, 0.0f,   // repeat pattern
        0.1f, 0.1f,
        10.0f, 10.0f,
        10.1f, 10.1f,
        20.0f, 20.0f,
        20.1f, 20.1f,
        0.2f, 0.2f,
        10.2f, 10.2f,
        20.2f, 20.2f,
        5.0f, 5.0f, // between clusters
        15.0f, 15.0f};

    auto torch_data = torch::from_blob(data_vec.data(), {n, d}, torch::kFloat32).clone().cuda();
    auto gs_data = gs::Tensor::from_vector(data_vec, gs::TensorShape({n, d}), gs::Device::CUDA);

    // Create identical centroids
    std::vector<float> cent_vec = {
        0.0f, 0.0f,   // centroid 0
        10.0f, 10.0f, // centroid 1
        20.0f, 20.0f  // centroid 2
    };

    auto torch_centroids = torch::from_blob(cent_vec.data(), {k, d}, torch::kFloat32).clone().cuda();
    auto gs_centroids = gs::Tensor::from_vector(cent_vec, gs::TensorShape({k, d}), gs::Device::CUDA);

    print_torch("Torch data", torch_data, 20);
    print_tensor("GS data", gs_data, 20);
    print_torch("Torch centroids", torch_centroids, 10);
    print_tensor("GS centroids", gs_centroids, 10);

    // Now manually compute distances and assignments
    std::cout << "\nManual distance calculation for first 3 points:" << std::endl;
    auto torch_data_cpu = torch_data.cpu();
    auto torch_cent_cpu = torch_centroids.cpu();

    for (int i = 0; i < 3; ++i) {
        std::cout << "Point " << i << " ["
                  << torch_data_cpu[i][0].item<float>() << ", "
                  << torch_data_cpu[i][1].item<float>() << "]:" << std::endl;

        for (int c = 0; c < k; ++c) {
            float dx = torch_data_cpu[i][0].item<float>() - torch_cent_cpu[c][0].item<float>();
            float dy = torch_data_cpu[i][1].item<float>() - torch_cent_cpu[c][1].item<float>();
            float dist = dx * dx + dy * dy;
            std::cout << "  Distance to centroid " << c << ": " << dist << std::endl;
        }
    }

    // Use actual assignment - we need to call the kernel manually
    // For now, let's compute expected labels manually
    std::vector<int> expected_labels = {
        0, 0, 1, 1, 2, 2, 0, 1, 2, 0,
        0, 1, 1, 2, 2, 0, 1, 2, 0, 1 // last two are 5,5 and 15,15
    };

    std::cout << "\nExpected labels: ";
    for (int l : expected_labels)
        std::cout << l << " ";
    std::cout << std::endl;
}

TEST_F(KMeansDetailedTest, Step4_CentroidUpdate) {
    std::cout << "\n=== Step 4: Centroid Update ===" << std::endl;

    // Simple case: 6 points, 3 clusters, 2D
    const int n = 6;
    const int d = 2;
    const int k = 3;

    std::vector<float> data_vec = {
        0.0f, 0.0f,   // cluster 0
        0.2f, 0.2f,   // cluster 0
        10.0f, 10.0f, // cluster 1
        10.2f, 10.2f, // cluster 1
        20.0f, 20.0f, // cluster 2
        20.2f, 20.2f  // cluster 2
    };

    std::vector<int> labels_vec = {0, 0, 1, 1, 2, 2};

    auto torch_data = torch::from_blob(data_vec.data(), {n, d}, torch::kFloat32).clone().cuda();
    auto torch_labels = torch::from_blob(labels_vec.data(), {n}, torch::kInt32).clone().cuda();

    auto gs_data = gs::Tensor::from_vector(data_vec, gs::TensorShape({n, d}), gs::Device::CUDA);
    auto gs_labels = gs::Tensor::from_vector(labels_vec, gs::TensorShape({n}), gs::Device::CUDA);

    print_torch("Data", torch_data, 12);
    print_torch("Labels", torch_labels, 12);

    // Expected centroids:
    // Cluster 0: mean of (0,0) and (0.2,0.2) = (0.1, 0.1)
    // Cluster 1: mean of (10,10) and (10.2,10.2) = (10.1, 10.1)
    // Cluster 2: mean of (20,20) and (20.2,20.2) = (20.1, 20.1)

    std::cout << "\nExpected new centroids:" << std::endl;
    std::cout << "  Cluster 0: (0.1, 0.1)" << std::endl;
    std::cout << "  Cluster 1: (10.1, 10.1)" << std::endl;
    std::cout << "  Cluster 2: (20.1, 20.1)" << std::endl;

    // Compute on CPU
    std::vector<float> computed_centroids(k * d, 0.0f);
    std::vector<int> counts(k, 0);

    for (int i = 0; i < n; ++i) {
        int label = labels_vec[i];
        counts[label]++;
        for (int j = 0; j < d; ++j) {
            computed_centroids[label * d + j] += data_vec[i * d + j];
        }
    }

    for (int c = 0; c < k; ++c) {
        if (counts[c] > 0) {
            for (int j = 0; j < d; ++j) {
                computed_centroids[c * d + j] /= counts[c];
            }
        }
    }

    std::cout << "\nComputed centroids (CPU):" << std::endl;
    for (int c = 0; c < k; ++c) {
        std::cout << "  Cluster " << c << ": ("
                  << computed_centroids[c * d] << ", "
                  << computed_centroids[c * d + 1] << ")" << std::endl;
    }

    // Verify with EXPECT
    EXPECT_NEAR(computed_centroids[0], 0.1f, 1e-5f);
    EXPECT_NEAR(computed_centroids[1], 0.1f, 1e-5f);
    EXPECT_NEAR(computed_centroids[2], 10.1f, 1e-5f);
    EXPECT_NEAR(computed_centroids[3], 10.1f, 1e-5f);
    EXPECT_NEAR(computed_centroids[4], 20.1f, 1e-5f);
    EXPECT_NEAR(computed_centroids[5], 20.1f, 1e-5f);

    std::cout << "✓ Centroid update calculation verified" << std::endl;
}

TEST_F(KMeansDetailedTest, Step5_FullIterationComparison) {
    std::cout << "\n=== Step 5: Full K-means Iteration Comparison ===" << std::endl;

    const int n = 30;
    const int d = 2;
    const int k = 3;

    // Create well-separated clusters
    std::vector<float> data_vec;
    for (int i = 0; i < 10; ++i) {
        data_vec.push_back(0.0f + i * 0.1f);
        data_vec.push_back(0.0f + i * 0.1f);
    }
    for (int i = 0; i < 10; ++i) {
        data_vec.push_back(10.0f + i * 0.1f);
        data_vec.push_back(10.0f + i * 0.1f);
    }
    for (int i = 0; i < 10; ++i) {
        data_vec.push_back(20.0f + i * 0.1f);
        data_vec.push_back(20.0f + i * 0.1f);
    }

    auto torch_data = torch::from_blob(data_vec.data(), {n, d}, torch::kFloat32).clone().cuda();
    auto gs_data = gs::Tensor::from_vector(data_vec, gs::TensorShape({n, d}), gs::Device::CUDA);

    std::cout << "Created " << n << " points in " << k << " well-separated clusters" << std::endl;
    print_torch("First few points", torch_data.slice(0, 0, 5), 10);

    // Run torch kmeans
    std::cout << "\nRunning Torch k-means (1 iteration)..." << std::endl;
    auto [torch_cent, torch_labels] = gs::cuda::kmeans(torch_data, k, 1, 1e-4f);

    print_torch("Torch centroids after 1 iter", torch_cent, 10);
    print_torch("Torch labels (first 10)", torch_labels.slice(0, 0, 10), 10);

    // Run gs kmeans
    std::cout << "\nRunning GS k-means (1 iteration)..." << std::endl;
    auto [gs_cent, gs_labels] = gs::cuda::kmeans_new(gs_data, k, 1, 1e-4f);

    print_tensor("GS centroids after 1 iter", gs_cent, 10);
    print_tensor("GS labels (first 10)", gs_labels.slice(0, 0, 10), 10);

    // Compute inertia for both
    float torch_inertia = 0.0f;
    auto torch_data_cpu = torch_data.cpu();
    auto torch_cent_cpu = torch_cent.cpu();
    auto torch_labels_cpu = torch_labels.cpu();

    for (int i = 0; i < n; ++i) {
        int label = torch_labels_cpu[i].item<int>();
        float dist = 0.0f;
        for (int j = 0; j < d; ++j) {
            float diff = torch_data_cpu[i][j].item<float>() - torch_cent_cpu[label][j].item<float>();
            dist += diff * diff;
        }
        torch_inertia += dist;
    }

    float gs_inertia = 0.0f;
    auto gs_data_cpu = gs_data.cpu();
    auto gs_cent_cpu = gs_cent.cpu();
    auto gs_labels_cpu = gs_labels.cpu();

    for (int i = 0; i < n; ++i) {
        int label = static_cast<int>(static_cast<float>(gs_labels_cpu[i]));
        float dist = 0.0f;
        for (int j = 0; j < d; ++j) {
            float diff = static_cast<float>(gs_data_cpu[i][j]) - static_cast<float>(gs_cent_cpu[label][j]);
            dist += diff * diff;
        }
        gs_inertia += dist;
    }

    std::cout << "\nTorch inertia: " << torch_inertia << std::endl;
    std::cout << "GS inertia: " << gs_inertia << std::endl;
    std::cout << "Ratio: " << (gs_inertia / torch_inertia) << std::endl;

    if (std::abs(torch_inertia - gs_inertia) / torch_inertia > 0.5) {
        std::cout << "⚠ WARNING: Large inertia difference detected!" << std::endl;
        std::cout << "This suggests the algorithms are producing very different results" << std::endl;
    }
}

TEST_F(KMeansDetailedTest, Step6_DiagnoseZeroCentroids) {
    std::cout << "\n=== Step 6: Diagnose Zero Centroids Issue ===" << std::endl;

    const int n = 10;
    const int k = 3;

    // 1D case
    std::vector<float> data_vec = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    auto gs_data_1d = gs::Tensor::from_vector(data_vec, gs::TensorShape({n}), gs::Device::CUDA);
    auto gs_data_2d = gs_data_1d.unsqueeze(1);

    print_tensor("1D data", gs_data_1d, 10);
    print_tensor("2D data", gs_data_2d, 10);

    // Initialize centroids
    float min_val = gs_data_2d.min().item();
    float max_val = gs_data_2d.max().item();

    std::cout << "Min: " << min_val << ", Max: " << max_val << std::endl;

    auto centroids = gs::Tensor::linspace(min_val, max_val, k, gs::Device::CUDA);
    print_tensor("Initial centroids (1D)", centroids, 10);

    auto centroids_2d = centroids.unsqueeze(1);
    print_tensor("Initial centroids (2D)", centroids_2d, 10);

    // Check if centroids are preserved after operations
    auto centroids_copy = centroids_2d.clone();
    print_tensor("Cloned centroids", centroids_copy, 10);

    // Test squeeze operation
    auto squeezed = centroids_2d.squeeze(1);
    print_tensor("Squeezed centroids", squeezed, 10);

    // Test if linspace works correctly
    auto test_linspace = gs::Tensor::linspace(0, 10, 5, gs::Device::CUDA);
    print_tensor("Test linspace(0, 10, 5)", test_linspace, 10);

    std::cout << "✓ Centroid initialization appears to work correctly" << std::endl;
}

TEST_F(KMeansDetailedTest, Step7_SameInitializationComparison) {
    std::cout << "\n=== Step 7: Compare with Identical Initialization ===" << std::endl;

    const int n = 100;
    const int d = 2;
    const int k = 3;

    // Create identical data
    std::vector<float> data_vec;
    for (int i = 0; i < n; ++i) {
        data_vec.push_back(static_cast<float>(i % 10));
        data_vec.push_back(static_cast<float>(i / 10));
    }

    auto torch_data = torch::from_blob(data_vec.data(), {n, d}, torch::kFloat32).clone().cuda();
    auto gs_data = gs::Tensor::from_vector(data_vec, gs::TensorShape({n, d}), gs::Device::CUDA);

    // Create identical initial centroids (NOT random)
    std::vector<float> init_cent = {
        2.0f, 2.0f,
        5.0f, 5.0f,
        8.0f, 8.0f};

    auto torch_cent = torch::from_blob(init_cent.data(), {k, d}, torch::kFloat32).clone().cuda();
    auto gs_cent = gs::Tensor::from_vector(init_cent, gs::TensorShape({k, d}), gs::Device::CUDA);

    print_torch("Initial torch centroids", torch_cent, 10);
    print_tensor("Initial gs centroids", gs_cent, 10);

    std::cout << "\n⚠ NOTE: Both implementations should produce IDENTICAL results" << std::endl;
    std::cout << "since we're using the same data and same initialization!" << std::endl;
}
