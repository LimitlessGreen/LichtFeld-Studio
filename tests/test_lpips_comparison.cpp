// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Simple test to compare TensorRT LPIPS vs LibTorch LPIPS
// This test doesn't include any training/metrics headers to avoid dependency issues

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <filesystem>

// TensorRT logger
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR)
            std::cerr << "[TRT] " << msg << std::endl;
    }
};

// Simple TensorRT LPIPS wrapper
class SimpleLPIPSTensorRT {
public:
    SimpleLPIPSTensorRT() {
        std::ifstream file("weights/lpips_vgg.trt", std::ios::binary);
        if (!file) {
            loaded_ = false;
            return;
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> data(size);
        file.read(data.data(), size);

        runtime_ = nvinfer1::createInferRuntime(logger_);
        engine_ = runtime_->deserializeCudaEngine(data.data(), size);
        context_ = engine_->createExecutionContext();
        loaded_ = (engine_ != nullptr && context_ != nullptr);
    }

    ~SimpleLPIPSTensorRT() {
        if (context_) delete context_;
        if (engine_) delete engine_;
        if (runtime_) delete runtime_;
    }

    float compute(const torch::Tensor& img1, const torch::Tensor& img2) {
        if (!loaded_) return -1.0f;

        // Normalize to [-1, 1]
        auto img1_norm = (2.0f * img1 - 1.0f).contiguous();
        auto img2_norm = (2.0f * img2 - 1.0f).contiguous();

        // Set input shapes
        int batch = img1.size(0);
        int h = img1.size(2);
        int w = img1.size(3);

        nvinfer1::Dims4 dims{batch, 3, h, w};
        context_->setInputShape(engine_->getIOTensorName(0), dims);
        context_->setInputShape(engine_->getIOTensorName(1), dims);

        // Allocate output buffer
        float output;
        void* output_buf;
        cudaMalloc(&output_buf, sizeof(float));

        // Set tensor addresses
        context_->setTensorAddress(engine_->getIOTensorName(0), img1_norm.data_ptr());
        context_->setTensorAddress(engine_->getIOTensorName(1), img2_norm.data_ptr());
        context_->setTensorAddress(engine_->getIOTensorName(2), output_buf);

        // Execute
        context_->enqueueV3(nullptr);
        cudaDeviceSynchronize();

        // Copy result
        cudaMemcpy(&output, output_buf, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(output_buf);

        return output;
    }

    bool is_loaded() const { return loaded_; }

private:
    TRTLogger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    bool loaded_ = false;
};

// Simple LibTorch LPIPS wrapper
class SimpleLPIPSTorch {
public:
    SimpleLPIPSTorch() {
        try {
            model_ = torch::jit::load("weights/lpips_vgg.pt");
            model_.eval();
            model_.to(torch::kCUDA);
            loaded_ = true;
        } catch (...) {
            loaded_ = false;
        }
    }

    float compute(const torch::Tensor& img1, const torch::Tensor& img2) {
        if (!loaded_) return -1.0f;

        // Normalize to [-1, 1]
        auto img1_norm = (2.0f * img1 - 1.0f).to(torch::kCUDA).contiguous();
        auto img2_norm = (2.0f * img2 - 1.0f).to(torch::kCUDA).contiguous();

        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(img1_norm);
        inputs.push_back(img2_norm);

        auto output = model_.forward(inputs).toTensor();
        return output.mean().item<float>();
    }

    bool is_loaded() const { return loaded_; }

private:
    torch::jit::script::Module model_;
    bool loaded_ = false;
};

// Test fixture
class LPIPSComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists("weights/lpips_vgg.trt")) {
            GTEST_SKIP() << "TensorRT engine not found. Run: scripts/convert_lpips_to_tensorrt.py";
        }
        if (!std::filesystem::exists("weights/lpips_vgg.pt")) {
            GTEST_SKIP() << "PyTorch model not found";
        }
    }
};

TEST_F(LPIPSComparisonTest, LoadModels) {
    SimpleLPIPSTorch lpips_torch;
    SimpleLPIPSTensorRT lpips_trt;

    EXPECT_TRUE(lpips_torch.is_loaded()) << "Failed to load LibTorch LPIPS";
    EXPECT_TRUE(lpips_trt.is_loaded()) << "Failed to load TensorRT LPIPS";
}

TEST_F(LPIPSComparisonTest, IdenticalImages) {
    SimpleLPIPSTorch lpips_torch;
    SimpleLPIPSTensorRT lpips_trt;

    ASSERT_TRUE(lpips_torch.is_loaded());
    ASSERT_TRUE(lpips_trt.is_loaded());

    // Create random image
    auto img = torch::rand({1, 3, 256, 256}, torch::kCUDA);

    float dist_torch = lpips_torch.compute(img, img);
    float dist_trt = lpips_trt.compute(img, img);

    // Both should be very close to 0
    EXPECT_LT(dist_torch, 1e-4f) << "LibTorch: identical images should have distance ~0";
    EXPECT_LT(dist_trt, 1e-4f) << "TensorRT: identical images should have distance ~0";
}

TEST_F(LPIPSComparisonTest, CompareRandomImages) {
    SimpleLPIPSTorch lpips_torch;
    SimpleLPIPSTensorRT lpips_trt;

    ASSERT_TRUE(lpips_torch.is_loaded());
    ASSERT_TRUE(lpips_trt.is_loaded());

    // Create two random images
    auto img1 = torch::rand({1, 3, 256, 256}, torch::kCUDA);
    auto img2 = torch::rand({1, 3, 256, 256}, torch::kCUDA);

    float dist_torch = lpips_torch.compute(img1, img2);
    float dist_trt = lpips_trt.compute(img1, img2);

    std::cout << "LibTorch LPIPS: " << dist_torch << "\n";
    std::cout << "TensorRT LPIPS: " << dist_trt << "\n";

    float diff = std::abs(dist_torch - dist_trt);
    float rel_error = diff / std::max(dist_torch, 1e-6f);

    std::cout << "Absolute difference: " << diff << "\n";
    std::cout << "Relative error: " << (rel_error * 100.0f) << "%\n";

    // Allow 5% relative error due to FP16 in TensorRT
    EXPECT_LT(rel_error, 0.05f)
        << "TensorRT should match LibTorch within 5%. "
        << "Torch: " << dist_torch << ", TRT: " << dist_trt;
}

TEST_F(LPIPSComparisonTest, MultipleComparisons) {
    SimpleLPIPSTorch lpips_torch;
    SimpleLPIPSTensorRT lpips_trt;

    ASSERT_TRUE(lpips_torch.is_loaded());
    ASSERT_TRUE(lpips_trt.is_loaded());

    float max_rel_error = 0.0f;
    int num_tests = 10;

    for (int i = 0; i < num_tests; i++) {
        auto img1 = torch::rand({1, 3, 256, 256}, torch::kCUDA);
        auto img2 = torch::rand({1, 3, 256, 256}, torch::kCUDA);

        float dist_torch = lpips_torch.compute(img1, img2);
        float dist_trt = lpips_trt.compute(img1, img2);

        float diff = std::abs(dist_torch - dist_trt);
        float rel_error = diff / std::max(dist_torch, 1e-6f);

        max_rel_error = std::max(max_rel_error, rel_error);

        std::cout << "Test " << (i+1) << ": Torch=" << dist_torch
                  << ", TRT=" << dist_trt
                  << ", RelErr=" << (rel_error*100.0f) << "%\n";
    }

    std::cout << "Maximum relative error: " << (max_rel_error * 100.0f) << "%\n";

    EXPECT_LT(max_rel_error, 0.05f)
        << "Maximum relative error should be < 5%";
}
