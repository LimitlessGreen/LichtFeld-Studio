/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include "core_new/tensor.hpp"
#include <vector>

using namespace lfs::core;

// ===================================================================================
// Bool Tensor Conversion Tests
// ===================================================================================

TEST(TensorBoolTest, Int32ToBoolConversion) {
    // Create int32 tensor and convert to bool
    std::vector<int32_t> int_vec = {0, 1, 0, 1, 1};
    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({5}), Device::CUDA);
    auto bool_tensor = int_tensor.to(DataType::Bool);
    
    // Convert back to int32 to verify
    auto back_to_int = bool_tensor.to(DataType::Int32);
    auto result_vec = back_to_int.cpu().to_vector();
    
    EXPECT_EQ(result_vec.size(), 5);
    EXPECT_FLOAT_EQ(result_vec[0], 0.0f);
    EXPECT_FLOAT_EQ(result_vec[1], 1.0f);
    EXPECT_FLOAT_EQ(result_vec[2], 0.0f);
    EXPECT_FLOAT_EQ(result_vec[3], 1.0f);
    EXPECT_FLOAT_EQ(result_vec[4], 1.0f);
}

TEST(TensorBoolTest, BoolTensorSum) {
    // Create bool tensor and sum it
    std::vector<int32_t> int_vec = {0, 1, 0, 1, 1};
    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({5}), Device::CUDA);
    auto bool_tensor = int_tensor.to(DataType::Bool);

    // Sum should count the number of true values
    auto bool_as_int = bool_tensor.to(DataType::Int32);
    int sum = bool_as_int.sum().item<int>();

    EXPECT_EQ(sum, 3) << "Bool tensor should have 3 true values";
}

TEST(TensorBoolTest, BoolZerosThenFill) {
    // Create bool zeros then try to fill
    auto bool_zeros = Tensor::zeros({10}, Device::CUDA, DataType::Bool);
    
    // Try to fill a slice with true
    bool_zeros.slice(0, 2, 7).fill_(true);
    
    // Check result
    auto result = bool_zeros.to(DataType::Int32).cpu().to_vector();
    
    EXPECT_EQ(result.size(), 10);
    EXPECT_FLOAT_EQ(result[0], 0.0f);  // Before slice
    EXPECT_FLOAT_EQ(result[1], 0.0f);  // Before slice
    EXPECT_FLOAT_EQ(result[2], 1.0f);  // In slice
    EXPECT_FLOAT_EQ(result[3], 1.0f);  // In slice
    EXPECT_FLOAT_EQ(result[4], 1.0f);  // In slice
    EXPECT_FLOAT_EQ(result[5], 1.0f);  // In slice
    EXPECT_FLOAT_EQ(result[6], 1.0f);  // In slice
    EXPECT_FLOAT_EQ(result[7], 0.0f);  // After slice
    EXPECT_FLOAT_EQ(result[8], 0.0f);  // After slice
    EXPECT_FLOAT_EQ(result[9], 0.0f);  // After slice
}

TEST(TensorBoolTest, BoolVectorDirectCreation) {
    // Try to create bool tensor directly from bool vector
    std::vector<bool> bool_vec = {false, true, false, true, true};

    auto bool_tensor = Tensor::from_vector(bool_vec, TensorShape({5}), Device::CUDA);
    auto result = bool_tensor.to(DataType::Int32).cpu().to_vector();

    EXPECT_EQ(result.size(), 5);
    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 1.0f);
    EXPECT_FLOAT_EQ(result[2], 0.0f);
    EXPECT_FLOAT_EQ(result[3], 1.0f);
    EXPECT_FLOAT_EQ(result[4], 1.0f);
}

TEST(TensorBoolTest, LogicalNot) {
    // Test logical_not operation
    std::vector<int32_t> int_vec = {0, 1, 0, 1, 1};
    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({5}), Device::CUDA);
    auto bool_tensor = int_tensor.to(DataType::Bool);
    
    auto bool_not = bool_tensor.logical_not();
    auto result = bool_not.to(DataType::Int32).cpu().to_vector();
    
    EXPECT_EQ(result.size(), 5);
    EXPECT_FLOAT_EQ(result[0], 1.0f);  // NOT 0 = 1
    EXPECT_FLOAT_EQ(result[1], 0.0f);  // NOT 1 = 0
    EXPECT_FLOAT_EQ(result[2], 1.0f);  // NOT 0 = 1
    EXPECT_FLOAT_EQ(result[3], 0.0f);  // NOT 1 = 0
    EXPECT_FLOAT_EQ(result[4], 0.0f);  // NOT 1 = 0
}

TEST(TensorBoolTest, LogicalOr) {
    // Test logical_or operation
    std::vector<int32_t> vec1 = {0, 1, 0, 1};
    std::vector<int32_t> vec2 = {0, 0, 1, 1};
    
    auto tensor1 = Tensor::from_vector(vec1, TensorShape({4}), Device::CUDA).to(DataType::Bool);
    auto tensor2 = Tensor::from_vector(vec2, TensorShape({4}), Device::CUDA).to(DataType::Bool);
    
    auto result_tensor = tensor1.logical_or(tensor2);
    auto result = result_tensor.to(DataType::Int32).cpu().to_vector();
    
    EXPECT_EQ(result.size(), 4);
    EXPECT_FLOAT_EQ(result[0], 0.0f);  // 0 OR 0 = 0
    EXPECT_FLOAT_EQ(result[1], 1.0f);  // 1 OR 0 = 1
    EXPECT_FLOAT_EQ(result[2], 1.0f);  // 0 OR 1 = 1
    EXPECT_FLOAT_EQ(result[3], 1.0f);  // 1 OR 1 = 1
}

TEST(TensorBoolTest, LogicalAnd) {
    // Test logical_and operation
    std::vector<int32_t> vec1 = {0, 1, 0, 1};
    std::vector<int32_t> vec2 = {0, 0, 1, 1};
    
    auto tensor1 = Tensor::from_vector(vec1, TensorShape({4}), Device::CUDA).to(DataType::Bool);
    auto tensor2 = Tensor::from_vector(vec2, TensorShape({4}), Device::CUDA).to(DataType::Bool);
    
    auto result_tensor = tensor1.logical_and(tensor2);
    auto result = result_tensor.to(DataType::Int32).cpu().to_vector();
    
    EXPECT_EQ(result.size(), 4);
    EXPECT_FLOAT_EQ(result[0], 0.0f);  // 0 AND 0 = 0
    EXPECT_FLOAT_EQ(result[1], 0.0f);  // 1 AND 0 = 0
    EXPECT_FLOAT_EQ(result[2], 0.0f);  // 0 AND 1 = 0
    EXPECT_FLOAT_EQ(result[3], 1.0f);  // 1 AND 1 = 1
}

TEST(TensorBoolTest, NonzeroOnBoolTensor) {
    // Test nonzero() on bool tensor
    std::vector<int32_t> int_vec = {0, 1, 0, 1, 1, 0, 1};
    auto int_tensor = Tensor::from_vector(int_vec, TensorShape({7}), Device::CUDA);
    auto bool_tensor = int_tensor.to(DataType::Bool);
    
    auto indices = bool_tensor.nonzero();
    
    // Should return a 2D tensor of shape [num_nonzero, 1] for 1D input
    EXPECT_EQ(indices.ndim(), 2);
    
    // Squeeze to get 1D indices
    auto indices_1d = indices.squeeze(-1);
    auto result = indices_1d.cpu().to_vector();
    
    // Should find indices: 1, 3, 4, 6
    EXPECT_EQ(result.size(), 4);
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 3.0f);
    EXPECT_FLOAT_EQ(result[2], 4.0f);
    EXPECT_FLOAT_EQ(result[3], 6.0f);
}

TEST(TensorBoolTest, BoolIndexing) {
    // Test using bool tensor for indexing
    auto data = Tensor::from_vector(std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f, 50.0f},
                                     TensorShape({5}), Device::CUDA);

    std::vector<int32_t> mask_vec = {1, 0, 1, 0, 1};
    auto mask_int = Tensor::from_vector(mask_vec, TensorShape({5}), Device::CUDA);
    auto mask = mask_int.to(DataType::Bool);

    // Use index_select with nonzero indices
    auto indices = mask.nonzero().squeeze(-1);
    auto selected = data.index_select(0, indices);

    auto result = selected.cpu().to_vector();
    EXPECT_EQ(result.size(), 3);
    EXPECT_FLOAT_EQ(result[0], 10.0f);
    EXPECT_FLOAT_EQ(result[1], 30.0f);
    EXPECT_FLOAT_EQ(result[2], 50.0f);
}

TEST(TensorBoolTest, BoolComparisonResult) {
    // Test that comparison operations return proper bool tensors
    auto tensor = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                                      TensorShape({5}), Device::CUDA);
    
    auto mask = tensor > 3.0f;  // Should be bool tensor
    
    // Convert to int to check
    auto result = mask.to(DataType::Int32).cpu().to_vector();
    
    EXPECT_EQ(result.size(), 5);
    EXPECT_FLOAT_EQ(result[0], 0.0f);  // 1 > 3 = false
    EXPECT_FLOAT_EQ(result[1], 0.0f);  // 2 > 3 = false
    EXPECT_FLOAT_EQ(result[2], 0.0f);  // 3 > 3 = false
    EXPECT_FLOAT_EQ(result[3], 1.0f);  // 4 > 3 = true
    EXPECT_FLOAT_EQ(result[4], 1.0f);  // 5 > 3 = true
}

TEST(TensorBoolTest, MCMCRemoveGaussiansScenario) {
    // Reproduce the exact scenario from MCMC remove_gaussians
    const int N = 100;
    
    // Create mask to remove indices 10-39 (30 elements)
    std::vector<int32_t> mask_vec(N, 0);
    for (int i = 10; i < 40; i++) {
        mask_vec[i] = 1;
    }
    
    auto mask_int = Tensor::from_vector(mask_vec, TensorShape({N}), Device::CUDA);
    auto mask = mask_int.to(DataType::Bool);

    // Convert to int and sum (this is what MCMC does)
    auto mask_int_back = mask.to(DataType::Int32);
    int n_remove = mask_int_back.sum().item<int>();

    EXPECT_EQ(n_remove, 30) << "Should detect 30 Gaussians to remove";
    
    // Get keep indices
    auto keep_mask = mask.logical_not();
    auto keep_indices = keep_mask.nonzero().squeeze(-1);
    
    EXPECT_EQ(keep_indices.numel(), 70) << "Should have 70 Gaussians to keep";
}

