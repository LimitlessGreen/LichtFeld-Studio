# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Optimizer sources
set(TRAINING_NEW_SOURCES
    optimizer/adam_optimizer.cpp
    optimizer/scheduler.cpp
    losses/regularization.cpp
    losses/photometric_loss.cpp
)

# lfs_training - Training module with optimizer
add_library(lfs_training STATIC ${TRAINING_NEW_SOURCES})

target_include_directories(lfs_training
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/training/fastgs  # For adam_api.h
)

# Link dependencies
target_link_libraries(lfs_training
    PUBLIC
        lfs_core              # Depends on new core module (SplatData, Tensor)
        fastgs_backend        # For adam_step_raw CUDA kernel
        gs_training_kernels   # For SSIM and regularization CUDA kernels
        ${TORCH_LIBRARIES}    # Needed for torch::Tensor wrappers in loss implementations
)

# Set C++ standard
set_target_properties(lfs_training PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

message(STATUS "╔════════════════════════════════════════════════════════════════╗")
message(STATUS "║  lfs_training - Training Module with Optimizer & Losses       ║")
message(STATUS "╚════════════════════════════════════════════════════════════════╝")
message(STATUS "  • Contains: AdamOptimizer, LR Schedulers, Losses (LibTorch-free API)")
message(STATUS "  • Namespace: lfs::training")
message(STATUS "  • Uses: CUDA kernels from gs_training_kernels, fastgs_backend")
