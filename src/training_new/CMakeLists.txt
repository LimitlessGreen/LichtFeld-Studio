# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# LibTorch-free CUDA kernels
set(LFS_TRAINING_KERNEL_SOURCES
    kernels/ssim.cu
)

# Create CUDA kernels library
add_library(lfs_training_kernels STATIC ${LFS_TRAINING_KERNEL_SOURCES})

target_include_directories(lfs_training_kernels
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(lfs_training_kernels
    PUBLIC
        lfs_tensor       # For lfs::core::Tensor
        CUDA::cudart
)

set_target_properties(lfs_training_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

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
        lfs_core                # Depends on new core module (SplatData, Tensor)
        lfs_training_kernels    # LibTorch-free SSIM kernels
        fastgs_backend          # For adam_step_raw CUDA kernel
        gs_training_kernels     # For old regularization CUDA kernels (TODO: port)
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
