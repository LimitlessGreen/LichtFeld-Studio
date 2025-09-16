# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Core module - fundamental data structures and utilities

# C++ sources (can use C++23)
set(CORE_SOURCES
        application.cpp
        argument_parser.cpp
        camera.cpp
        image_io.cpp
        parameters.cpp
        splat_data.cpp
        sogs.cpp
        tensor.cpp              # Main tensor implementation
        tensor_ops.cpp          # Math operations with broadcasting support
        tensor_utils.cpp        # Utility functions
        tensor_matrix_ops.cpp   # Matrix operations (matmul, transpose, etc.)
        tensor_random_ops.cpp   # Random generation operations
        tensor_broadcast.cpp    # Broadcasting implementation
        tensor_math_ops.cpp     # Math functions
        tensor_reduction_ops.cpp # Reduction operations
        tinyply.cpp
)

# CUDA sources (limited to C++20)
set(CORE_CUDA_SOURCES
        tensor_ops.cu           # CUDA kernels for tensor operations
        tensor_matrix_ops.cu    # CUDA kernels for matrix operations
        tensor_broadcast_ops.cu # CUDA kernels for broadcasting (NEW)
)

# Create CUDA library for tensor operations (C++20)
if(CORE_CUDA_SOURCES)
    add_library(gs_tensor_kernels STATIC ${CORE_CUDA_SOURCES})
    
    set_target_properties(gs_tensor_kernels PROPERTIES
        CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 20  # CUDA only supports up to C++20
        CUDA_STANDARD_REQUIRED ON
    )
    
    target_include_directories(gs_tensor_kernels
        PUBLIC
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}/include
            ${CMAKE_SOURCE_DIR}/src       # For inter-module includes
        PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    target_link_libraries(gs_tensor_kernels
        PUBLIC
            CUDA::cudart
            CUDA::cublas      # For matrix operations
            CUDA::curand      # For random number generation
            CUDA::cusolver    # Optional: for advanced linear algebra
            spdlog::spdlog
    )
    
    target_compile_options(gs_tensor_kernels PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -g -G -lineinfo>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 -use_fast_math --ptxas-options=-v>
    )
endif()

# Main core library (C++23)
add_library(gs_core STATIC ${CORE_SOURCES})

target_include_directories(gs_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include  # For generated config.h
        ${CMAKE_SOURCE_DIR}/gsplat    # For gsplat headers (Common.h)
        ${CMAKE_SOURCE_DIR}/fastgs    # For fastgs headers
        ${CUDAToolkit_INCLUDE_DIRS}   # For CUDA interop in Camera
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src       # For inter-module includes (training/, visualizer/, etc.)
)

target_link_libraries(gs_core
    PUBLIC
        gs_tensor_kernels     # Link CUDA tensor kernels
        ${TORCH_LIBRARIES}
        spdlog::spdlog
        glm::glm
        OpenImageIO::OpenImageIO
        CUDA::cudart
        CUDA::cublas          # Added for matrix operations
        CUDA::curand          # Added for random operations
    PRIVATE
        # Add any private dependencies here
)

target_compile_options(gs_core PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
)

if(MSVC)
    target_compile_definitions(gs_core PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()

set_target_properties(gs_core PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
