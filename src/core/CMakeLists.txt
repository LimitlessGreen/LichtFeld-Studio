# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# C++ sources (can use C++23)
set(CORE_SOURCES
        application.cpp
        argument_parser.cpp
        camera.cpp
        camera_ref.cpp
        camera_new.cpp
        image_io.cpp
        image_io_new.cpp
        parameters.cpp
        pinned_memory_allocator.cpp    # Pinned memory allocator for fast CPU-GPU transfers
        splat_data.cpp
        splat_data_ref.cpp
        splat_data_new.cpp
        sogs.cpp
        sogs_new.cpp
        tensor/tensor.cpp              # Main tensor implementation (includes expr templates)
        tensor/tensor_utils.cpp        # Minimal utilities MemoryInfo)
        tensor/tensor_matrix_ops.cpp   # Matrix operations (matmul, transpose, etc.)
        tensor/tensor_unified_ops.cpp  # Unified operations (load, unary, binary, reduce, ternary)
        tensor/tensor_movement_ops.cpp # Movement operations (reshape, permute, etc.)
        tensor/tensor_random_ops.cpp   # Random generation operations
        tensor/tensor_broadcast.cpp    # Broadcasting implementation
        tensor/tensor_shape_ops.cpp    # Shape functions (reshape, slice, etc.)
        tensor/tensor_masking_ops.cpp  # Masking and indexing operations
        tensor/tensor_advanced_ops.cpp
        tinyply.cpp
)

# CUDA sources (limited to C++20)
set(CORE_CUDA_SOURCES
        tensor/tensor_ops.cu           # CUDA kernels for tensor operations
        tensor/tensor_warp_reduce.cu   # Optimized warp-level reduction kernels (tiny-cuda-nn inspired)
        tensor/tensor_matrix_ops.cu    # CUDA kernels for matrix operations
        tensor/tensor_broadcast_ops.cu # CUDA kernels for broadcasting
        tensor/tensor_masking_ops.cu   # CUDA kernels for masking/indexing
        tensor/tensor_random_ops.cu    # CUDA kernels for random ops
        tensor/tensor_strided_ops.cu   # CUDA kernels for strided tensor operations
	kmeans_new.cu
	morton_encoding_new.cu
)

add_compile_definitions(_PSTL_USAGE_WARNINGS=0)

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
            ${CMAKE_SOURCE_DIR}/src/tensor      # For inter-module includes
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
            # Windows: Ensure MSVC gets /utf-8 when invoked by nvcc
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:
                -Xcompiler=/utf-8
                -Xcompiler=/D_CRT_SECURE_NO_WARNINGS
                --diag-suppress=27  # Suppress "character value out of range" warnings from fmt
            >
    )

    # Windows: Add compile definition to tell fmt that UTF-8 is enabled for CUDA files
    if(MSVC)
        target_compile_definitions(gs_tensor_kernels PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
        )
    endif()

    # Windows NVCC ICE workaround: Reduce optimization for tensor_broadcast_ops.cu
    # NVCC 12.8 on Windows hits an internal compiler error with -O3 on complex template code
    # Error: "unexpected expression with aggregate type!" during codegen
    # Workaround: Use -O2 instead of -O3 for this file only
    if(WIN32)
        set_source_files_properties(
            tensor/tensor_broadcast_ops.cu
            PROPERTIES
            COMPILE_OPTIONS "$<$<CONFIG:Release>:-O2;--ptxas-options=-v>"
        )
    endif()
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
        ${CMAKE_SOURCE_DIR}/external  # For tinyply and STB headers
        ${OPENGL_INCLUDE_DIRS}
)

# Find the actual WebP library target
if(TARGET WebP::webp)
    set(WEBP_LIB WebP::webp)
elseif(TARGET WebP::libwebp)
    set(WEBP_LIB WebP::libwebp)
elseif(TARGET webp)
    set(WEBP_LIB webp)
else()
    find_library(WEBP_LIBRARY NAMES webp libwebp REQUIRED)
    set(WEBP_LIB ${WEBP_LIBRARY})
endif()

# Find the actual LibArchive library target
if(TARGET LibArchive::LibArchive)
    set(ARCHIVE_LIB LibArchive::LibArchive)
elseif(TARGET archive)
    set(ARCHIVE_LIB archive)
elseif(TARGET archive_static)
    set(ARCHIVE_LIB archive_static)
else()
    find_library(ARCHIVE_LIBRARY NAMES archive libarchive REQUIRED)
    set(ARCHIVE_LIB ${ARCHIVE_LIBRARY})
endif()

target_link_libraries(gs_core
        PUBLIC
        gs_tensor_kernels     # Link CUDA tensor kernels
        ${TORCH_LIBRARIES}
        TBB::tbb
        nlohmann_json::nlohmann_json
        glm::glm
        Threads::Threads
        spdlog::spdlog
        OpenImageIO::OpenImageIO
        CUDA::cudart
        CUDA::cublas          # For matrix operations
        CUDA::curand          # For random operations
        ${WEBP_LIB}           # WebP for SOG compression
        ${ARCHIVE_LIB}        # LibArchive for SOG bundle support
        PRIVATE
        taywee::args          # Only used in argument_parser.cpp
)

# Platform-specific settings
if(UNIX)
    target_link_libraries(gs_core PUBLIC dl)
endif()

# Compiler options
if(MSVC)
    target_compile_options(gs_core PRIVATE
            $<$<CONFIG:Debug>:/Od /Z7>
            $<$<CONFIG:Release>:/O2 /DNDEBUG>
    )
    target_compile_definitions(gs_core PRIVATE _USE_MATH_DEFINES NOMINMAX)
else()
    target_compile_options(gs_core PRIVATE
            $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer -DDEBUG>
            $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
    )
endif()

# AVX2 support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(gs_core PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
        )
        target_compile_definitions(gs_core PRIVATE HAS_AVX2_SUPPORT)
        message(STATUS "✓ AVX2 support enabled for gs_core")
    else()
        message(WARNING "✗ Compiler does not support AVX2")
    endif()
endif()

# OpenMP support for multi-threaded CPU operations
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gs_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "✓ OpenMP support enabled for gs_core (multi-threaded .contiguous())")
else()
    message(WARNING "✗ OpenMP not found - .contiguous() will use single-threaded code")
endif()

# Debug symbols in debug builds
set_target_properties(gs_core PROPERTIES DEBUG_POSTFIX d)

# Set properties
set_target_properties(gs_core PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
        EXPORT_COMPILE_COMMANDS ON
)

# Configure build type
function(configure_build_type target)
    get_target_property(target_type ${target} TYPE)

    if(target_type STREQUAL "INTERFACE_LIBRARY")
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_definitions(${target} INTERFACE DEBUG_BUILD)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_definitions(${target} INTERFACE RELEASE_BUILD)
        endif()
    else()
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_definitions(${target} PRIVATE DEBUG_BUILD)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_definitions(${target} PRIVATE RELEASE_BUILD)
        endif()
    endif()
endfunction()

configure_build_type(gs_core)

message(STATUS "Core module configured with ${CMAKE_BUILD_TYPE} build type")
message(STATUS "  WebP library: ${WEBP_LIB}")
message(STATUS "  Archive library: ${ARCHIVE_LIB}")
message(STATUS "  Tensor kernels: CUDA C++20 (gs_tensor_kernels)")
message(STATUS "  Core library: C++23 (gs_core)")
message(STATUS "  NOTE: Using Thrust (bundled with CUDA) - no libtorch dependency")
