# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors

# SPDX-License-Identifier: GPL-3.0-or-later

# Testing module - comprehensive tests for all components

# Find GTest
find_package(GTest CONFIG REQUIRED)

# Find external libraries needed by core module
find_package(PkgConfig REQUIRED)
pkg_check_modules(WebP REQUIRED libwebp)
pkg_check_modules(LibArchive REQUIRED libarchive)

# Test sources - START WITH ONLY EXISTING FILES
set(TEST_SOURCES
    test_geometry.cpp
)

# Check if new test files exist and add them
set(OPTIONAL_TEST_FILES
    test_tensor_basic.cpp
    test_tensor_ops.cpp
    test_tensor_memory.cpp
    test_tensor_reduction.cpp
    test_tensor_views.cpp
    test_tensor_math.cpp
    test_tensor_utils.cpp
    test_tensor_torch_compat.cpp
    test_tensor_advanced.cpp
    test_tensor_stress.cpp
    test_tensor_matrix.cpp
    test_tensor_random.cpp
    test_tensor_broadcast.cpp
    test_tensor_masking.cpp
    test_tensor_bitwise.cpp
    test_tensor_accessor.cpp
    test_tensor_random_advanced.cpp
    test_tensor_indexing_advanced.cpp
    test_tensor_clamping.cpp
    test_tensor_compat.cpp
    test_tensor_cumsum_convenience.cpp
    test_tensor_conversions_shape.cpp
    test_tensor_advanced_new.cpp
    test_kmeans.cpp
    test_critical_fixes.cpp
    test_kmeans_comparison.cpp
    test_tensor_ops_comparison.cpp
    test_kmeans_details.cpp
    test_sort_minimal.cpp
    test_tensor_debug.cpp
    test_bugs_found.cpp
    test_tensor_vs_torch.cpp
    test_morton_comparison.cpp
    test_point_cloud_comparison.cpp
    test_tensor_move.cpp
    test_splatdata.cpp
    test_camera.cpp
    test_image_io_comparison.cpp
    #test_sogs_comparison.cpp
)

foreach(TEST_FILE ${OPTIONAL_TEST_FILES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}")
        list(APPEND TEST_SOURCES ${TEST_FILE})
        message(STATUS "Found test file: ${TEST_FILE}")
    else()
        message(STATUS "Test file not found (skipping): ${TEST_FILE}")
    endif()
endforeach()

# Create test executable
add_executable(lichtfeld_tests ${TEST_SOURCES})

# Set CUDA architectures
set_target_properties(lichtfeld_tests PROPERTIES
    CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(lichtfeld_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${WebP_INCLUDE_DIRS}
    ${LibArchive_INCLUDE_DIRS}
)

# Link libraries - Fix linking order and add missing dependencies
target_link_libraries(lichtfeld_tests PRIVATE
    # Core libraries (in dependency order)
    gs_visualizer
    gs_rendering
    gs_training
    gs_training_kernels
    gs_loader
    gs_project
    gs_geometry
    gs_core
    gs_tensor_kernels
    gs_kernels
    
    # Backend libraries
    gsplat_backend
    fastgs_backend
    
    # External libraries
    GTest::gtest
    GTest::gtest_main
    ${TORCH_LIBRARIES}
    ${OPENGL_LIBRARIES}
    
    # CUDA libraries
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    
    # System libraries
    spdlog::spdlog
    
    # Libraries needed by sogs.cpp
    ${WebP_LIBRARIES}
    ${LibArchive_LIBRARIES}
)

# Add link directories for external libraries
target_link_directories(lichtfeld_tests PRIVATE
    ${WebP_LIBRARY_DIRS}
    ${LibArchive_LIBRARY_DIRS}
)

# Compiler options for CUDA debugging
target_compile_options(lichtfeld_tests PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -lineinfo -Xcudafe --device-debug>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG>
    ${WebP_CFLAGS_OTHER}
    ${LibArchive_CFLAGS_OTHER}
)

# Enable testing
include(GoogleTest)
gtest_discover_tests(lichtfeld_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES TIMEOUT 30
)

# Custom target to run tests with output
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V
    DEPENDS lichtfeld_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests with verbose output"
)

# Platform-specific settings
if(WIN32)
    # Copy Torch DLLs for testing on Windows
    file(GLOB TORCH_DLLS "${Torch_DIR}/../../../lib/*.dll")
    foreach(TORCH_DLL ${TORCH_DLLS})
        add_custom_command(
            TARGET lichtfeld_tests
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_DLL}"
            "$<TARGET_FILE_DIR:lichtfeld_tests>")
    endforeach()
endif()

message(STATUS "Tests configured with ${CMAKE_BUILD_TYPE} build type")
message(STATUS "  Test sources found: ${TEST_SOURCES}")
message(STATUS "  Build with: ninja lichtfeld_tests")
message(STATUS "  Run with: ninja run_tests or ctest --output-on-failure")
