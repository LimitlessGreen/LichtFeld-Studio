# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Testing module - comprehensive tests for all components

# Find GTest
find_package(GTest CONFIG REQUIRED)

# Test sources - START WITH ONLY EXISTING FILES
set(TEST_SOURCES
    test_geometry.cpp
)

# Check if new test files exist and add them
set(OPTIONAL_TEST_FILES
    test_tensor_basic.cpp
    test_tensor_ops.cpp
    test_tensor_memory.cpp
    test_tensor_reductions.cpp
    test_tensor_views.cpp
    test_tensor_math.cpp
    test_tensor_utils.cpp
    test_tensor_torch_compat.cpp
    test_tensor_advanced.cpp
    test_tensor_stress.cpp
    test_tensor_matrix.cpp
    test_tensor_random.cpp
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
)

# Link libraries - Add cuBLAS and cuRAND
target_link_libraries(lichtfeld_tests PRIVATE
    gs_core
    gs_geometry
    gs_project
    gs_kernels
    gs_loader
    gs_training
    gs_rendering
    gs_visualizer
    gsplat_backend
    fastgs_backend
    GTest::gtest
    GTest::gtest_main
    ${TORCH_LIBRARIES}
    ${OPENGL_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    spdlog::spdlog
)

# Compiler options for CUDA debugging
target_compile_options(lichtfeld_tests PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -lineinfo -Xcudafe --device-debug>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG>
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
