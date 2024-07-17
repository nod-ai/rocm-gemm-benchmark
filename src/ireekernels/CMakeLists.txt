cmake_minimum_required(VERSION 3.21...3.24)

project(IREEGemm VERSION 1.0 LANGUAGES CXX C)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(RUNTIME_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(ARCHIVE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")

set(CMAKE_BUILD_TYPE Release)
set(IREE_BUILD_COMPILER ON)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)

set(IREE_TARGET_BACKEND_ROCM ON)
set(IREE_EXTERNAL_HAL_DRIVERS rocm)
set(IREE_HAL_DRIVER_HIP ON)
set(IREE_HIP_DRIVER_TARGET_CHIP gfx942)
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON)

if(CMAKE_BUILD_TYPE MATCHES MinSizeRel)
set(IREE_SIZE_OPTIMIZED ON)
endif()

set(IREE_SRC_DIR ${CMAKE_SOURCE_DIR}/../../third_party/iree)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(${IREE_SRC_DIR} ./build/iree)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/gemm_codegen)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/gemm_compile)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/gemm_benchmark)

set_target_properties(gemm_mlir_lib iree_gemm_compile_lib iree_gemm_benchmark_lib PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_PATH}
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    ARCHIVE_OUTPUT_DIRECTORY ${ARCHIVE_OUTPUT_PATH}
)

add_executable(main_driver src/driver.cpp)
set_target_properties(main_driver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_PATH}
)

target_link_libraries(main_driver PRIVATE gemm_mlir_lib iree_gemm_compile_lib iree_gemm_benchmark_lib)

target_include_directories(main_driver PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/gemm_codegen
    ${CMAKE_SOURCE_DIR}/src/gemm_compile
    ${CMAKE_SOURCE_DIR}/src/gemm_benchmark
)

install(TARGETS gemm_mlir_lib iree_gemm_compile_lib iree_gemm_benchmark_lib
    EXPORT IREEGemmTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

add_library(iree_gemm SHARED)
target_link_libraries(iree_gemm PUBLIC
    gemm_mlir_lib
    iree_gemm_compile_lib
    iree_gemm_benchmark_lib
)
set_target_properties(iree_gemm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    ARCHIVE_OUTPUT_DIRECTORY ${ARCHIVE_OUTPUT_PATH}
)
target_include_directories(iree_gemm PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS iree_gemm
    EXPORT IREEGemmTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(TARGETS main_driver
    RUNTIME DESTINATION bin
)

install(EXPORT IREEGemmTargets
    FILE IREEGemmTargets.cmake
    DESTINATION lib/cmake/IREEGemm
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ 
    DESTINATION include
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/IREEGemmConfigVersion.cmake
    COMPATIBILITY AnyNewerVersion
)

configure_file(IREEGemmConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/IREEGemmConfig.cmake
    COPYONLY
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/IREEGemmConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/IREEGemmConfigVersion.cmake
    DESTINATION lib/cmake/IREEGemm
)