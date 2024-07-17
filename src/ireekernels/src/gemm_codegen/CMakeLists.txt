cmake_minimum_required(VERSION 3.13)

set(LLVM_DIR ${CMAKE_SOURCE_DIR}/../../third_party/llvm-project/build/lib/cmake/llvm)
set(MLIR_DIR ${CMAKE_SOURCE_DIR}/../../third_party/llvm-project/build/lib/cmake/mlir)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(TableGen)

add_definitions(${LLVM_DEFINITIONS})
link_directories(${LLVM_LIBRARY_DIRS})

set(_CODEGEN_NAME gemm_mlir)
add_library(${_CODEGEN_NAME}_lib SHARED
  gemm_mlir.cpp
)
llvm_update_compile_flags(${_CODEGEN_NAME}_lib)


target_link_libraries(${_CODEGEN_NAME}_lib
  PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
)

target_include_directories(${_CODEGEN_NAME}_lib PUBLIC 
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  $<INSTALL_INTERFACE:include>
)

add_executable(${_CODEGEN_NAME} driver.cpp)
set_target_properties(${_CODEGEN_NAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${RUNTIME_OUTPUT_PATH}"
)
install(TARGETS ${_CODEGEN_NAME}
  RUNTIME DESTINATION bin
)
target_link_libraries(${_CODEGEN_NAME} ${_CODEGEN_NAME}_lib)
