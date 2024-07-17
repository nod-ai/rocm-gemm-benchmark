set(_COMPILE_NAME "iree_gemm_compile")
add_library(${_COMPILE_NAME}_lib SHARED gemm_compile.c)

set(_IREE_COMPILER_ROOT ${IREE_SRC_DIR}/compiler)
set(_IREE_COMPILER_API ${_IREE_COMPILER_ROOT}/bindings/c)

target_link_libraries(${_COMPILE_NAME}_lib PUBLIC iree_compiler_bindings_c_loader)
# target_include_directories(${_COMPILE_NAME}_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${_COMPILE_NAME}_lib PUBLIC 
    $<BUILD_INTERFACE:${_IREE_COMPILER_API}>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(${_COMPILE_NAME}_lib PUBLIC ${IREE_DEFAULT_COPTS})

add_executable(${_COMPILE_NAME} driver.c)
set_target_properties(${_COMPILE_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${RUNTIME_OUTPUT_PATH}"
)
install(TARGETS ${_COMPILE_NAME}
    RUNTIME DESTINATION bin
)
target_link_libraries(${_COMPILE_NAME} ${_COMPILE_NAME}_lib)