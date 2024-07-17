set(_BENCH_NAME "iree_gemm_benchmark")

add_library(${_BENCH_NAME}_lib SHARED gemm_benchmark.c)
target_compile_options(${_BENCH_NAME}_lib PUBLIC ${IREE_DEFAULT_COPTS})
# target_link_options(${_BENCH_NAME}_lib PUBLIC -fsanitize=address -static-libasan)
target_link_libraries(${_BENCH_NAME}_lib PUBLIC iree_runtime_runtime)

add_executable(${_BENCH_NAME} driver.c gemm_benchmark.c)
target_compile_options(${_BENCH_NAME} PUBLIC ${IREE_DEFAULT_COPTS})
# target_compile_options(${_BENCH_NAME} PUBLIC ${IREE_DEFAULT_COPTS} -g -fsanitize=address)
# target_link_options(${_BENCH_NAME} PUBLIC -fsanitize=address)
set_target_properties(${_BENCH_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${RUNTIME_OUTPUT_PATH}"
)
install(TARGETS ${_BENCH_NAME}
    RUNTIME DESTINATION bin
)
target_link_libraries(${_BENCH_NAME} PUBLIC iree_runtime_runtime)
# target_link_libraries(${_BENCH_NAME} ${_BENCH_NAME}_lib)