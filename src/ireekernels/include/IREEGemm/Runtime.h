#ifndef IREE_GEMM_RUNTIME
#define IREE_GEMM_RUNTIME

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

typedef struct runtime_state_t runtime_state_t;
typedef struct runtime_storage_t runtime_storage_t;
typedef struct runtime_loop_data_t runtime_loop_data_t;

int ireeGemmRuntimeInitialize(const char* device_uri, bool rotate_buffer,
                              runtime_state_t** s);
int ireeGemmRuntimeSetupProblem(runtime_state_t* s, const char* module_path,
                                int M, int K, int N, bool transposeA,
                                bool transposeB, const char* dtype,
                                const float* A, const float* B,
                                size_t mem_capacity);
void ireeGemmRuntimeWaitForCompletion(runtime_loop_data_t* s);
void ireeGemmRuntimeExecute(runtime_state_t* s, int num_iterations);
int ireeGemmRuntimeDumpOutput(runtime_state_t* s);
void ireeGemmRuntimeCleanup(runtime_state_t* s);
void ireeGemmRuntimeShutdown(runtime_state_t* s);

#ifdef __cplusplus
}
#endif

#endif