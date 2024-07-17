#ifndef IREE_GEMM_COMPILE
#define IREE_GEMM_COMPILE

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct compiler_state_t compiler_state_t;

compiler_state_t* ireeGemmCompilerInitialize(int argc, char** argv);

int ireeGemmCompilerCompile(compiler_state_t* s, const char* inputFilePath,
                            const char* outputFilePath);

void ireeGemmCompilerCleanup(compiler_state_t* s);

void ireeGemmCompilerShutdown(compiler_state_t* s);

#ifdef __cplusplus
}
#endif

#endif