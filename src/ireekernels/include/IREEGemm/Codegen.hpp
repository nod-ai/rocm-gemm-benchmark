#ifndef IREE_GEMM_CODEGEN
#define IREE_GEMM_CODEGEN

#include <stdbool.h>

#include <string>

int ireeGemmMLIRGenerate(int M, int K, int N, bool transposeA, bool transposeB,
                     std::string dtype, std::string filePath);

#endif