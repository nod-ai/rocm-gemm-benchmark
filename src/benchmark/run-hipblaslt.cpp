#include "gemm-bench.hpp"
#include <cstring>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>

void HipBLASLtGEMMBench::allocateBuffers(
    float** dA, float** dB, float** dC, uint M, uint N, uint K, const char* dtype)
{
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    hipMalloc(dA, sizeA);
    hipMalloc(dB, sizeB);
    hipMalloc(dC, sizeC);
}

void HipBLASLtGEMMBench::deallocateBuffers(float* dA, float* dB, float* dC)
{
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
}

void HipBLASLtGEMMBench::initialize()
{
    hipblasLtCreate(&handle);
}

void HipBLASLtGEMMBench::setDevice(int device_id)
{
    this->device_id = device_id;
    hipSetDevice(device_id);
}

void HipBLASLtGEMMBench::destroy()
{
    hipblasLtDestroy(handle);
}

void HipBLASLtGEMMBench::run(Problem problem) override
{
    float *dA, *dB, *dC;

    // Allocate device buffers
    allocateBuffers(&dA, &dB, &dC, problem.M, problem.N, problem.K, problem.dtype);

    // Create matrix descriptors
    hipblasLtMatrixLayout_t descA, descB, descC;
    hipblasLtMatrixLayoutCreate(&descA, HIPBLASLT_R_32F, problem.M, problem.K, problem.M);
    hipblasLtMatrixLayoutCreate(&descB, HIPBLASLT_R_32F, problem.K, problem.N, problem.K);
    hipblasLtMatrixLayoutCreate(&descC, HIPBLASLT_R_32F, problem.M, problem.N, problem.M);

    // Create GEMM descriptors
    hipblasLtMatmulDesc_t gemmDesc;
    hipblasLtMatmulDescCreate(&gemmDesc, HIPBLASLT_COMPUTE_F32, HIPBLASLT_R_32F);

    // Transpose flags
    hipblasOperation_t opA = (problem.A == 'N') ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    hipblasOperation_t opB = (problem.B == 'N') ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    // Set problem attributes
    hipblasLtMatmulDescSetAttribute(gemmDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtMatmulDescSetAttribute(gemmDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    float alpha = 1.0f, beta = 0.0f;

    // Cold iterations
    for(int i = 0; i < 50; ++i)
    {
        hipblasLtMatmul(
            handle, gemmDesc, &alpha, dA, descA, dB, descB, &beta, dC, descC, dC, descC, nullptr);
    }

    // Warm iterations
    for(int i = 0; i < 100; ++i)
    {
        hipblasLtMatmul(
            handle, gemmDesc, &alpha, dA, descA, dB, descB, &beta, dC, descC, dC, descC, nullptr);
    }

    // Cleanup
    hipblasLtMatrixLayoutDestroy(descA);
    hipblasLtMatrixLayoutDestroy(descB);
    hipblasLtMatrixLayoutDestroy(descC);
    hipblasLtMatmulDescDestroy(gemmDesc);
    deallocateBuffers(dA, dB, dC);
}
