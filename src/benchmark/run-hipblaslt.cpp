#include "FrequencyMonitor.hpp"
#include "Timer.hpp"
#include "gemm-bench.hpp"

#include <algorithm>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>

#define CHECK_HIPBLASLT_ERROR(expr)                                                     \
    do                                                                                  \
    {                                                                                   \
        hipblasStatus_t status = (expr);                                                \
        if(status != HIPBLAS_STATUS_SUCCESS)                                            \
        {                                                                               \
            std::cerr << "hipBLASLt error: " << hipblasStatusToString(status) << " at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;                      \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while(0)

#define CHECK_HIP_ERROR(expr)                                                                    \
    do                                                                                           \
    {                                                                                            \
        hipError_t status = (expr);                                                              \
        if(status != hipSuccess)                                                                 \
        {                                                                                        \
            std::cerr << "HIP error: " << hipGetErrorString(status) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                                  \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while(0)

using namespace GEMMBench;

HipBLASLtGEMMBench::HipBLASLtGEMMBench()
    : handle(nullptr)
{
}

HipBLASLtGEMMBench::~HipBLASLtGEMMBench()
{
    destroy();
}

void HipBLASLtGEMMBench::initialize()
{
    std::cout << "Initializing hipblaslt" << std::endl;
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
}

void HipBLASLtGEMMBench::destroy()
{
    std::cout << "Destroying hipblaslt" << std::endl;
    if(handle)
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
        handle = nullptr;
    }
}

void HipBLASLtGEMMBench::setDevice(int device_id)
{
    std::cout << "Setting hipblaslt device to " << device_id << std::endl;
    CHECK_HIP_ERROR(hipSetDevice(device_id));
}

Result HipBLASLtGEMMBench::run(Problem problem)
{
    std::cout << "Running hipblaslt with handle " << &handle << std::endl;
    int64_t            m = problem.M, n = problem.N, k = problem.K;
    hipblasOperation_t transA = (problem.A == 'N') ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    hipblasOperation_t transB = (problem.B == 'N') ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    hipblasDatatype_t computeType;
    if(strcmp(problem.dtype, "fp16") == 0)
    {
        computeType = HIPBLAS_R_16F;
    }
    else if(strcmp(problem.dtype, "bf16") == 0)
    {
        computeType = HIPBLAS_R_16B;
    }
    else
    {
        std::cerr << "Unsupported data type: " << problem.dtype << std::endl;
        return {.ok = false};
    }

    void * d_A, *d_B, *d_C, *d_D;
    size_t sizeA = m * k * sizeof(hipblasLtHalf);
    size_t sizeB = k * n * sizeof(hipblasLtHalf);
    size_t sizeC = m * n * sizeof(hipblasLtHalf);
    CHECK_HIP_ERROR(hipMalloc(&d_A, sizeA));
    CHECK_HIP_ERROR(hipMalloc(&d_B, sizeB));
    CHECK_HIP_ERROR(hipMalloc(&d_C, sizeC));
    CHECK_HIP_ERROR(hipMalloc(&d_D, sizeC));

    // Allocate workspace
    void*  workspace      = nullptr;
    size_t workspace_size = 32 * 1024 * 1024; // 32 MB, adjust as needed
    CHECK_HIP_ERROR(hipMalloc(&workspace, workspace_size));

    // Set up alpha and beta
    float alpha = 1.0f, beta = 0.0f;

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    int num_cold_iterations = 50;
    int num_warm_iterations = 100;

    for(int i = 0; i < num_cold_iterations; ++i)
    {
        executeGEMM(transA,
                    transB,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    d_A,
                    d_B,
                    d_C,
                    d_D,
                    workspace,
                    workspace_size,
                    stream);
    }

    auto timer = Timer();
    auto fm    = Frequency::getFrequencyMonitor();

    fm->start();
    timer.tic();
    for(int i = 0; i < num_warm_iterations; ++i)
    {
        executeGEMM(transA,
                    transB,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    d_A,
                    d_B,
                    d_C,
                    d_D,
                    workspace,
                    workspace_size,
                    stream);
    }
    timer.toc();
    fm->stop();

    // Clean up
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    CHECK_HIP_ERROR(hipFree(d_D));
    CHECK_HIP_ERROR(hipFree(workspace));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    auto [sclk, mclk, fclk] = fm->statistics();

    return Result{true,
                  num_warm_iterations,
                  double(timer.nanoseconds()) / num_warm_iterations / 1.0e3,
                  sclk.min,
                  sclk.mean,
                  sclk.max,
                  mclk.min,
                  mclk.mean,
                  mclk.max,
                  fclk.min,
                  fclk.mean,
                  fclk.max,
                  -1,
                  fm->systemFrequency(),
                  fm->memoryFrequency(),
                  fm->dataFrequency(),
                  fm->temperature(),
                  fm->power()};
}

void HipBLASLtGEMMBench::executeGEMM(hipblasOperation_t transA,
                                     hipblasOperation_t transB,
                                     int64_t            m,
                                     int64_t            n,
                                     int64_t            k,
                                     const float&       alpha,
                                     const float&       beta,
                                     void*              d_A,
                                     void*              d_B,
                                     void*              d_C,
                                     void*              d_D,
                                     void*              workspace,
                                     size_t             workspace_size,
                                     hipStream_t        stream)
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    const int                        request_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 1;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    if(returnedAlgoCount == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                          matmul,
                                          &alpha,
                                          d_A,
                                          matA,
                                          d_B,
                                          matB,
                                          &beta,
                                          d_C,
                                          matC,
                                          d_D,
                                          matD,
                                          &heuristicResult[0].algo,
                                          workspace,
                                          workspace_size,
                                          stream));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
}