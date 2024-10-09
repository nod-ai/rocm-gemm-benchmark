#include "FrequencyMonitor.hpp"
#include "RotatingBuffer.hpp"
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
    , workspace(nullptr)
    , stream(nullptr)
{
}

HipBLASLtGEMMBench::~HipBLASLtGEMMBench()
{
    destroy();
}

void HipBLASLtGEMMBench::initialize()
{
    // CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    // CHECK_HIP_ERROR(hipMalloc(&workspace, 1e9));
}

void HipBLASLtGEMMBench::destroy()
{
    if(handle)
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
        handle = nullptr;
    }
    if(workspace)
    {
        CHECK_HIP_ERROR(hipFree(workspace));
        workspace = nullptr;
    }
}

void HipBLASLtGEMMBench::setDevice(int device_id)
{
    CHECK_HIP_ERROR(hipSetDevice(device_id));
}

Result HipBLASLtGEMMBench::run(Problem problem)
{
    size_t             m = problem.M, n = problem.N, k = problem.K;
    hipblasOperation_t transA = (problem.A == 'N') ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    hipblasOperation_t transB = (problem.B == 'N') ? HIPBLAS_OP_N : HIPBLAS_OP_T;

    size_t batch_count        = 1;
    size_t max_workspace_size = 1e9;
    void * a, *b, *c, *d, *alphaVec; // host
    void * d_a, *d_b, *d_c, *d_d, *d_alphaVec; // device
    void*  d_workspace;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipMalloc(&d_alphaVec, m * batch_count * sizeof(float)));

    CHECK_HIP_ERROR(hipHostMalloc(&a, m * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipHostMalloc(&b, n * k * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipHostMalloc(&c, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipHostMalloc(&d, m * n * batch_count * sizeof(hipblasLtHalf)));
    CHECK_HIP_ERROR(hipHostMalloc(&alphaVec, m * batch_count * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));

    memset(a, 0, m * k * batch_count * sizeof(hipblasLtHalf));
    memset(b, 0, n * k * batch_count * sizeof(hipblasLtHalf));
    memset(c, 0, m * n * batch_count * sizeof(hipblasLtHalf));
    for(int i = 0; i < m * batch_count; ++i)
        ((float*)alphaVec)[i] = static_cast<float>((rand() % 7) - 3);

    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_a, a, m * k * batch_count * sizeof(hipblasLtHalf), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_b, b, n * k * batch_count * sizeof(hipblasLtHalf), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_c, c, m * n * batch_count * sizeof(hipblasLtHalf), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_alphaVec, alphaVec, m * batch_count * sizeof(float), hipMemcpyHostToDevice, stream));

    float alpha = 1.0f, beta = 1.0f;
    int   num_cold_iterations = 50, num_warm_iterations = 100;

    auto timer = Timer();
    auto fm    = Frequency::getFrequencyMonitor();

    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    executeGemm(num_cold_iterations,
                nullptr,
                nullptr,
                handle,
                transA,
                transB,
                m,
                n,
                k,
                batch_count,
                alpha,
                beta,
                d_a,
                d_b,
                d_c,
                d_d,
                d_workspace,
                max_workspace_size,
                stream);

    executeGemm(num_warm_iterations,
                &timer,
                fm,
                handle,
                transA,
                transB,
                m,
                n,
                k,
                batch_count,
                alpha,
                beta,
                d_a,
                d_b,
                d_c,
                d_d,
                d_workspace,
                max_workspace_size,
                stream);

    // Clean up
    CHECK_HIP_ERROR(hipFree(d_workspace));
    CHECK_HIP_ERROR(hipFree(a));
    CHECK_HIP_ERROR(hipFree(b));
    CHECK_HIP_ERROR(hipFree(c));
    CHECK_HIP_ERROR(hipFree(d));
    CHECK_HIP_ERROR(hipFree(alphaVec));
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIP_ERROR(hipFree(d_alphaVec));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
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

void HipBLASLtGEMMBench::executeGemm(int                 num_iterations,
                                     Timer*              timer,
                                     Frequency::Monitor* monitor,
                                     hipblasLtHandle_t   handle,
                                     hipblasOperation_t  trans_a,
                                     hipblasOperation_t  trans_b,
                                     int64_t             m,
                                     int64_t             n,
                                     int64_t             k,
                                     int64_t             batch_count,
                                     float&              alpha,
                                     float&              beta,
                                     void*               d_a,
                                     void*               d_b,
                                     void*               d_c,
                                     void*               d_d,
                                     void*               d_workspace,
                                     int64_t             max_workspace_size,
                                     hipStream_t         stream)
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    int64_t shapeA[2] = {trans_a == HIPBLAS_OP_N ? m : k, trans_a == HIPBLAS_OP_N ? k : m};
    int64_t shapeB[2] = {trans_b == HIPBLAS_OP_N ? k : n, trans_a == HIPBLAS_OP_N ? n : k};
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&matA, HIP_R_8I, shapeA[0], shapeA[1], shapeA[0]));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&matB, HIP_R_8I, shapeB[0], shapeB[1], shapeB[0]));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_8I, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_8I, m, n, m));

    if(batch_count > 1)
    {
        int64_t stride_a = m * k;
        int64_t stride_b = k * n;
        int64_t stride_c = m * n;
        int64_t stride_d = m * n;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_d, sizeof(stride_d)));
    }

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // Set User Preference attributes
    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    const int                        request_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
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

    size_t workspace_size = 0;
    for(int i = 0; i < returnedAlgoCount; i++)
    {
        workspace_size = std::max(workspace_size, heuristicResult[i].workspaceSize);
    }

    if(timer)
        timer->tic();
    if(monitor)
        monitor->start();
    for(int i = 0; i < num_iterations; i++)
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                              matmul,
                                              &alpha,
                                              d_a,
                                              matA,
                                              d_b,
                                              matB,
                                              &beta,
                                              d_c,
                                              matC,
                                              d_d,
                                              matD,
                                              &heuristicResult[0].algo,
                                              d_workspace,
                                              workspace_size,
                                              stream));
    }
    hipStreamSynchronize(stream);
    if(timer)
        timer->toc();
    if(monitor)
        monitor->stop();

    return;
}
