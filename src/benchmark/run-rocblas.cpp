#include <iostream>

#include <rocblas/rocblas.h>

#include "FrequencyMonitor.hpp"
#include "RotatingBuffer.hpp"
#include "Timer.hpp"
#include "gemm-bench.hpp"

#define CHECK_ROCBLAS_STATUS(status)                           \
    if(status != rocblas_status_success)                       \
    {                                                          \
        std::cout << "rocblas error: " << status << std::endl; \
        return Result{.ok = false};                            \
    }

using namespace GEMMBench;

void RocBLASGEMMBench::setDevice(int device_id)
{
    auto err = hipSetDevice(device_id);
    if(err != hipSuccess)
    {
        std::cout << "Unable to set HIP device." << std::endl;
        exit(EXIT_FAILURE);
    }
}

Result RocBLASGEMMBench::run(Problem problem)
{
    rocblas_status rstatus = rocblas_status_success;

    int              bpe;
    rocblas_datatype dtype;
    if(std::string(problem.dtype) == "fp16")
    {
        dtype = rocblas_datatype_f16_r;
        bpe   = 2;
    }
    else if(std::string(problem.dtype) == "bf16")
    {
        dtype = rocblas_datatype_bf16_r;
        bpe   = 2;
    }
    else if(std::string(problem.dtype) == "fp32")
    {
        dtype = rocblas_datatype_f32_r;
        bpe   = 4;
    }
    else
    {
        std::cout << "bad dtype" << std::endl;
        return {.ok = false};
    }

    rocblas_datatype compute_dtype = rocblas_datatype_f32_r;

    rocblas_int M = problem.M;
    rocblas_int N = problem.N;
    rocblas_int K = problem.K;

    float hAlpha = 1.0;
    float hBeta  = 0.0;

    rocblas_operation transA
        = problem.A == 'N' ? rocblas_operation_none : rocblas_operation_transpose;
    rocblas_operation transB
        = problem.B == 'N' ? rocblas_operation_none : rocblas_operation_transpose;

    rocblas_int lda, ldb, ldc, sizeA, sizeB, sizeC;

    if(transA == rocblas_operation_none)
    {
        lda   = M;
        sizeA = K * lda;
    }
    else
    {
        lda   = K;
        sizeA = M * lda;
    }
    if(transB == rocblas_operation_none)
    {
        ldb   = K;
        sizeB = N * ldb;
    }
    else
    {
        ldb   = N;
        sizeB = K * ldb;
    }
    ldc   = M;
    sizeC = N * ldc;

    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    // XXX
    auto  buffer = RotatingBuffer::Buffer::getInstance();
    void* dC;
    hipMalloc(&dC, sizeC * bpe);

    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);

    auto timer = Timer();
    auto fm    = Frequency::getFrequencyMonitor();

    int num_cold_iterations = 50;
    int num_warm_iterations = 100;

    for(int iteration = 0; iteration < num_cold_iterations; ++iteration)
    {
        void* dA = buffer->device<char>(sizeA * bpe);
        void* dB = buffer->device<char>(sizeB * bpe);

        int  solution_index = -1;
        auto algo           = rocblas_gemm_algo_standard;
        int  flags          = 0;

        rstatus = rocblas_gemm_ex(handle,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  &hAlpha,
                                  dA,
                                  dtype,
                                  lda,
                                  dB,
                                  dtype,
                                  ldb,
                                  &hBeta,
                                  dC,
                                  dtype,
                                  ldc,
                                  dC,
                                  dtype,
                                  ldc,
                                  compute_dtype,
                                  algo,
                                  solution_index,
                                  flags);

        CHECK_ROCBLAS_STATUS(rstatus);
    }
    hipDeviceSynchronize();

    fm->start();
    timer.tic();
    for(int iteration = 0; iteration < num_warm_iterations; ++iteration)
    {
        void* dA = buffer->device<char>(sizeA * bpe);
        void* dB = buffer->device<char>(sizeB * bpe);

        int  solution_index = -1;
        auto algo           = rocblas_gemm_algo_standard;
        int  flags          = 0;

        rstatus = rocblas_gemm_ex(handle,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  &hAlpha,
                                  dA,
                                  dtype,
                                  lda,
                                  dB,
                                  dtype,
                                  ldb,
                                  &hBeta,
                                  dC,
                                  dtype,
                                  ldc,
                                  dC,
                                  dtype,
                                  ldc,
                                  compute_dtype,
                                  algo,
                                  solution_index,
                                  flags);

        CHECK_ROCBLAS_STATUS(rstatus);
    }
    hipDeviceSynchronize();
    timer.toc();
    fm->stop();

    hipFree(dC);

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);

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
