#include <sys/stat.h>

#include <csignal>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "DataInitialization.hpp"
#include "FrequencyMonitor.hpp"
#include "Timer.hpp"
#include "gemm-bench.hpp"

#include "IREEGemm/Codegen.hpp"
#include "IREEGemm/Compile.h"
#include "IREEGemm/Runtime.hpp"

using namespace GEMMBench;

inline bool fileExists(const std::string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

std::string
    generateGemmString(int M, int K, int N, bool transposeA, bool transposeB, std::string dtype)
{
    std::string result = "gemm_" + std::to_string(M) + "_" + std::to_string(K) + "_"
                         + std::to_string(N) + "_" + dtype;

    if(transposeA)
        result += "_tA";
    if(transposeB)
        result += "_tB";

    return result;
}

void IREEGEMMBench::initialize()
{
    char** compileArgs = new char*[2];
    compileArgs[0]     = (char*)"--iree-hal-target-backends=rocm";
    compileArgs[1]     = (char*)"--iree-rocm-target-chip=gfx942";
    compile_state      = nullptr; // ireeGemmCompilerInitialize(2, compileArgs);
    device_id          = 7;

    storage_fp16 = new IREEGemmDeviceStorage("fp16");
    storage_bf16 = new IREEGemmDeviceStorage("bf16");
}

void IREEGEMMBench::linkData(GEMMData* data)
{
    storage_fp16->allocate(runtime_state->device, data->getCapacity(), data->getBufferA());
    storage_bf16->allocate(runtime_state->device, data->getCapacity(), data->getBufferB());
}

void IREEGEMMBench::setDevice(int device_id)
{
    this->device_id = device_id;
    runtime_state   = new IREEGemmRuntimeState("rocm://" + std::to_string(device_id));
    std::cout << "Running on " << device_id << std::endl;
}

Result IREEGEMMBench::run(Problem problem)
{
    IREEGemmRunner runner(
        runtime_state, problem.dtype == "fp16" ? storage_fp16 : storage_bf16, true);

    int         M          = problem.M;
    int         K          = problem.K;
    int         N          = problem.N;
    bool        transposeA = problem.A != 'N';
    bool        transposeB = problem.B != 'N';
    std::string dtype      = problem.dtype;

    if(transposeA && transposeB)
        transposeA = false;

    int num_cold_iterations = 50;
    int num_warm_iterations = 100;

    std::string gemmName = generateGemmString(M, K, N, transposeA, transposeB, dtype);
    std::string mlirPath = "kernels/mlir/" + gemmName + ".mlir";
    std::string vmfbPath = "kernels/vmfb/" + gemmName + ".vmfb";

    if(!fileExists(mlirPath))
    {
        ireeGemmMLIRGenerate(M, K, N, transposeA, transposeB, dtype, mlirPath);
    }

    if(!fileExists(vmfbPath))
    {
        ireeGemmCompilerCompile(compile_state, mlirPath.c_str(), vmfbPath.c_str());
        ireeGemmCompilerCleanup(compile_state);
    }

    runner.linkInput({(size_t)(transposeA ? K : M), (size_t)(transposeA ? M : K)}, dtype, nullptr);
    runner.linkInput({(size_t)(transposeB ? N : K), (size_t)(transposeB ? K : N)}, dtype, nullptr);
    runner.setupProblem(vmfbPath);

    auto timer = Timer();
    auto fm    = Frequency::getFrequencyMonitor();

    runner.preExecution(num_cold_iterations);
    runner.execute(num_cold_iterations);

    runner.preExecution(num_warm_iterations);
    fm->start();
    timer.tic();
    runner.execute(num_warm_iterations);
    timer.toc();
    fm->stop();

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

void IREEGEMMBench::destroy()
{
    std::cout << "Destroying bench" << std::endl;
    ireeGemmCompilerShutdown(compile_state);
    delete runtime_state;
    delete storage_fp16;
    delete storage_bf16;
}