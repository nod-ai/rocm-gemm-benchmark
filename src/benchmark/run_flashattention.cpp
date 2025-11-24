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
    generateFAString(size_t B, size_t H, size_t S_Q, size_t S_KV, size_t DH, std::string dtype)
{
    std::string iree_dtype = dtype == "fp16" ? "f16" : "f8";
    std::string result     = "attention_B" + std::to_string(B) + "_H" + std::to_string(H) + "_SQ"
                         + std::to_string(S_Q) + "_SKV" + std::to_string(S_KV) + "_DH"
                         + std::to_string(DH) + "_" + iree_dtype;

    return result;
}

void AMDSHARKFABench::initialize()
{
    device_id    = 7;
    storage_fp16 = new IREEGemmDeviceStorage("fp16");
    storage_fp32 = new IREEGemmDeviceStorage("fp32");
}

void AMDSHARKFABench::linkData(GEMMData* data)
{
    storage_fp16->allocate(runtime_state->device, data->getCapacity(), data->getBufferA());
    storage_fp32->allocate(runtime_state->device, data->getCapacity(), data->getBufferB());
}

void AMDSHARKFABench::setDevice(int device_id)
{
    this->device_id = device_id;
    runtime_state   = new IREEGemmRuntimeState("rocm://" + std::to_string(device_id));
}

Result AMDSHARKFABench::run(Problem problem)
{
    size_t      S_Q   = (size_t)problem.M;
    size_t      S_KV  = (size_t)problem.N;
    size_t      DH    = (size_t)problem.K;
    size_t      B     = (size_t)problem.A;
    size_t      H     = (size_t)problem.B;
    std::string dtype = problem.dtype;

    IREEGemmRunner runner(runtime_state, dtype == "fp16" ? storage_fp16 : storage_fp32, false);

    int num_cold_iterations = 50;
    int num_warm_iterations = 100;

    std::string faName   = generateFAString(B, H, S_Q, S_KV, DH, dtype);
    std::string mlirPath = "attention/mlir/" + faName + ".mlir";
    std::string vmfbPath = "attention/vmfb/" + faName + ".vmfb";

    if(!fileExists(vmfbPath))
    {
        return {.ok = false};
    }

    runner.linkInput({B, H, S_Q, DH}, dtype, nullptr); // query
    runner.linkInput({B, H, S_KV, DH}, dtype, nullptr); // key
    runner.linkInput({B, H, S_KV, DH}, dtype, nullptr); // value
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

void AMDSHARKFABench::destroy()
{
    std::cout << "Destroying bench" << std::endl;
    delete runtime_state;
    delete storage_fp16;
}
