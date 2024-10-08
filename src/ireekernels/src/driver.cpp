#include <sys/stat.h>

#include <csignal>
#include <fstream>
#include <iostream>
#include <vector>

#include "IREEGemm/Benchmark.hpp"

#define RETURN_ON_ERROR(func_call, error_msg)       \
  {                                                 \
    int status = (func_call);                       \
    if (status != 0) {                              \
      fprintf(stderr, " - Error: %s\n", error_msg); \
      return (double)-1;                            \
    }                                               \
  }

inline bool fileExists(const std::string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

std::string IREEGemmBenchmarkPipeline::generateGemmString(const Problem& p) {
  std::string result = "gemm_" + std::to_string(p.M) + "_" +
                       std::to_string(p.K) + "_" + std::to_string(p.N) + "_" +
                       p.dtype;

  if (p.transposeA) result += "_tA";
  if (p.transposeB) result += "_tB";

  return result;
}

IREEGemmBenchmarkPipeline::IREEGemmBenchmarkPipeline() {
  currOp = 0;
  totalOps = 0;

  char** compileArgs = new char*[2];
  compileArgs[0] = (char*)"--iree-hal-target-backends=rocm";
  compileArgs[1] = (char*)"--iree-rocm-target-chip=gfx942";
  compile_state = ireeGemmCompilerInitialize(2, compileArgs);

  runtime_state = std::make_unique<IREEGemmRuntimeState>("rocm://7");
  storage_fp16 = std::make_unique<IREEGemmDeviceStorage>("fp16");
  storage_bf16 = std::make_unique<IREEGemmDeviceStorage>("bf16");

  std::vector<float> input_buff(1e9);
  std::fill(input_buff.begin(), input_buff.end(), 0);

  storage_fp16->allocate(runtime_state->device, 1e9, input_buff.data());
  storage_bf16->allocate(runtime_state->device, 1e9, input_buff.data());
}

IREEGemmBenchmarkPipeline::~IREEGemmBenchmarkPipeline() {
  ireeGemmCompilerShutdown(compile_state);
}

double IREEGemmBenchmarkPipeline::benchmarkProblem(const Problem& p) {
  bool showProgress = totalOps > 0 && ++currOp <= totalOps;

  std::string gemmName = generateGemmString(p);
  std::string mlirPath = "kernels/mlir/" + gemmName + ".mlir";
  std::string vmfbPath = "kernels/vmfb/" + gemmName + ".vmfb";

  IREEGemmRunner runner(
      runtime_state.get(),
      p.dtype == "fp16" ? storage_fp16.get() : storage_bf16.get(), false);

  if (!fileExists(mlirPath)) {
    if (showProgress)
      print_progress(currOp, totalOps,
                     ("Generating " + gemmName + ".mlir").c_str());
    RETURN_ON_ERROR(ireeGemmMLIRGenerate(p.M, p.K, p.N, p.transposeA,
                                         p.transposeB, p.dtype, mlirPath),
                    "Failed to generate MLIR");
  }

  if (!fileExists(vmfbPath)) {
    if (showProgress)
      print_progress(currOp, totalOps,
                     ("Compiling to " + gemmName + ".vmfb").c_str());
    RETURN_ON_ERROR(ireeGemmCompilerCompile(compile_state, mlirPath.c_str(),
                                            vmfbPath.c_str()),
                    "Failed to compile");
    ireeGemmCompilerCleanup(compile_state);
  }

  int numIterations = 50;

  // const float* A = new float[(size_t)1e8];
  // const float* B = new float[(size_t)1e8];

  if (showProgress)
    print_progress(currOp, totalOps, ("Setting up " + gemmName).c_str());
  double executionTime = 0.0;

  runner.linkInput(
      {(size_t)(p.transposeA ? p.K : p.M), (size_t)(p.transposeA ? p.M : p.K)},
      p.dtype, nullptr);
  runner.linkInput(
      {(size_t)(p.transposeB ? p.N : p.K), (size_t)(p.transposeB ? p.K : p.N)},
      p.dtype, nullptr);
  runner.setupProblem(vmfbPath);

  if (showProgress)
    print_progress(currOp, totalOps, ("Executing " + gemmName).c_str());
  runner.preExecution(numIterations);
  runner.execute(numIterations);

  // delete[] A;
  // delete[] B;

  return executionTime;
}

int main(void) {
  using Problem = IREEGemmBenchmarkPipeline::Problem;

  IREEGemmBenchmarkPipeline gemmBench;
  std::vector<Problem> problems;

  std::cout << "Generating problems" << std::endl;
  for (int m = 1024; m <= 4096; m *= 2) {
    for (int k = 1024; k <= 4096; k *= 2) {
      for (int n = 1024; n <= 4096; n *= 2) {
        for (bool transposeA : {true, false}) {
          for (bool transposeB : {true, false}) {
            if (!(transposeA && transposeB)) {
              for (const std::string& dtype : {"fp16", "bf16"}) {
                problems.push_back(
                    Problem{m, k, n, transposeA, transposeB, dtype});
              }
            }
          }
        }
      }
    }
  }

  std::cout << "Successfully generated " << problems.size() << " GEMM problems "
            << std::endl
            << "Running Benchmarks" << std::endl;

  std::ofstream outFile("benchmark_results.csv");
  if (!outFile.is_open()) {
    std::cerr << "Failed to open the file." << std::endl;
    return 1;
  }

  outFile << "M,K,N,Transpose A,Transpose B,Data Type,Execution Time (ms)\n";

  gemmBench.initializeProgressBar(problems.size());

  for (const Problem& p : problems) {
    outFile << p.M << "," << p.K << "," << p.N << "," << p.transposeA << ","
            << p.transposeB << "," << p.dtype << ",";
    double executionTime = gemmBench.benchmarkProblem(p) * 1000;
    if (executionTime < 0)
      outFile << "ERROR\n";
    else
      outFile << executionTime << "\n";
  }

  outFile.close();
  std::cout << "Benchmark results saved to benchmark_results.csv" << std::endl;

  return 0;
}