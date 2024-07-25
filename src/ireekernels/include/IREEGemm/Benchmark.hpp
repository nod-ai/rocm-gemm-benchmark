#include <memory>
#include <string>

#include "IREEGemm/Codegen.hpp"
#include "IREEGemm/Compile.h"
#include "IREEGemm/Runtime.hpp"
#include "IREEGemm/Utils.h"

class IREEGemmBenchmarkPipeline {
 public:
  struct Problem {
    int M, K, N;
    bool transposeA, transposeB;
    std::string dtype;
  };

 private:
  compiler_state_t* compile_state;
  std::unique_ptr<IREEGemmRuntimeState> runtime_state;
  std::unique_ptr<IREEGemmDeviceStorage> storage_fp16;
  std::unique_ptr<IREEGemmDeviceStorage> storage_bf16;

  int currOp, totalOps;

  std::string generateGemmString(const Problem& p);

 public:
  IREEGemmBenchmarkPipeline();
  ~IREEGemmBenchmarkPipeline();
  double benchmarkProblem(const Problem& p);
  void initializeProgressBar(int totalOps) { this->totalOps = totalOps; }
};