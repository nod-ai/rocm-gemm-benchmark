#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "IREEGemm/Runtime.h"

int main(int argc, char** argv) {
  if (argc != 10) {
    fprintf(stderr,
            "Usage: %s <flatbuffer_path> <iterations> <M> <K> <N> <transposeA> "
            "<transposeB> <dtype> <device>\n",
            argv[0]);
    return 1;
  }

  const char* flatbuffer_path = argv[1];
  int num_iterations = atoi(argv[2]);
  int M = atoi(argv[3]);
  int K = atoi(argv[4]);
  int N = atoi(argv[5]);
  bool transposeA = atoi(argv[6]);
  bool transposeB = atoi(argv[7]);
  const char* dtype = argv[8];
  const char* device = argv[9];

  runtime_state_t* s = NULL;
  int status = ireeGemmRuntimeInitialize(device, false, &s);

  // Perform GEMM operation
  if (status == 0) {
    ireeGemmRuntimeSetupProblem(s, flatbuffer_path, M, K, N, transposeA,
                                transposeB, dtype, NULL, NULL, 0);
    // printf("Executing cold\n");
    // ireeGemmRuntimeExecute(s, num_iterations);
    printf("Executing warm\n");
    ireeGemmRuntimeExecute(s, num_iterations);
  }

  ireeGemmRuntimeCleanup(s);
}