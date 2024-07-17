#include "IREEGemm/Compile.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("Usage: %s <inputFilePath> <outputFilePath> <--...iree args>\n",
           argv[0]);
    return 1;
  }

  const char* inputFilePath = argv[1];
  const char* outputFilePath = argv[2];
  int ireeArgsIdx = 3;

  compiler_state_t* s =
      ireeGemmCompilerInitialize(argc - ireeArgsIdx, &argv[ireeArgsIdx]);

  ireeGemmCompilerCompile(s, inputFilePath, outputFilePath);

  ireeGemmCompilerCleanup(s);

  return 0;
}
