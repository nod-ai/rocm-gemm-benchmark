#include <iostream>

#include "IREEGemm/Codegen.hpp"

int main(int argc, char** argv) {
  if (argc < 6 || argc > 8) {
    std::cout
        << "Usage: " << argv[0]
        << " <M> <K> <N> [--transposea] [--transposeb] <dtype> <filePath>\n";
    return 1;
  }

  int argOffset = 1;
  bool transposeA = false;
  bool transposeB = false;

  int M = std::stoi(argv[argOffset++]);
  int K = std::stoi(argv[argOffset++]);
  int N = std::stoi(argv[argOffset++]);

  if (std::string(argv[argOffset]) == "--transposea") {
    transposeA = true;
    ++argOffset;
  }

  if (std::string(argv[argOffset]) == "--transposeb") {
    transposeB = true;
    ++argOffset;
  }

  std::string dtype = argv[argOffset++];
  std::string filePath = argv[argOffset++];

  ireeGemmMLIRGenerate(M, K, N, transposeA, transposeB, dtype, filePath);

  return 0;
}
