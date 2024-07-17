#include "IREEGemm/Codegen.hpp"
#include "mlir/IR/Builders.h"

int printMLIR(mlir::ModuleOp& finalOp, std::string filePath) {
  std::error_code errorCode;
  llvm::raw_fd_ostream outputFile(filePath, errorCode);
  if (errorCode) {
    llvm::errs() << "Could not open file: " << errorCode.message() << "\n";
    return 1;
  }
  finalOp->print(outputFile);
  outputFile.close();
  return 0;
}

mlir::Type getDtypeFromString(mlir::OpBuilder& builder, std::string dtype) {
  if (dtype == "fp8") return builder.getFloat8E4M3FNUZType();
  if (dtype == "bf16") return builder.getBF16Type();
  if (dtype == "fp16") return builder.getF16Type();
  if (dtype == "fp32") return builder.getF32Type();
  if (dtype == "fp32") return builder.getF64Type();
  if (dtype == "fp128") return builder.getF128Type();
  return builder.getF32Type();
}