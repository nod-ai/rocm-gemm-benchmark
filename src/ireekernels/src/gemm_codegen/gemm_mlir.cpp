#include "IREEGemm/Codegen.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

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
  if (dtype == "bf16") return builder.getBF16Type();
  if (dtype == "fp8") return builder.getFloat8E4M3FNUZType();
  if (dtype == "fp16") return builder.getF16Type();
  if (dtype == "fp32") return builder.getF32Type();
  if (dtype == "fp64") return builder.getF64Type();
  if (dtype == "fp128") return builder.getF128Type();
  return builder.getF32Type();
}

int ireeGemmMLIRGenerate(int M, int K, int N, bool transposeA, bool transposeB,
                         std::string dtype, std::string filePath) {
  using namespace mlir;

  MLIRContext context;
  context.loadDialect<linalg::LinalgDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<tensor::TensorDialect>();

  auto loc = UnknownLoc::get(&context);
  ModuleOp module = ModuleOp::create(loc);

  OpBuilder builder(module.getBodyRegion());

  int64_t shapeA[2] = {transposeA ? K : M, transposeA ? M : K};
  int64_t shapeB[2] = {transposeB ? N : K, transposeB ? K : N};
  int64_t shapeC[2] = {M, N};

  auto inDty = getDtypeFromString(builder, dtype);
  auto outDty = inDty;

  auto typeA = RankedTensorType::get({shapeA[0], shapeA[1]}, inDty);
  auto typeB = RankedTensorType::get({shapeB[0], shapeB[1]}, inDty);
  auto typeC = RankedTensorType::get({shapeC[0], shapeC[1]}, outDty);

  auto funcTy = builder.getFunctionType({typeA, typeB}, {typeC});

  auto func = builder.create<func::FuncOp>(loc, "main_0", funcTy);
  Block& entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  auto cst =
      builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(inDty, 0.0));

  auto emptyTensor = builder.create<tensor::EmptyOp>(
      loc, llvm::ArrayRef<int64_t>{shapeC[0], shapeC[1]}, outDty);

  auto fillOp = builder.create<linalg::FillOp>(loc, ValueRange{cst},
                                               ValueRange{emptyTensor});

  Value matmulResult;
  if (transposeA && transposeB) {
    llvm::outs() << "matmul(A_transpose, B_transpose) not supported\n";
    return 2;
  } else if (transposeA) {
    auto matmulOp = builder.create<linalg::MatmulTransposeAOp>(
        loc, ValueRange{entryBlock.getArgument(0), entryBlock.getArgument(1)},
        ValueRange{fillOp.result()});
    matmulResult = matmulOp.getResult(0);
  } else if (transposeB) {
    auto matmulOp = builder.create<linalg::MatmulTransposeBOp>(
        loc, ValueRange{entryBlock.getArgument(0), entryBlock.getArgument(1)},
        ValueRange{fillOp.result()});
    matmulResult = matmulOp.getResult(0);
  } else {
    auto matmulOp = builder.create<linalg::MatmulOp>(
        loc, ValueRange{entryBlock.getArgument(0), entryBlock.getArgument(1)},
        ValueRange{fillOp.result()});
    matmulResult = matmulOp.getResult(0);
  }

  builder.create<func::ReturnOp>(loc, matmulResult);

  return printMLIR(module, filePath);
}
