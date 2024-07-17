module {
  func.func @main_0(%arg0: tensor<16x65536xf16>, %arg1: tensor<65536x16xf16>) -> tensor<16x16xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<16x16xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<16x65536xf16>, tensor<65536x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
    return %2 : tensor<16x16xf16>
  }
}
