module {
  func.func @main_0(%arg0: tensor<16x1024xf16>, %arg1: tensor<1024x16xf16>) -> tensor<16x16xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<16x16xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<16x1024xf16>, tensor<1024x16xf16>) outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
    return %2 : tensor<16x16xf16>
  }
}
