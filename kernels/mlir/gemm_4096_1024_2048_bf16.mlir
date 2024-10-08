module {
  func.func @main_0(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<1024x2048xbf16>) -> tensor<4096x2048xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<4096x2048xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<4096x2048xbf16>) -> tensor<4096x2048xbf16>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4096x1024xbf16>, tensor<1024x2048xbf16>) outs(%1 : tensor<4096x2048xbf16>) -> tensor<4096x2048xbf16>
    return %2 : tensor<4096x2048xbf16>
  }
}
