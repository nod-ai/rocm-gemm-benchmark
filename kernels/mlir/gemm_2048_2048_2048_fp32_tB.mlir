module {
  func.func @main_0(%arg0: tensor<2048x2048xf32>, %arg1: tensor<2048x2048xf32>) -> tensor<2048x2048xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2048x2048xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %2 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<2048x2048xf32>, tensor<2048x2048xf32>) outs(%1 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    return %2 : tensor<2048x2048xf32>
  }
}
