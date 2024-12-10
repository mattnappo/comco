module attributes {gpu.container_module} {
    comco.func @comco_func() {
        %A = tensor.empty() : tensor<32x32xf32>
        %B = tensor.empty() : tensor<32x32xf32>
        %C = tensor.empty() : tensor<32x32xf32>

        %0 = arith.constant 3.14 : f32

        %out = linalg.matmul
            ins(%A, %B: tensor<32x32xf32>, tensor<32x32xf32>)
            outs(%C: tensor<32x32xf32>) -> tensor<32x32xf32>

        %s = gpu.all_reduce add %0 uniform {} : (f32) -> (f32)

        comco.return
    }
}
