module attributes {gpu.container_module} {
    comco.func @comco_func() {
        %A = tensor.empty() : tensor<2x2xf32>
        %B = tensor.empty() : tensor<2x2xf32>
        %C = tensor.empty() : tensor<2x2xf32>

        %0 = arith.constant 3.14 : f32

        %s = gpu.all_reduce add %0 uniform {} : (f32) -> (f32)

        // A[i, j] = %s

        %out = linalg.matmul
            ins(%A, %B: tensor<2x2xf32>, tensor<2x2xf32>)
            outs(%C: tensor<2x2xf32>) -> tensor<2x2xf32>

        comco.return
    }
}

