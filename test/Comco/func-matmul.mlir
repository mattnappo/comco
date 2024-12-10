module {
    func.func @func() {
        %A = tensor.empty() : tensor<2x2xf32>
        %B = tensor.empty() : tensor<2x2xf32>
        %C = tensor.empty() : tensor<2x2xf32>

        %out = linalg.matmul
            ins(%A, %B: tensor<2x2xf32>, tensor<2x2xf32>)
            outs(%C: tensor<2x2xf32>) -> tensor<2x2xf32>

        func.return
    }
}
