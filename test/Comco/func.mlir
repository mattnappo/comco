module {
    comco.func @comco_func() {
        // %C = comco.call @arg_func(%A: tensor<2x2xf32>) : tensor<2x2xf32> // Calling is not necessary now
        comco.return
    }

    comco.func @arg_func(%A: tensor<2x2xf32>) -> tensor<2x2xf32> {
        comco.return %A : tensor<2x2xf32> 
    }
}
