module attributes {gpu.container_module} {
//module {
    comco.func @comco_func() {
        %A = tensor.empty() : tensor<2x2xf32>
        %B = tensor.empty() : tensor<2x2xf32>
        %C = tensor.empty() : tensor<2x2xf32>

        // comco.thread_id // TODO
        // %0 = A.load ...

        %0 = arith.constant 3.14 : f32

        %s = gpu.all_reduce add %0 uniform {} : (f32) -> (f32)

        comco.return
    }
}

