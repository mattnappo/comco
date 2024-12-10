module {
    func.func @main() {
        %t = tensor.empty() : tensor<2x2xf32>
        comco.print(%t:tensor<2x2xf32>)
        gpu.printf "hi\n"
        return
    }
}
