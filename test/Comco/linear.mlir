// RUN: comco-opt %s | comco-opt | FileCheck %s

module attributes {gpu.container_module} {

  // Kernels (gpu.func) and device-side library functions (func.func)
  gpu.module @kernels {
    gpu.func @kernel() kernel {
      // Define a variable
      %3 = arith.constant 3.14 : f32

      // Print the value of %3
      gpu.printf "Value: %f\n" %3 : f32

      // some linear algebruh operations here
      // or better yet, make a new kernel for that

      gpu.return
    }

    gpu.func @math_kernel() kernel {
      %1 = arith.constant 1.1 : f32
      %2 = arith.constant 2.2 : f32
      %3 = arith.constant 3.3 : f32
      %4 = arith.constant 4.4 : f32

      %A = tensor.from_elements %1, %2, %3, %4 : tensor<2x2xf32>
      %B = tensor.from_elements %4, %3, %2, %1 : tensor<2x2xf32>

      %C = tensor.empty() : tensor<2x2xf32>

      %r0 = linalg.matmul
        ins(%A, %B : tensor<2x2xf32>, tensor<2x2xf32>)
        outs(%C : tensor<2x2xf32>) -> tensor<2x2xf32>

      gpu.printf "done\n"
      gpu.return
    }
  }

  // Host functions (func.func)
  // CHECK-LABEL: func @addition()
  func.func @addition() -> f32 {
    %r1 = arith.constant 3.5 : f32
    %r2 = arith.constant 7.8 : f32
    %r3 = arith.addf %r1, %r2 : f32
    return %r3 : f32
  }

  func.func @main() -> f32 {
    func.call @addition() : () -> f32
    %3 = arith.constant 3.0 : f32

    %cst = arith.constant 1 : index

    gpu.launch_func @kernels::@kernel
      blocks in (%cst, %cst, %cst)
      threads in (%cst, %cst, %cst)
      args()

    return %3: f32
  }
}
