// RUN: comco-opt %s | comco-opt | mlir-opt --gpu-lower-to-nvvm-pipeline | mlir-cpu-runner

module {
  // CHECK-LABEL: func @addition
  func.func @addition() -> f32 {
    %r1 = arith.constant 3.5 : f32
    %r2 = arith.constant 7.8 : f32
    %r3 = arith.addf %r1, %r2 : f32
    return %r3 : f32
  }

  // CHECK-LABEL: func 2.00
  func.func @main() -> f32 {
    func.call @addition() : () -> f32
    %0 = arith.constant 2.0 : f32
    return %0: f32
  }
}