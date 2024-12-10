// RUN: comco-opt %s | comco-opt | FileCheck %s

module {
    // A smoke "test"
    // CHECK-LABEL: func @smoke()
    func.func @smoke() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = comco.foo %{{.*}} : i32
        %res = comco.foo %0 : i32
        %r2 = comco.foo %res : i32
        return
    }

    // Check installation of tensor dialect
    func.func @with_tensors() {
        %0 = tensor.empty() : tensor<4x4xf32>

        %sum = math.sqrt %0 : tensor<4x4xf32>
        return
    }

    func.func @create_empty(%d1: index, %d2: index) -> tensor<?x?xf32> {
      %t = tensor.empty(%d1, %d2) : tensor<?x?xf32>
      comco.print(%t : tensor<?x?xf32>)
      return %t : tensor<?x?xf32>
    }

    func.func @addition() {
      %r1 = arith.constant 3.5 : f32
      %r2 = arith.constant 7.8 : f32
      %r3 = arith.addf %r1, %r2 : f32
      return
    }

    func.func @norm() {
      %in = tensor.empty() : tensor<4x32x8xf32>
      %d = comco.norm(%in : tensor<4x32x8xf32>) : f32

      return
    }

    func.func @example() {
      // Init tensor
      %A = tensor.empty() : tensor<8x8xf32>
      %f = arith.constant 5.5 : f32
      %out = linalg.fill ins(%f : f32) outs(%A : tensor<8x8xf32>) -> tensor<8x8xf32>

      // AllGather (on 4 ranks, since 8*4 = 32)
      %B = comco.all_gather(%out : tensor<8x8xf32>) : tensor<32x32xf32>

      // Additive AllReduce
      %op = arith.constant 0 : index
      %C = comco.all_reduce(%op, %out : tensor<8x8xf32>) : tensor<32x32xf32>

      %D = comco.relu(%B : tensor<32x32xf32>) : tensor<32x32xf32>

      %E = comco.scalar_mul(%B : tensor<32x32xf32>, %f) : tensor<32x32xf32>

      return
    }
}
