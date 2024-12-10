// RUN: comco-opt %s | FileCheck %s
module {

  mesh.mesh @mesh0(shape = 4)

  // CHECK: func @allreduce_func
  func.func @allreduce_func(%A: tensor<32x32xf32>) {
    // Note that its fine to use the mesh.all_reduce op here.
    // Ideally, I'd write a frontend where the comco all reduce op just
    // directly gets changed into the mesh all reduce op, but that's kind of extra
    // unnecessary work. Expect input dialect to be mesh.all_reduce
    // %sum = gpu.all_reduce add %arg0 uniform {} : (tensor<32x32xf32>) -> (tensor<32x32xf32>) // must be f32 -> f32 (scalar only)



    mesh.all_reduce %A on @mesh0 reduction = max
      : tensor<32x32xf32> -> tensor<32x32xf64>

    func.return
  }

}


// // GOAL: this should be lowered into something along the lines of:
// module attributes {gpu.container_module} {
//   gpu.module @kernels {
//     // Not sure about the exact input type here
//     // But I think its fine to work at the tensor level then I can
//     // one-shot bufferize
//     // But will need to keep track of in/out GPU copying
//     gpu.func @all_reduce0(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
//       // The inside of this kernel contains all the code necessary to do an allreduce.
//       // Besides the tensor copies. Where does that copying code go (CUDA question honestly)
//       // We want to copy out of dev into host
//       // Also note that in theory in the future, this could be replaced with a call to the wrapped nccl dialect
//       // (similar to how the matmul inner kernel code could be a gpu.matmul or a cublas.matmul wrapper)
//       %sum = gpu.all_reduce add %arg0 uniform {} : (f32) -> (f32)
//       // copy out ??
//       return %sum : tensor<32x32xf32>
//     }
//     // Note how the above kernel is created by this transformation with the appropriate code inside.
//     // Also note that there is a pipeline for lowering gpu.all_reduce into simple operations
//     // https://mlir.llvm.org/doxygen/AllReduceLowering_8cpp_source.html
//   }

//   // This CPU FUNCTION is changed to replace the old all_reduce op with a
//   // call to the newly-generated kernel
//   func.func @allreduce_func(%arg0: tensor<32x32xf32>) {
//     gpu.launch_func @kernels::@all_reduce0
//       args(%arg0)


//     %0 = index.constant 1
//     gpu.launch_func @kernels::@all_reduce0
//       blocks in (%0, %0, %0)
//       threads in (%0, %0, %0) : i64 args(%arg0 : f32)

//     func.return
//   }

// }

// // THERE EXISTS A EXACT MIRROR TEST WITH MATMUL INSTEAD OF ALLREDUCE.
// // ...and other operations...
// // (but matmul+allreduce is the simplest)
// // ((I will make that one now))