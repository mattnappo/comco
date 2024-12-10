#!/bin/bash

set -u
set -x
set -e

# setup
mlir_runner_utils="/home/paperspace/llvm-project/build/lib/libmlir_cuda_runtime.so.20.0git"
mlir_c_runner_utils="/home/paperspace/llvm-project/build/lib/libmlir_c_runner_utils.so.20.0git"
mlir_cuda_runtime="/usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudart.so.11.7.99"

# perform GPU lowerings
mlir-opt "$1" -lower-affine -gpu-lower-to-nvvm-pipeline="cubin-format=fatbin" \
  -o out.mlir

# not required for anything, just for debugging
# mlir-translate --mlir-to-llvmir out.mlir -o out.ll

# run the MLIR program
mlir-cpu-runner out.mlir \
  --shared-libs=$mlir_cuda_runtime \
  --shared-libs=$mlir_runner_utils \
  --shared-libs=$mlir_c_runner_utils \
  --entry-point-result=void
