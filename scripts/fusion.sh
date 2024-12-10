#!/bin/bash

./build.sh && COMCO_FUSION=outline ./build/bin/comco-compiler test/Comco/matmul-allreduce.mlir 2| mlir-opt --convert-linalg-to-affine-loops --convert-gpu-to-nvvm
