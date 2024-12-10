#!/bin/bash

# Path to MLIR build
MLIR_DIR=/home/paperspace/llvm-project/build

if [ "$1" == "new" ]; then
    mkdir build
    if [ $? != 0 ]; then
        echo "please delete current build directory"
        exit 1
    fi

    cd build
    cmake -G Ninja .. \
        -DMLIR_DIR=$MLIR_DIR/lib/cmake/mlir \
        -DLLVM_EXTERNAL_LIT=$MLIR_DIR/bin/llvm-lit
    cd ..
fi


if [ "$1" == "all" ]; then
    cmake --build build --target check-comco-opt comco-compiler
else
    # just build the compiler
    cmake --build build --target comco-compiler
fi

if [ "$1" == "doc" ]; then
    cmake --build build --target mlir-doc
fi
