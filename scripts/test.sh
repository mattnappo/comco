#!/bin/bash

cmake --build build --target comco-compiler
./build/bin/comco-compiler ./test/Comco/lower_math.mlir
