//===- ComcoDialect.td - Comco dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMCO_DIALECT
#define COMCO_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Tensor/IR/TensorBase.td"

//===----------------------------------------------------------------------===//
// Comco dialect definition.
//===----------------------------------------------------------------------===//

def Comco_Dialect : Dialect {
    let name = "comco";
    let summary = "The Comco MLIR dialect.";
    let description = [{
        Comco is an ML compiler based on CocoNet that allows for analysis of
        fusion and communication overlapping.
    }];
    let cppNamespace = "::mlir::comco";
}

//===----------------------------------------------------------------------===//
// Base comco operation definition.
//===----------------------------------------------------------------------===//

class Comco_Op<string mnemonic, list<Trait> traits = []> :
        Op<Comco_Dialect, mnemonic, traits>;

#endif // COMCO_DIALECT
