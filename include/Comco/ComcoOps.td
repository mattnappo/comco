//===- ComcoOps.td - Comco dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMCO_OPS
#define COMCO_OPS

include "ComcoDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

def Comco_FooOp : Comco_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `comco.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = comco.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def Comco_NormOp : Comco_Op<"norm", [Pure]> {
  let summary = "Compute the L2 norm";
  let description = [{
      The `comco.norm` operation computes the L2 norm of the given tensor.

      Example:

      ```mlir
      %t = tensor.empty() : tensor<4xf32>
      %n = comco.norm(%0 : tensor<4xf32>) : f32
      ```
  }];

  let arguments = (ins F32Tensor:$input);
  let results = (outs F32);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `:` type(results)
  }];
}

// TODO: ADD A CUSTOM ATTRIBUTE WHICH IS USED FOR ADDITION
def Comco_AllReduce : Comco_Op<"all_reduce", [Pure]> {
  let summary = "AllReduce operation";
  let description = [{
      The `comco.all_reduce` operation performs AllReduce on the given tensor.

      Unti attributes are stabilized, supply an integer to choose reduction
      operator.

      operator | code
      ---------------
      add      | 0
      subtract | 1
      multiply | 2
      divide   | 3

      Example:

      ```mlir
      %t = tensor.empty() : tensor<4x4xf32>
      %op = arith.const 0 : index // Addition
      %g = comco.all_reduce(%op, %0 : tensor<4x4xf32>) : tensor<16x16xf32>
      ```
  }];

  let arguments = (ins Index:$op, F32Tensor:$input);
  let results = (outs F32Tensor);

  let assemblyFormat = [{
    `(` $op `,` $input `:` type($input) `)` attr-dict `:` type(results)
  }];
}

def Comco_AllGather : Comco_Op<"all_gather", [Pure]> {
  let summary = "AllGather operation";
  let description = [{
      The `comco.all_gather` operation performs AllGather on the given tensor.

      Example:

      ```mlir
      %t = tensor.empty() : tensor<4x4xf32>
      %g = comco.all_gather(%0 : tensor<4x4xf32>) : tensor<16x16xf32>
      ```
  }];

  let arguments = (ins F32Tensor:$input);
  let results = (outs F32Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `:` type(results)
  }];
}

def Comco_ReLU : Comco_Op<"relu", [Pure]> {
  let summary = "ReLU activation";
  let description = [{
      The `comco.relu` operation computes ReLU on the given tensor.
      The output is a tensor where every element is max(0, x).

      Example:

      ```mlir
      %t = tensor.empty() : tensor<4x4xf32>
      %g = comco.relu(%0 : tensor<4x4xf32>) : tensor<4x4xf32>
      ```
  }];

  let arguments = (ins F32Tensor:$input);
  let results = (outs F32Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `:` type(results)
  }];
}

def Comco_Update : Comco_Op<"update", [Pure]> {
  let summary = "Update a tensor";
  let description = [{
      The `comco.update` operation on tensor A with scalar f
      returns a new tensor containing a_i + f.

      Example:

      ```mlir
      %A = tensor.empty() : tensor<4x4xf32>
      %f = arith.const 1.5 : f32
      %B = comco.update(%A, %0) : tensor<4x4xf32>
      ```
  }];

  let arguments = (ins F32Tensor:$A, F32:$f);
  let results = (outs F32Tensor);

  let assemblyFormat = [{
    `(` $A `:` type($A) `,` $f `)` attr-dict `:` type(results)
  }];
}

def Comco_UpdateTensor : Comco_Op<"updatet", [Pure]> {
  let summary = "Update a tensor with a tensor";
  let description = [{
      The `comco.update` operation on tensor A with tensor B
      returns a new tensor containing a_i + b_i.
  }];

  let arguments = (ins F32Tensor:$A, F32Tensor:$B);
  let results = (outs F32Tensor);

  let assemblyFormat = [{
    `(` $A `:` type($A) `,` $B `:` type($B) `)` attr-dict `:` type(results)
  }];
}

def Comco_ScalarMul : Comco_Op<"scalar_mul", [Pure]> {
  let summary = "Multiply a scalar with a tensor.";
  let description = [{
      The `comco.scalar_mul` operation on tensor A with scalar f
      returns A * f.

      Example:

      ```mlir
      %A = tensor.empty() : tensor<4x4xf32>
      %f = arith.const 1.5 : f32
      %B = comco.scalar_mul(%A, %0) : tensor<4x4xf32>
      ```
  }];

  let arguments = (ins F32Tensor:$A, F32:$f);
  let results = (outs F32Tensor);

  let assemblyFormat = [{
    `(` $A `:` type($A) `,` $f `)` attr-dict `:` type(results)
  }];
}

def Comco_Dropout : Comco_Op<"dropout", [Pure]> {
  let summary = "Dropout nodes within the threshold";
  let description = [{
      Compute Neural Network Dropout
      ```
  }];

  let arguments = (ins F32Tensor:$A, F32:$f);
  let results = (outs F32Tensor);

  let assemblyFormat = [{
    `(` $A `:` type($A) `,` $f `)` attr-dict `:` type(results)
  }];
}

def Comco_Print: Comco_Op<"print", [Pure]> {
  let summary = "Print operation (for testing and debugging)";
  let description = [{
      Print a tensor for testing and debugging.
      ```
  }];

  let arguments = (ins F32Tensor:$A);

  let assemblyFormat = [{
    `(` $A `:` type($A) `)` attr-dict
  }];
}

def Comco_Func: Comco_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "comco.func" operation represents a user defined function. These are
    callable SSA-region operations that contain toy computations.

    Example:

    ```mlir
    comco.func @main() {
      comco.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region on the function operation that is callable.
    Region *getCallableRegion() { return &getBody(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def Comco_Return: Comco_Op<"return", [Pure, HasParent<"Func">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      comco.func @foo() -> tensor<2xf32> {
        ...
        comco.return %0 : tensor<2xf32>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F32Tensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

#endif // COMCO_OPS
