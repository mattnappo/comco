//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/Diagnostics.h"

#include "Comco/ComcoDialect.h"
#include "Comco/ComcoOpsDialect.cpp.inc"
#include "Comco/Passes.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

int loadMLIR(std::string inputFilename, llvm::SourceMgr &sourceMgr,
             mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int dumpMLIR(std::string inputFilename) {
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  registry.insert<mlir::comco::ComcoDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  // TODO: add more input dialects
  registerAllDialects(registry); // whatever

  mlir::MLIRContext context(registry);
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::comco::ComcoDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  if (int error = loadMLIR(inputFilename, sourceMgr, context, module))
    return error;

  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Use fusion or not
  const char *fusionEnv = std::getenv("COMCO_FUSION");

  if (fusionEnv && std::strcmp(fusionEnv, "1") == 0) {
    // With fusion
    pm.addPass(mlir::bufferization::createOneShotBufferizePass());
    pm.addPass(mlir::comco::createLowerFuncPass());
    pm.addPass(mlir::comco::createLowerBufferPass());

    // Have all reduce search for a existing kernel
    // If one exists, put the all reduce at the end of the kernel along with its
    // dependencies.
    // If not, then create a new kernel and put the reduce op with
    // its dependencies in the kernel.

    // Have matmul search for a existing kernel
    // If one exists, put the matmul at the end of the kernel along with its
    // dependencies.
    // If not, then create a new kernel and put the matmul inside the kernel.
    // Will also need to figure out the inputs and outputs to the matmul and
    // make them inputs and outputs to the kernel function.

    // Then, and only then, perform the actual lowerings to the device code

    pm.addPass(mlir::comco::createOutlineAllReducePass());
    pm.addPass(mlir::comco::createLowerMatmulPass());
    // pm.addPass(mlir::createGpuKernelOutliningPass());
    // pm.addPass(mlir::comco::createLowerAllReducePass());
  } else if (fusionEnv && std::strcmp(fusionEnv, "outline") == 0) {
    pm.addPass(mlir::bufferization::createOneShotBufferizePass());
    pm.addPass(mlir::comco::createLowerFuncPass());
    pm.addPass(mlir::comco::createLowerBufferPass());

    pm.addPass(mlir::comco::createLauncherPass());
    pm.addPass(mlir::createGpuKernelOutliningPass());
    pm.addPass(mlir::comco::createFusionPass());
  } else {
    // Without fusion
    pm.addPass(mlir::bufferization::createOneShotBufferizePass());
    pm.addPass(mlir::comco::createLowerFuncPass());
    pm.addPass(mlir::comco::createLowerBufferPass());
    pm.addPass(mlir::comco::createLowerMatmulPass());
    pm.addPass(mlir::createGpuKernelOutliningPass());
    pm.addPass(mlir::comco::createOutlineAllReducePass());
    pm.addPass(mlir::comco::createLowerAllReducePass());
  }

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Failed to apply the pass manager.\n";
    return 4;
  }

  module->dump();
  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  std::string inputFilename = argv[1];
  return dumpMLIR(inputFilename);
}
