//===- comco-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Comco/ComcoDialect.h"
#include "Comco/ComcoOpsDialect.cpp.inc"

using namespace mlir;

llvm::LogicalResult makeMlirOpt(int argc, char **argv, llvm::StringRef toolName,
                                DialectRegistry &registry) {

  // Register and parse command line options.
  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, toolName, registry);

  //    return MlirOptMain(argc, argv, inputFilename, outputFilename, registry);

  llvm::InitLLVM y(argc, argv);

  MlirOptMainConfig config = MlirOptMainConfig::createFromCLOptions();
  //   config.passPipelineCallback();

  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user know
  // about it!
  //   if (inputFilename == "-" &&
  //       llvm::sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
  //     llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to
  //     "
  //                     "interrupt)\n";

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return llvm::failure();
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return llvm::failure();
  }
  if (failed(MlirOptMain(output->os(), std::move(file), registry, config)))
    return llvm::failure();

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return llvm::success();
}

int main(int argc, char **argv) {
  registerAllPasses();
  DialectRegistry registry;
  registry.insert<comco::ComcoDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<affine::AffineDialect>();
  registry.insert<gpu::GPUDialect>();
  registry.insert<mesh::MeshDialect>();

  // Include all MLIR Core dialects (for parsing)
  // registerAllDialects(registry);

  // Output file is stdout
  llvm::outs();

  // Read from stdin
  if (argc == 1) {

  } else if (argc == 2) {
    // TODO: Read input file
    // std::string inputFilename = argv[1];
    // std::string errMsg;
    // auto file = mlir::openInputFile(inputFilename, &errMsg);
    // if (!file) {
    //     llvm::errs() << errMsg << "\n";
    //     return mlir::asMainReturnCode(llvm::failure());
    // }
  }

  return asMainReturnCode(
      makeMlirOpt(argc, argv, "Comco optimizer driver\n", registry));
}