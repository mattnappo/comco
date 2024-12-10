#include "Comco/ComcoDialect.h"
#include "Comco/ComcoOps.h"
#include "Comco/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>

#define DEBUG(x)                                                               \
  do {                                                                         \
    llvm::errs() << "\n" << x << "\n\n";                                       \
  } while (0);

using namespace mlir;

namespace {
struct ComcoAllReduceLoweringPass
    : public PassWrapper<ComcoAllReduceLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComcoAllReduceLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // Input/output dialects. Overall scope of dialects.
    registry.insert<comco::ComcoDialect, func::FuncDialect, gpu::GPUDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() final;
};
} // namespace

std::unique_ptr<Pass> mlir::comco::createLowerAllReducePass() {
  return std::make_unique<ComcoAllReduceLoweringPass>();
}

void ComcoAllReduceLoweringPass::runOnOperation() {
  // Define target
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect, arith::ArithDialect>();
  target.addIllegalDialect<comco::ComcoDialect>();
  // target.addIllegalOp<gpu::AllReduceOp>();

  RewritePatternSet patterns(&getContext());
  patterns.add<mlir::comco::GpuAllReduceRewrite>(&getContext());

  // mlir::populateGpuAllReducePatterns(patterns); // from mlir

  walkAndApplyPatterns(getOperation(), std::move(patterns), nullptr);

  // if (failed(
  //         applyPartialConversion(getOperation(), target,
  //         std::move(patterns))))
  //   signalPassFailure();
}
