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
struct ComcoLauncherPass
    : public PassWrapper<ComcoLauncherPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComcoLauncherPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // Input/output dialects. Overall scope of dialects.
    registry.insert<comco::ComcoDialect, func::FuncDialect, gpu::GPUDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() final;
};
} // namespace

std::unique_ptr<Pass> mlir::comco::createLauncherPass() {
  return std::make_unique<ComcoLauncherPass>();
}

template <typename OpTy>
LogicalResult placeInLaunchOp(PatternRewriter &rewriter, OpTy op) {

  Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value four = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 4);
  auto launchOp = rewriter.create<gpu::LaunchOp>(op.getLoc(), four, one, one,
                                                 four, one, one);
  rewriter.setInsertionPointToEnd(&launchOp.getBody().back());
  auto termOp = rewriter.create<gpu::TerminatorOp>(op.getLoc());
  rewriter.modifyOpInPlace(op, [&]() { op->moveBefore(termOp); });
  rewriter.setInsertionPoint(termOp);
  return success();
}

struct AllReduceLauncher : OpRewritePattern<gpu::AllReduceOp> {
  using OpRewritePattern<gpu::AllReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::AllReduceOp reduceOp,
                                PatternRewriter &rewriter) const final {
    return placeInLaunchOp(rewriter, reduceOp);
  }
};

struct MatmulLauncher : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const final {
    return placeInLaunchOp(rewriter, matmulOp);
  }
};

void ComcoLauncherPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<AllReduceLauncher>(&getContext());
  patterns.add<MatmulLauncher>(&getContext());

  walkAndApplyPatterns(getOperation(), std::move(patterns), nullptr);
}
