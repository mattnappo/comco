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
struct ComcoFusionPass
    : public PassWrapper<ComcoFusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComcoFusionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // Input/output dialects. Overall scope of dialects.
    registry.insert<comco::ComcoDialect, func::FuncDialect, gpu::GPUDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() final;
};
} // namespace

std::unique_ptr<Pass> mlir::comco::createFusionPass() {
  return std::make_unique<ComcoFusionPass>();
}

// For a matmul followed by an AllReduce.
struct AllReduceMatmulFusion : OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final {

    // Goal file: matmul-allreduce.mlir
    // Find the matmul func
    // Find its term op
    // Move the allreduce before the term op
    // Delete the old launch

    // Block funcBlocks = funcOp.getBlocks();

    // gpu::GPUModuleOp gpuModuleOp;
    bool found = false;
    // Block &funcBody = funcOp.getFunctionBody().front();
    auto blk = funcOp->getBlock();
    gpu::GPUFuncOp matmulFunc;
    gpu::GPUFuncOp allreduceFunc;
    for (auto &op : *blk) {
      if (mlir::isa<gpu::GPUModuleOp>(op)) {
        gpu::GPUModuleOp gpuModuleOp = mlir::cast<gpu::GPUModuleOp>(op);
        Block *gpuModBody = gpuModuleOp.getBody();
        Operation *gpuFuncOp = &gpuModBody->front();
        gpu::GPUFuncOp funcOp = mlir::cast<gpu::GPUFuncOp>(gpuFuncOp);
        if (found) {
          allreduceFunc = funcOp;
        } else {
          matmulFunc = funcOp;
        }
        found = true;
      }
    }

    if (!found) {
      DEBUG("unreachable (assumption)");
      return failure();
    }

    linalg::MatmulOp matmulOp;
    if (find<linalg::MatmulOp>(matmulFunc, &matmulOp)) {
      DEBUG("no matmul");
      return failure();
    }

    gpu::AllReduceOp reduceOp;
    if (find<gpu::AllReduceOp>(allreduceFunc, &reduceOp)) {
      DEBUG("no all reduce");
      return failure();
    }

    // Find the second launch
    auto allReduceLaunchOp = funcOp.getFunctionBody().front().end();
    allReduceLaunchOp--;
    allReduceLaunchOp--;

    // Get the first arg from the launch op
    auto arg = allReduceLaunchOp->getOperand(6).getDefiningOp();
    // Move the arg right after the matmul
    rewriter.moveOpAfter(arg, matmulOp);

    // Move the reduceOp right after the newly-moved arg
    rewriter.moveOpAfter(reduceOp, arg);

    // Update the reduceOp to use the new arg
    rewriter.modifyOpInPlace(reduceOp,
                             [&]() { reduceOp.setOperand(arg->getResult(0)); });

    rewriter.eraseOp(&*allReduceLaunchOp);

    auto allreduceModule = allreduceFunc->getParentOfType<gpu::GPUModuleOp>();
    rewriter.eraseOp(allreduceModule);

    // Lower to loops
    llvm::FailureOr<linalg::LinalgLoops> matmulOps =
        linalg::linalgOpToAffineLoops(rewriter, matmulOp);
    if (failed(matmulOps)) {
      return failure();
    }

    return success();
  }

private:
  template <typename OpTy>
  bool find(gpu::GPUFuncOp func, OpTy *out) const {

    Block &block = func.getFunctionBody().front();
    for (auto &op : block) {
      if (mlir::isa<OpTy>(op)) {
        *out = mlir::cast<OpTy>(op);
        return 0;
      }
    }
    return 1;
  }
};

void ComcoFusionPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<AllReduceMatmulFusion>(&getContext());

  walkAndApplyPatterns(getOperation(), std::move(patterns), nullptr);
}
