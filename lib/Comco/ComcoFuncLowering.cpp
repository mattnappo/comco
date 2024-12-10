#include "Comco/ComcoDialect.h"
#include "Comco/ComcoOps.h"
#include "Comco/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>

#define DEBUG(x)                                                               \
  do {                                                                         \
    std::cout << std::endl << x << std::endl << std::endl;                     \
  } while (0);

using namespace mlir;

namespace {
struct ComcoFuncLoweringPass
    : public PassWrapper<ComcoFuncLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComcoFuncLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // Input/output dialects. Overall scope of dialects.
    registry.insert<comco::ComcoDialect, func::FuncDialect>();
  }

  void runOnOperation() final;
};
} // namespace

std::unique_ptr<Pass> mlir::comco::createLowerFuncPass() {
  return std::make_unique<ComcoFuncLoweringPass>();
}

struct ComcoFuncLowering : public OpConversionPattern<comco::Func> {
  using OpConversionPattern<comco::Func>::OpConversionPattern;

  // Convert a comco.func into a func.func
  LogicalResult
  matchAndRewrite(comco::Func op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    // Create a standard function
    auto func = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(),
                                              op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ComcoReturnLowering : OpRewritePattern<comco::Return> {
  using OpRewritePattern<comco::Return>::OpRewritePattern;

  LogicalResult matchAndRewrite(comco::Return op,
                                PatternRewriter &rewriter) const final {
    func::ReturnOp newOp =
        rewriter.create<func::ReturnOp>(op.getLoc(), op.getOperands());

    rewriter.replaceOp(op, newOp);

    return success();
  }
};

struct ComcoEnsureGPUContainer : OpRewritePattern<ModuleOp> {
  using OpRewritePattern<ModuleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModuleOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.modifyOpInPlace(op, [&]() {
      op->setAttr(StringRef("gpu.container_module"), rewriter.getUnitAttr());
    });

    return success();
  }
};

void ComcoFuncLoweringPass::runOnOperation() {
  // Define target
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect>();
  target.addIllegalDialect<comco::ComcoDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<ComcoFuncLowering>(&getContext());
  patterns.add<ComcoReturnLowering>(&getContext());
  patterns.add<ComcoEnsureGPUContainer>(&getContext());

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}