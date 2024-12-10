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
#include <utility>

#define DEBUG(x)                                                               \
  do {                                                                         \
    std::cout << std::endl << x << std::endl << std::endl;                     \
  } while (0);

using namespace mlir;

namespace {
struct ComcoBufferLoweringPass
    : public PassWrapper<ComcoBufferLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComcoBufferLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // Input/output dialects. Overall scope of dialects.
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() final;
};
} // namespace

std::unique_ptr<Pass> mlir::comco::createLowerBufferPass() {
  return std::make_unique<ComcoBufferLoweringPass>();
}

struct LowerMemRefToGPU : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const final {

    rewriter.modifyOpInPlace(op, [&]() {
      // Place the memref in global GPU memory
      op->setAttr("memory_space", rewriter.getAttr<gpu::AddressSpaceAttr>(
                                      gpu::AddressSpace::Global));
    });

    return success();
  }
};

void ComcoBufferLoweringPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<LowerMemRefToGPU>(&getContext());

  walkAndApplyPatterns(getOperation(), std::move(patterns), nullptr);
}