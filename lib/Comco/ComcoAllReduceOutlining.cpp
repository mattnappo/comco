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

std::unique_ptr<Pass> mlir::comco::createOutlineAllReducePass() {
  return std::make_unique<ComcoLauncherPass>();
}

struct AllReduceOutlining : OpRewritePattern<gpu::AllReduceOp> {
  using OpRewritePattern<gpu::AllReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::AllReduceOp reduceOp,
                                PatternRewriter &rewriter) const final {
    // Check that the op is not already inside a gpu.func
    if (mlir::isa<gpu::GPUFuncOp>(reduceOp->getParentOp())) {
      DEBUG("parent is a func op -- nothing to do");
      return failure();
    }

    // Find the module op of the reduce op
    ModuleOp moduleOp = reduceOp->getParentOfType<ModuleOp>();

    // TODO: Check there isn't already a gpu module

    // Insert a new gpu.module in the module at the top
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto gpuModule = rewriter.create<gpu::GPUModuleOp>(moduleOp.getLoc(),
                                                       StringRef("kernels"));

    // Insert a gpu.func kernel into the gpu.module
    rewriter.setInsertionPointToStart(gpuModule.getBody());
    auto gpuFunc = rewriter.create<gpu::GPUFuncOp>(
        gpuModule.getLoc(), StringRef("comco_kernel"),
        FunctionType::get(getContext(), {}, {}), TypeRange{}, TypeRange{},
        ArrayRef<NamedAttribute>{});
    rewriter.modifyOpInPlace(gpuFunc, [&]() {
      // Add the gpu.kernel attribute
      gpuFunc->setAttr(StringRef("gpu.kernel"), rewriter.getUnitAttr());
    });

    // Put a gpu.return in the end of the kernel func
    rewriter.setInsertionPointToEnd(&gpuFunc.getBody().front());
    auto returnOp = rewriter.create<gpu::ReturnOp>(gpuFunc.getLoc());

    rewriter.setInsertionPoint(reduceOp);

    // Setup for creating a gpu.launch_func
    Value kb =
        rewriter.create<arith::ConstantIntOp>(reduceOp.getLoc(), 1024, 32);
    Value one = rewriter.create<arith::ConstantIntOp>(reduceOp.getLoc(), 1, 32);
    Value four =
        rewriter.create<arith::ConstantIntOp>(reduceOp.getLoc(), 4, 32);

    // Create a gpu.launch_func in the op's original body
    rewriter.create<gpu::LaunchFuncOp>(
        reduceOp.getLoc(), gpuFunc, gpu::KernelDim3{four, one, one},
        gpu::KernelDim3{four, one, one}, kb, ValueRange{}, Type(), ValueRange{},
        std::nullopt);

    // Move the reduceOp itself into the kernel function
    rewriter.modifyOpInPlace(reduceOp,
                             [&]() { reduceOp->moveBefore(returnOp); });

    // Move the dependencies of the reduceOp inside the kernel function
    auto arg = reduceOp->getOperand(0).getDefiningOp();
    rewriter.moveOpBefore(arg, reduceOp);

    return success();
  }
};

void ComcoLauncherPass::runOnOperation() {
  // Define target
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect, arith::ArithDialect>();
  target.addIllegalDialect<comco::ComcoDialect>();
  // target.addIllegalOp<gpu::AllReduceOp>();

  RewritePatternSet patterns(&getContext());
  patterns.add<AllReduceOutlining>(&getContext());

  walkAndApplyPatterns(getOperation(), std::move(patterns), nullptr);

  // if (failed(
  //         applyPartialConversion(getOperation(), target,
  //         std::move(patterns))))
  //   signalPassFailure();
}
