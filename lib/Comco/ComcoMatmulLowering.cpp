#include "Comco/ComcoDialect.h"
#include "Comco/ComcoOps.h"
#include "Comco/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
struct ComcoMatmulLoweringPass
    : public PassWrapper<ComcoMatmulLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComcoMatmulLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // Input/output dialects. Overall scope of dialects.
    registry.insert<comco::ComcoDialect, func::FuncDialect, gpu::GPUDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() final;
};
} // namespace

std::unique_ptr<Pass> mlir::comco::createLowerMatmulPass() {
  return std::make_unique<ComcoMatmulLoweringPass>();
}

struct MatmulToGPULaunchLowering : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const final {
    // Check that the op is not already inside a gpu.func
    if (mlir::isa<gpu::GPUFuncOp>(matmulOp->getParentOp())) {
      // Should never happen (for now)
      DEBUG("parent is a func op -- nothing to do");
      return failure();
    }

    // Find the module op of the matmul
    ModuleOp moduleOp = matmulOp->getParentOfType<ModuleOp>();

    // Find the gpu module (assume one exists)
    bool foundGpuModuleOp = false;
    gpu::GPUModuleOp gpuModuleOp;
    for (auto &op : moduleOp) {
      if (mlir::isa<gpu::GPUModuleOp>(op)) {
        foundGpuModuleOp = true;
        gpuModuleOp = mlir::cast<gpu::GPUModuleOp>(op);
        break;
      }
    }

    // Just assume for now that one exists
    if (!foundGpuModuleOp) {
      DEBUG("unreachable");
      return failure();
    }

    // Get the gpu kernel itself
    // Note we are assuming that there is already an allreduce in here

    // The first op should be a func op
    gpu::GPUFuncOp gpuKernel =
        mlir::cast<gpu::GPUFuncOp>(gpuModuleOp.getBody()->front());

    auto returnOp = gpuKernel.getBlocks().front().getTerminator();

    // Get the ins() and outs() of the matmul

    // matmulOp->getOperand();
    // matmulOp.getResult(0);

    // Find the launch operation by iterating through every op in the matmul's
    // containing block
    auto opBlock = matmulOp->getBlock();
    bool foundLaunchOp = false;
    gpu::LaunchFuncOp launchOp;
    for (auto &op : *opBlock) {
      if (mlir::isa<gpu::LaunchFuncOp>(op)) {
        foundLaunchOp = true;
        launchOp = mlir::cast<gpu::LaunchFuncOp>(op);
      }
    }

    // Assume we have a launch func op already
    if (!foundLaunchOp) {
      DEBUG("unreachable: expecting launch op");
      return failure();
    }
    // Update the args() of the launch to include the matmul ins() and outs()

    // Move the matmul into the kernel function
    rewriter.modifyOpInPlace(matmulOp,
                             [&]() { matmulOp->moveBefore(returnOp); });

    Value ins = matmulOp->getOperand(0);
    Value outs = matmulOp->getOperand(1);

    Type inTy = ins.getType();

    // Add the dependencies of the matmul in the launchOp call
    rewriter.modifyOpInPlace(launchOp, [&]() {
      // Add the params
      // launchOp.insertArgument(0);
    });

    DEBUG("dump");
    moduleOp.dump();

    // Add the dependencies of the matmul as arguments to the kernel op
    rewriter.modifyOpInPlace(gpuKernel, [&]() {
      // TODO: add the ins and outs of the matmul as args
      gpuKernel.insertArgument(0, inTy, DictionaryAttr{}, gpuKernel.getLoc());
    });
    return success();
  }

private:
  void outlineKernel(linalg::MatmulOp op, PatternRewriter &rewriter) const {
    // TODO(mid): Parameterize this
    Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    Value four = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 4);
    // TODO(low): can use getParentOfType to add a proper kernel module section
    auto launchOp = rewriter.create<gpu::LaunchOp>(op.getLoc(), four, one, one,
                                                   one, one, one);
    rewriter.setInsertionPointToEnd(&launchOp.getBody().back());

    auto termOp = rewriter.create<gpu::TerminatorOp>(op.getLoc());
    rewriter.modifyOpInPlace(op, [&]() { op->moveBefore(termOp); });
    rewriter.setInsertionPoint(termOp);
  }

  LogicalResult lowerMatmulToLoops(linalg::MatmulOp op,
                                   PatternRewriter &rewriter) const {

    llvm::FailureOr<linalg::LinalgLoops> matmulOps =
        linalg::linalgOpToAffineLoops(rewriter, op);
    if (failed(matmulOps)) {
      return failure();
    }

    return success();
  }

  DiagnosedSilenceableFailure createGpuLaunch(
      RewriterBase &rewriter, Location loc, gpu::LaunchOp &launchOp,
      std::optional<int64_t> gridDimX, std::optional<int64_t> gridDimY,
      std::optional<int64_t> gridDimZ, std::optional<int64_t> blockDimX,
      std::optional<int64_t> blockDimY,
      std::optional<int64_t> blockDimZ) const {

    auto createConst = [&](int dim) {
      return rewriter.create<arith::ConstantIndexOp>(loc, dim);
    };
    OpBuilder::InsertionGuard guard(rewriter);
    Value one = createConst(1);
    Value gridSizeX =
        gridDimX.has_value() ? createConst(gridDimX.value()) : one;
    Value gridSizeY =
        gridDimY.has_value() ? createConst(gridDimY.value()) : one;
    Value gridSizeZ =
        gridDimZ.has_value() ? createConst(gridDimZ.value()) : one;
    Value blkSizeX =
        blockDimX.has_value() ? createConst(blockDimX.value()) : one;
    Value blkSizeY =
        blockDimY.has_value() ? createConst(blockDimY.value()) : one;
    Value blkSizeZ =
        blockDimZ.has_value() ? createConst(blockDimZ.value()) : one;
    launchOp = rewriter.create<gpu::LaunchOp>(
        loc, gridSizeX, gridSizeY, gridSizeZ, blkSizeX, blkSizeY, blkSizeZ);
    return DiagnosedSilenceableFailure::success();
  }
};

void ComcoMatmulLoweringPass::runOnOperation() {
  // Define target
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect, gpu::GPUDialect,
                         arith::ArithDialect>();
  target.addIllegalDialect<comco::ComcoDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<MatmulToGPULaunchLowering>(&getContext());

  walkAndApplyPatterns(getOperation(), std::move(patterns), nullptr);
}
