#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace {

struct SplitFusePattern : public OpRewritePattern<nnayhl::Split> {
  using OpRewritePattern<nnayhl::Split>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      nnayhl::Split splitOp, PatternRewriter &rewriter) const override {
    auto splitSizes = splitOp.getSplitSizes();

    // Check if split sizes are the same and equal to 2
    if (splitSizes.size() != 2 || splitSizes[0] != splitSizes[1]) {
      return failure();
    }

    if (auto convActOp =
            dyn_cast<nnayhl::ConvAct>(splitOp->getOperand(0).getDefiningOp())) {
      auto newResultTypes = splitOp->getResultTypes();
      auto oldOperands = convActOp->getOperands();
      mlir::NamedAttrList newAttrs = convActOp->getAttrs();
      newAttrs.set("split", rewriter.getBoolAttr(true));
      auto newConvActOp = rewriter.create<nnayhl::ConvAct>(
          convActOp->getLoc(), newResultTypes, oldOperands, newAttrs.getAttrs());
      rewriter.replaceOp(splitOp, newConvActOp->getResults());
      rewriter.eraseOp(convActOp);
      return success();
    }

    return failure();
  }
};

struct SplitFusePass
    : public PassWrapper<SplitFusePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitFusePass)

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<SplitFusePattern>(patterns.getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  StringRef getArgument() const override { return "nnay-split-fuse"; }
};

} // namespace

namespace onnx_mlir {
namespace nnay {

std::unique_ptr<mlir::Pass> createSplitFusePass() {
  return std::make_unique<SplitFusePass>();
}

} // namespace nnay
} // namespace onnx_mlir