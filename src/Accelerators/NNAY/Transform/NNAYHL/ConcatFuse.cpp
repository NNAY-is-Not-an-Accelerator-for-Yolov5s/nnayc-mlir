#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace {
struct ConcatFusePattern : public OpRewritePattern<nnayhl::ConvAct> {
  using OpRewritePattern<nnayhl::ConvAct>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      nnayhl::ConvAct convActOp, PatternRewriter &rewriter) const override {
    if (convActOp.getInputs().size() != 1) {
      return failure();
    }

    auto inputOp = convActOp.getInputs()[0];
    if (inputOp.getDefiningOp() == nullptr) {
      return failure();
    }

    if (auto concatOp = dyn_cast<nnayhl::Concat>(inputOp.getDefiningOp())) {
      SmallVector<Value> newOperands;
      for (auto concatInput : concatOp.getInputs()) {
        newOperands.push_back(concatInput);
      }
      newOperands.push_back(convActOp.getWeights());
      newOperands.push_back(convActOp.getBias());
      auto newConvActOp = rewriter.create<nnayhl::ConvAct>(convActOp.getLoc(),
          convActOp.getResultTypes(), newOperands, convActOp->getAttrs());
      rewriter.replaceOp(convActOp, newConvActOp->getResults());
      rewriter.eraseOp(concatOp);
      return success();
    }
    return failure();
  }
};

struct ConcatFusePass
    : public PassWrapper<ConcatFusePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConcatFusePass)

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConcatFusePattern>(patterns.getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  StringRef getArgument() const override { return "nnay-concat-fuse"; }
};
} // namespace

namespace onnx_mlir {
namespace nnay {

std::unique_ptr<mlir::Pass> createConcatFusePass() {
  return std::make_unique<ConcatFusePass>();
}

} // namespace nnay
} // namespace onnx_mlir