#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace {
class MoveReshapePastSigmoid : public OpRewritePattern<nnayhl::Sigmoid> {
public:
  using OpRewritePattern<nnayhl::Sigmoid>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      nnayhl::Sigmoid sigmoidOp, PatternRewriter &rewriter) const override {
    auto reshapeOp = sigmoidOp.getInput().getDefiningOp<mlir::ONNXReshapeOp>();
    if (!reshapeOp) {
      return failure();
    }

    auto reshapeInput = reshapeOp->getOperand(0);
    auto reshapeShape = reshapeOp->getOperand(1);

    auto newSigmoidType = reshapeInput.getType();
    if (!isa<mlir::RankedTensorType>(newSigmoidType)) {
      return failure();
    }

    auto newSigmoidOp = rewriter.create<nnayhl::Sigmoid>(sigmoidOp->getLoc(),
        newSigmoidType, reshapeInput, sigmoidOp->getAttrs());
    auto newReshapeOp =
        rewriter.create<mlir::ONNXReshapeOp>(reshapeOp->getLoc(),
            sigmoidOp.getType(), newSigmoidOp.getResult(), reshapeShape);
    rewriter.replaceOp(reshapeOp, newReshapeOp.getResult());
    return success();
  }
};

class MOveTransposePastSigmoid : public OpRewritePattern<nnayhl::Sigmoid> {
public:
  using OpRewritePattern<nnayhl::Sigmoid>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      nnayhl::Sigmoid sigmoidOp, PatternRewriter &rewriter) const override {
    auto transposeOp =
        sigmoidOp.getInput().getDefiningOp<mlir::ONNXTransposeOp>();
    if (!transposeOp) {
      return failure();
    }

    auto transposeInput = transposeOp->getOperand(0);
    auto newSigmoidType = transposeInput.getType();
    if (!isa<mlir::RankedTensorType>(newSigmoidType)) {
      return failure();
    }

    auto newSigmoidOp = rewriter.create<nnayhl::Sigmoid>(sigmoidOp->getLoc(),
        newSigmoidType, transposeInput, sigmoidOp->getAttrs());
    auto newTransposeOp = rewriter.create<mlir::ONNXTransposeOp>(
        transposeOp->getLoc(), sigmoidOp.getType(), newSigmoidOp.getResult(),
        transposeOp.getPermAttr());

    rewriter.replaceOp(transposeOp, newTransposeOp.getResult());
    return success();
  }
};

class MoveReshapeTransposePastSigmoid
    : public OpRewritePattern<nnayhl::Sigmoid> {
public:
  using OpRewritePattern<nnayhl::Sigmoid>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      nnayhl::Sigmoid sigmoidOp, PatternRewriter &rewriter) const override {
    auto transposeOp =
        sigmoidOp.getInput().getDefiningOp<mlir::ONNXTransposeOp>();
    if (!transposeOp)
      return failure();
    auto reshapeOp =
        transposeOp->getOperand(0).getDefiningOp<mlir::ONNXReshapeOp>();
    if (!reshapeOp)
      return failure();

    Value inputToReshape = reshapeOp->getOperand(0);
    auto newSigmoidType = inputToReshape.getType();
    if (!isa<mlir::RankedTensorType>(newSigmoidType)) {
      return failure();
    }
    auto newSigmoidOp = rewriter.create<nnayhl::Sigmoid>(sigmoidOp.getLoc(),
        newSigmoidType, inputToReshape, sigmoidOp->getAttrs());
    auto newReshapeOp = rewriter.create<mlir::ONNXReshapeOp>(
        reshapeOp->getLoc(), reshapeOp->getResult(0).getType(),
        newSigmoidOp.getResult(), reshapeOp->getOperand(1));

    auto newTransposeOp =
        rewriter.create<mlir::ONNXTransposeOp>(transposeOp.getLoc(),
            sigmoidOp.getType(), // Original sigmoid's output type
            newReshapeOp.getResult(), transposeOp.getPermAttr());

    rewriter.replaceOp(sigmoidOp, newTransposeOp.getResult());
    return success();
  }
};

} // namespace

namespace onnx_mlir::nnay {

struct OptimizeSigmoidLayoutPass
    : public mlir::PassWrapper<OptimizeSigmoidLayoutPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());

    patterns.add<MoveReshapeTransposePastSigmoid>(&getContext());
    patterns.add<MoveReshapePastSigmoid>(&getContext());
    patterns.add<MOveTransposePastSigmoid>(&getContext());

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "optimize-sigmoid-layout";
  }
};

std::unique_ptr<mlir::Pass> createOptimizeSigmoidLayoutPass() {
  return std::make_unique<OptimizeSigmoidLayoutPass>();
}

} // namespace onnx_mlir::nnay
