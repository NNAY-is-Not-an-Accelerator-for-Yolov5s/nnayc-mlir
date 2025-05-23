#include "ONNXToNNAYHL.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace {

template <typename NewOp, typename OldOp, typename... Args>
auto replaceOpWithNewOpAndSetOnnxNodeName(
    mlir::PatternRewriter &rewriter, OldOp oldOp, Args &&...args) {
  auto onnxNodeName = oldOp->getAttr("onnx_node_name");
  auto newOp =
      rewriter.replaceOpWithNewOp<NewOp>(oldOp, std::forward<Args>(args)...);
  newOp->setAttr("onnx_node_name", onnxNodeName);
  return newOp;
}

struct LowerAddOpPattern : public OpRewritePattern<mlir::ONNXAddOp> {
  using OpRewritePattern<mlir::ONNXAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::ONNXAddOp addOp, mlir::PatternRewriter &rewriter) const override {
    replaceOpWithNewOpAndSetOnnxNodeName<onnx_mlir::nnay::nnayhl::Add>(
        rewriter, addOp, addOp.getC().getType(), addOp.getA(), addOp.getB());
    return success();
  }
};

struct LowerSiLUOpPattern : public RewritePattern {
  LowerSiLUOpPattern(mlir::MLIRContext *context)
      : RewritePattern("onnx.Mul", 1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, mlir::PatternRewriter &rewriter) const override {
    auto mulOp = cast<mlir::ONNXMulOp>(op);
    Operation *sigmoidOp = nullptr;
    Value otherOperand = nullptr;

    if (auto sig = mulOp->getOperand(1).getDefiningOp<mlir::ONNXSigmoidOp>()) {
      sigmoidOp = sig;
      otherOperand = mulOp->getOperand(0);
    } else if (auto sig =
                   mulOp->getOperand(0).getDefiningOp<mlir::ONNXSigmoidOp>()) {
      sigmoidOp = sig;
      otherOperand = mulOp->getOperand(1);
    }

    if (!sigmoidOp || !otherOperand)
      return failure();

    if (sigmoidOp->getOperand(0) != otherOperand)
      return failure();

    replaceOpWithNewOpAndSetOnnxNodeName<onnx_mlir::nnay::nnayhl::SiLU>(
        rewriter, mulOp, mulOp.getType(), otherOperand);
    return success();
  }
};

struct LowerConvOpPattern : public OpRewritePattern<mlir::ONNXConvOp> {
  using OpRewritePattern<mlir::ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::ONNXConvOp convOp, mlir::PatternRewriter &rewriter) const override {
    auto op =
        replaceOpWithNewOpAndSetOnnxNodeName<onnx_mlir::nnay::nnayhl::Conv>(
            rewriter, convOp, convOp.getType(), convOp.getX(), convOp.getW(),
            convOp.getB(), convOp.getStridesAttr(), convOp.getPadsAttr());
    op->setAttr("input_layout", rewriter.getStringAttr("NCHW"));
    op->setAttr("weight_layout", rewriter.getStringAttr("OIHW"));
    return success();
  }
};

struct LowerConcatOpPattern : public OpRewritePattern<mlir::ONNXConcatOp> {
  using OpRewritePattern<mlir::ONNXConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXConcatOp concatOp,
      mlir::PatternRewriter &rewriter) const override {
    replaceOpWithNewOpAndSetOnnxNodeName<onnx_mlir::nnay::nnayhl::Concat>(
        rewriter, concatOp, concatOp.getType(), concatOp.getInputs(),
        concatOp.getAxisAttr());
    return success();
  }
};

struct LowerSplitOpPattern : public OpRewritePattern<mlir::ONNXSplitOp> {
  using OpRewritePattern<mlir::ONNXSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXSplitOp splitOp,
      mlir::PatternRewriter &rewriter) const override {
    auto splitPointsDefiningOp = splitOp.getSplit().getDefiningOp();
    auto splitPointsConstantOp =
        dyn_cast<mlir::ONNXConstantOp>(splitPointsDefiningOp);
    if (!splitPointsConstantOp) {
      llvm::errs() << "Split points tensor is not a constant\n";
      return failure();
    }

    auto splitPointsAttr =
        dyn_cast<DenseIntElementsAttr>(splitPointsConstantOp.getValueAttr());
    if (!splitPointsAttr) {
      auto disposable = dyn_cast<DisposableElementsAttr>(
          splitPointsConstantOp.getValueAttr());
      if (!disposable) {
        llvm::errs()
            << "Split points tensor is not a dense int elements attr\n";
        return failure();
      }
      splitPointsAttr =
          dyn_cast<DenseIntElementsAttr>(disposable.toDenseElementsAttr());
    }

    SmallVector<int64_t, 4> splitSizes;
    for (auto attr : splitPointsAttr.getValues<IntegerAttr>()) {
      splitSizes.push_back(attr.getInt());
    }

    auto splitSizesAttr = rewriter.getI64ArrayAttr(splitSizes);
    SmallVector<Type, 4> outputTypes;

    auto inputType = dyn_cast<RankedTensorType>(splitOp.getInput().getType());
    if (!inputType || !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          splitOp, "input must be statically shaped");

    auto inputShape = inputType.getShape();
    int64_t axis = splitOp.getAxis();

    outputTypes.reserve(splitSizes.size());

    for (int64_t size : splitSizes) {
      SmallVector<int64_t, 4> outShape(inputShape.begin(), inputShape.end());
      outShape[axis] = size;

      outputTypes.push_back(
          RankedTensorType::get(outShape, inputType.getElementType()));
    }

    replaceOpWithNewOpAndSetOnnxNodeName<onnx_mlir::nnay::nnayhl::Split>(
        rewriter, splitOp, outputTypes, splitOp.getInput(), splitOp.getAxis(),
        splitSizesAttr);
    return success();
  }
};

} // namespace

namespace onnx_mlir {

void getONNXToNNAYHLPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<LowerAddOpPattern, LowerSiLUOpPattern, LowerConvOpPattern,
      LowerConcatOpPattern, LowerSplitOpPattern>(patterns.getContext());
}

void getONNXToNNAYHLDynamicallyLegal(mlir::ConversionTarget *target) {
  target->addDynamicallyLegalOp<mlir::ONNXAddOp>(
      [](mlir::ONNXAddOp op) { return true; });
}

} // namespace onnx_mlir