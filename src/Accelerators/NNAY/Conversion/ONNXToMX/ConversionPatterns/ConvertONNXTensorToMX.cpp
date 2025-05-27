#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Accelerators/NNAY/Dialect/MX/MXOps.hpp"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace {

// Base class for tensor conversion to MX format
class ConvertTensorToMXBase {
protected:
  // Check if the constant is used as a specific operand in Conv
  bool isUsedAsConvOperand(ONNXConstantOp op, unsigned operandIdx,
      StringRef requiredLayout = "") const {
    Value constantResult = op.getResult();

    for (Operation *user : constantResult.getUsers()) {
      if (auto convOp = dyn_cast<nnayhl::ConvAct>(user)) {
        if (convOp.getOperand(operandIdx) == constantResult) {
          if (requiredLayout.empty())
            return true;

          if (auto layoutAttr =
                  convOp->getAttrOfType<StringAttr>("weight_layout")) {
            if (layoutAttr.getValue() == requiredLayout)
              return true;
          }
        }
      }
    }
    return false;
  }

  // Update Conv operations that use this constant
  void updateConvUsers(PatternRewriter &rewriter, Value oldValue,
      Value newValue, unsigned operandIdx, StringRef newLayout) const {
    for (Operation *user : oldValue.getUsers()) {
      if (auto convOp = dyn_cast<nnayhl::ConvAct>(user)) {
        if (convOp.getOperand(operandIdx) == oldValue) {
          convOp->setOperand(operandIdx, newValue);
          if (!newLayout.empty())
            convOp->setAttr("weight_layout", rewriter.getStringAttr(newLayout));
        }
      }
    }
  }
};

// Weight tensor conversion
struct ConvertWeightToMX : public ConvertTensorToMXBase,
                           public mlir::OpRewritePattern<ONNXConstantOp> {
  using mlir::OpRewritePattern<ONNXConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConstantOp op, PatternRewriter &rewriter) const override {
    // Check if this constant is used as Conv weight with OIHW layout
    if (!isUsedAsConvOperand(op, 1, "OIHW"))
      return failure();

    auto attr = op->getAttr("value");
    if (!attr)
      return failure();

    auto constAttr = dyn_cast<DisposableElementsAttr>(attr);
    if (!constAttr)
      return failure();

    auto shape = constAttr.getShape();
    if (shape.size() != 4)
      return failure();

    auto outputChannels = shape[0];
    auto inputChannels = shape[1];
    auto height = shape[2];
    auto width = shape[3];

    if (inputChannels % 16 != 0)
      return failure();

    auto values = constAttr.getValues<float>();

    // Pack values in OHIbWB format
    SmallVector<float> packedValues;
    for (int o = 0; o != outputChannels; ++o) {
      for (int h = 0; h != height; ++h) {
        for (int ib = 0; ib != inputChannels / 16; ++ib) {
          for (int w = 0; w != width; ++w) {
            for (int blockOffset = 0; blockOffset != 16; ++blockOffset) {
              auto i = ib * 16 + blockOffset;
              auto index = o * inputChannels * height * width +
                           i * height * width + h * width + w;
              packedValues.push_back(values[index]);
            }
          }
        }
      }
    }

    // Create new MX constant with packed layout
    ArrayRef<float> packedValuesRef(packedValues);
    auto newShape = RankedTensorType::get(
        {outputChannels, height, inputChannels / 16, width, 16},
        Float32Type::get(op->getContext()));

    auto newPackedAttr = DenseElementsAttr::get(newShape, packedValuesRef);
    auto mxConstOp = rewriter.create<onnx_mlir::nnay::mx::ConstantOp>(
        op->getLoc(), newShape, newPackedAttr);

    // Update users and replace old op
    updateConvUsers(
        rewriter, op.getResult(), mxConstOp.getResult(), 1, "OHIbWB");
    rewriter.replaceOp(op, mxConstOp.getResult());

    return success();
  }
};

// Bias tensor conversion
struct ConvertBiasToMX : public ConvertTensorToMXBase,
                         public mlir::OpRewritePattern<ONNXConstantOp> {
  using mlir::OpRewritePattern<ONNXConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXConstantOp op, PatternRewriter &rewriter) const override {
    // Check if this constant is used as Conv bias
    if (!isUsedAsConvOperand(op, 2))
      return failure();

    auto attr = op->getAttr("value");
    if (!attr)
      return failure();

    auto constAttr = dyn_cast<DisposableElementsAttr>(attr);
    if (!constAttr)
      return failure();

    auto shape = constAttr.getShape();
    if (shape.size() != 1) // Bias should be 1D
      return failure();

    auto values = constAttr.getValues<float>();

    // Convert values to SmallVector
    SmallVector<float> biasValues;
    for (auto value : values) {
      biasValues.push_back(value);
    }

    // Create new MX constant for bias
    auto newShape =
        RankedTensorType::get(shape, Float32Type::get(op->getContext()));
    auto newAttr =
        DenseElementsAttr::get(newShape, ArrayRef<float>(biasValues));
    auto mxConstOp = rewriter.create<onnx_mlir::nnay::mx::ConstantOp>(
        op->getLoc(), newShape, newAttr);

    // Update users and replace old op
    updateConvUsers(rewriter, op.getResult(), mxConstOp.getResult(), 2, "");
    rewriter.replaceOp(op, mxConstOp.getResult());

    return success();
  }
};

} // namespace

namespace onnx_mlir {
namespace nnay {

void populateConvertONNXTensorToMXPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertWeightToMX>(patterns.getContext());
  patterns.add<ConvertBiasToMX>(patterns.getContext());
}

} // namespace nnay
} // namespace onnx_mlir