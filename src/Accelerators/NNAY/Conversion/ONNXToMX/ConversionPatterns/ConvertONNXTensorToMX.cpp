#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Accelerators/NNAY/Dialect/MX/MXOps.hpp"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace {

// Base class for tensor conversion to MX format
class ConvertTensorToMXBase {
protected:
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
    if (op->getUsers().empty())
      return failure();

    for (auto user : op->getUsers()) {
      if (auto convActOp = dyn_cast<nnayhl::ConvAct>(user)) {
        if (convActOp.getWeights() == op->getResult(0)) {
          auto attr = op->getAttr("value");
          if (!attr)
            return failure();
        } else {
          return failure();
        }
      } else {
        return failure();
      }
    }

    auto attr = op->getAttr("value");
    if (!attr)
      return failure();

    auto constAttr = dyn_cast<DenseElementsAttr>(attr);

    auto shape = constAttr.getType().getShape();
    if (shape.size() != 4)
      return failure();

    auto outputChannels = shape[0];
    auto inputChannels = shape[1];
    auto height = shape[2];
    auto width = shape[3];

    if (outputChannels % 16 != 0) {
      llvm::errs() << "Output channels must be divisible by 16\n";
      return failure();
    }

    auto values = constAttr.getValues<float>();

    // Pack values in OHIbWB format
    SmallVector<float> packedValues;

    auto ocGroupNum = outputChannels / 16;

    for (decltype(ocGroupNum) ocGroup = 0; ocGroup != ocGroupNum; ++ocGroup) {
      for (decltype(height) h = 0; h != height; ++h) {
        for (decltype(inputChannels) ic = 0; ic != inputChannels; ++ic) {
          for (decltype(width) w = 0; w != width; ++w) {
            for (decltype(16) blockOffset = 0; blockOffset != 16;
                ++blockOffset) {
              auto oc = ocGroup * 16 + blockOffset;
              auto index = oc * inputChannels * height * width +
                           ic * height * width + h * width + w;
              packedValues.push_back(values[index]);
            }
          }
        }
      }
    }

    // Create new MX constant with packed layout
    ArrayRef<float> packedValuesRef(packedValues);
    auto newShape = RankedTensorType::get(
        {ocGroupNum, height, inputChannels, width, 16},
        Float32Type::get(op->getContext()));

    auto newPackedAttr = DenseElementsAttr::get(newShape, packedValuesRef);
    auto mxConstOp = rewriter.create<onnx_mlir::nnay::mx::ConstantOp>(
        op->getLoc(), newShape, newPackedAttr);

    // Update users and replace old op
    updateConvUsers(
        rewriter, op.getResult(), mxConstOp.getResult(), 1, "OgHIWG");
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
  // patterns.add<ConvertBiasToMX>(patterns.getContext());
}

} // namespace nnay
} // namespace onnx_mlir
