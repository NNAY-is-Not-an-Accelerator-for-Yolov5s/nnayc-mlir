#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.hpp"
#include "src/Accelerators/NNAY/Dialect/MX/MXOps.hpp"
#include "src/Accelerators/NNAY/Dialect/MX/MXTypes.hpp"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace {

struct ConvertResultToMX : public mlir::OpRewritePattern<nnayhl::ConvAct> {
  using mlir::OpRewritePattern<nnayhl::ConvAct>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      nnayhl::ConvAct op, PatternRewriter &rewriter) const override {
    if (cast<StringAttr>(op->getAttr("input_layout")) != "NCHW") {
      return failure();
    }

    auto resultTypes = op->getResultTypes();
    llvm::SmallVector<Type> newResultTypes;
    for (auto resultType : resultTypes) {
      if (isa<RankedTensorType>(resultType)) {
        auto shape = cast<RankedTensorType>(resultType).getShape();
        if (shape.size() != 4)
          return failure();
        auto n = shape[0];
        auto c = shape[1];
        auto h = shape[2];
        auto w = shape[3];
        if (c % 16 != 0) {
          op->emitWarning() << "Output channels must be divisible by 16, "
                               "result type will not be converted to MX";
          return failure();
        }
        auto newShape = RankedTensorType::get(
            {n, h, c / 16, w}, mx::MXBlockType::getMX9(op->getContext()));

        newResultTypes.push_back(newShape);
      } else {
        newResultTypes.push_back(resultType);
      }
    }

    if (!newResultTypes.empty()) {
      auto newOp = rewriter.replaceOpWithNewOp<nnayhl::ConvAct>(
          op, newResultTypes, op->getOperands(), op->getAttrs());
      newOp->setAttr("input_layout", rewriter.getStringAttr("NHCgWG"));
    }

    return failure();
  }
};

struct ConvertWeightToMX : public mlir::OpRewritePattern<ONNXConstantOp> {
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
    auto values = constAttr.getValues<float>();

    SmallVector<char> mxData;

    auto ocGroupNum = (outputChannels + 15) / 16;
    auto lastGroupSize = outputChannels % 16;

    for (decltype(ocGroupNum) ocGroup = 0; ocGroup != ocGroupNum; ++ocGroup) {
      for (decltype(height) h = 0; h != height; ++h) {
        for (decltype(inputChannels) ic = 0; ic != inputChannels; ++ic) {
          for (decltype(width) w = 0; w != width; ++w) {
            SmallVector<float> blockData;
            for (decltype(16) blockOffset = 0; blockOffset != 16;
                ++blockOffset) {
              if (ocGroup == ocGroupNum - 1 && blockOffset >= lastGroupSize) {
                blockData.push_back(0.0f);
              } else {
                auto oc = ocGroup * 16 + blockOffset;
                auto index = oc * inputChannels * height * width +
                             ic * height * width + h * width + w;
                blockData.push_back(values[index]);
              }
            }
            auto mxBlockData = mx::getMXBlockData(blockData);
            mxData.append(mxBlockData.begin(), mxBlockData.end());
          }
        }
      }
    }

    // Create new MX constant with packed layout
    auto newShape =
        RankedTensorType::get({ocGroupNum, height, inputChannels, width},
            mx::MXBlockType::getMX9(op.getContext()));

    auto newPackedAttr = mx::MXBlockElementsAttr::get(newShape, mxData);

    auto mxConstOp = rewriter.create<onnx_mlir::nnay::mx::ConstantOp>(
        op->getLoc(), newShape, newPackedAttr);

    for (auto *user : op.getResult().getUsers()) {
      if (auto convActOp = dyn_cast<nnayhl::ConvAct>(user)) {
        if (convActOp.getWeights() == op->getResult(0)) {
          convActOp.getWeightsMutable().assign(mxConstOp.getResult());
          convActOp->setAttr("weight_layout", rewriter.getStringAttr("OgHIWG"));
        }
      }
    }

    rewriter.replaceOp(op, mxConstOp.getResult());

    return success();
  }
};

// Bias tensor conversion
// struct ConvertBiasToMX : public ConvertTensorToMXBase,
//                          public mlir::OpRewritePattern<ONNXConstantOp> {
//   using mlir::OpRewritePattern<ONNXConstantOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(
//       ONNXConstantOp op, PatternRewriter &rewriter) const override {
//     // Check if this constant is used as Conv bias
//     auto attr = op->getAttr("value");
//     if (!attr)
//       return failure();

//     auto constAttr = dyn_cast<DisposableElementsAttr>(attr);
//     if (!constAttr)
//       return failure();

//     auto shape = constAttr.getShape();
//     if (shape.size() != 1) // Bias should be 1D
//       return failure();

//     auto values = constAttr.getValues<float>();

//     // Convert values to SmallVector
//     SmallVector<float> biasValues;
//     for (auto value : values) {
//       biasValues.push_back(value);
//     }

//     // Create new MX constant for bias
//     auto newShape =
//         RankedTensorType::get(shape, Float32Type::get(op->getContext()));
//     auto newAttr =
//         DenseElementsAttr::get(newShape, ArrayRef<float>(biasValues));
//     auto mxConstOp = rewriter.create<onnx_mlir::nnay::mx::ConstantOp>(
//         op->getLoc(), newShape, newAttr);

//     // Update users and replace old op
//     updateConvUsers(rewriter, op.getResult(), mxConstOp.getResult(), 2, "");
//     rewriter.replaceOp(op, mxConstOp.getResult());

//     return success();
//   }
// };

} // namespace

namespace onnx_mlir {
namespace nnay {

void populateConvertONNXTensorToMXPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertWeightToMX, ConvertResultToMX>(patterns.getContext());
  // patterns.add<ConvertBiasToMX>(patterns.getContext());
}

} // namespace nnay
} // namespace onnx_mlir
