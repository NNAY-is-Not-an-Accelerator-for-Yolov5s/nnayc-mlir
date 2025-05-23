#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Accelerators/NNAY/Dialect/MX/MXOps.hpp"
#include "src/Accelerators/NNAY/Dialect/MX/MXTypes.hpp"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace {

struct PackConvInputTensor : public mlir::RewritePattern {
  PackConvInputTensor(mlir::MLIRContext *context)
      : mlir::RewritePattern("nnayhl.Conv", 1, context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
    if (!isa<nnayhl::Conv>(op))
      return mlir::failure();
    auto convOp = cast<nnayhl::Conv>(op);
    auto inputOperand = convOp.getInput();

    // element type of input operand must be F32
    if (auto inputTensorType = dyn_cast<RankedTensorType>(inputOperand.getType())) {
      if (!inputTensorType.getElementType().isF32()) {
        return mlir::failure();
      }
    } else {
      return mlir::failure();
    }

    auto inputShape = dyn_cast<RankedTensorType>(inputOperand.getType());
    if (!inputShape || inputShape.getRank() != 4) {
      return mlir::failure();
    }

    if (inputShape.getDimSize(1) % 16 != 0) {
      return mlir::failure();
    }

    auto n = inputShape.getDimSize(0);
    auto c = inputShape.getDimSize(1);
    auto h = inputShape.getDimSize(2);
    auto w = inputShape.getDimSize(3);

    mx::MXBlockType blockType =
        mx::MXBlockType::get(rewriter.getContext(), 16, 2, 8, 1, 7);

    RankedTensorType outputShape = RankedTensorType::get(
        {n, h, c / 16, w}, blockType);

    auto packed = rewriter.create<mx::PackInputTensorOp>(
        convOp.getLoc(), outputShape, inputOperand);

    SmallVector<Value> newOperands = convOp.getOperands();
    newOperands[0] = packed;

    auto newConvOp = rewriter.create<nnayhl::Conv>(convOp.getLoc(),
        convOp->getResultTypes(), newOperands, convOp->getAttrs());
    newConvOp->setAttr("input_layout", rewriter.getStringAttr("NHCbWB"));
    rewriter.replaceOp(convOp, newConvOp->getResults());
    return mlir::success();
  }
};

} // namespace

namespace onnx_mlir {
namespace nnay {
namespace mx {

void populatePackConvInputTensorPatterns(RewritePatternSet &patterns) {
  patterns.add<PackConvInputTensor>(patterns.getContext());
}

} // namespace mx
} // namespace nnay
} // namespace onnx_mlir