#include "src/Accelerators/NNAY/Conversion/NNAYHLToNNAYLL/NNAYHLToNNAYLL.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"
#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLL.hpp"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace {
struct LowerConvActOpPattern : public mlir::OpRewritePattern<nnayhl::ConvAct> {
  using mlir::OpRewritePattern<nnayhl::ConvAct>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      nnayhl::ConvAct op, mlir::PatternRewriter &rewriter) const override {
    return failure();
  }
};
} // namespace