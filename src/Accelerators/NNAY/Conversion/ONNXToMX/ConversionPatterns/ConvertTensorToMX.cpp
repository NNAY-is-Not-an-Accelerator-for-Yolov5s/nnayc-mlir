#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct ConvertTensorToMX : public mlir::OpRewritePattern<ONNXConstantOp> {
  using mlir::OpRewritePattern<ONNXConstantOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ONNXConstantOp op, mlir::PatternRewriter &rewriter) const override;
};

} // namespace

mlir::LogicalResult ConvertTensorToMX::matchAndRewrite(
    ONNXConstantOp op, mlir::PatternRewriter &rewriter) const {
  auto type = op.getType();
  if (!isa<mlir::TensorType>(type)) {
    llvm::errs() << "type is not a tensor type\n";
    return mlir::failure();
  }

  auto tensorType = cast<mlir::TensorType>(type);
  for (auto dim : tensorType.getShape()) {
    llvm::outs() << dim << " ";
  }
  llvm::outs() << "\n";
  return mlir::success();
}
