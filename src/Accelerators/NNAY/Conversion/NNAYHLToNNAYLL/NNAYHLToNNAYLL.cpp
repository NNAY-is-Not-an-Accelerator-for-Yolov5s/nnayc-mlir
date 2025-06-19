#include "src/Accelerators/NNAY/Conversion/NNAYHLToNNAYLL/NNAYHLToNNAYLL.hpp"

#include "Dialect/MX/MXAttributes.hpp"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "src/Accelerators/NNAY/Dialect/MX/MXOps.hpp"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"
#include <deque>

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

namespace onnx_mlir::nnay {

void GlobalizeMXConstantsPass::runOnOperation() {
  auto module = getOperation();

  auto moduleBuilder = mlir::OpBuilder(module->getContext());
  moduleBuilder.setInsertionPointToEnd(module.getBody());

  std::deque<mx::ConstantOp> constantsToProcess;

  module->walk([&](mx::ConstantOp op) { constantsToProcess.push_back(op); });

  for (mx::ConstantOp constantOp : constantsToProcess) {
    if (!constantOp.getOperation()) {
      continue;
    }

    mx::MXBlockElementsAttr attr =
        dyn_cast_or_null<mx::MXBlockElementsAttr>(constantOp.getValue());
    if (!attr) {
      constantOp->emitError()
          << "mx.Constant does not have a MXBlockElementsAttr value, skipping";
      continue;
    }

    SmallString<128> globalName;
    for (auto user : constantOp->getResult(0).getUsers()) {
      if (auto convActOp = dyn_cast<nnayhl::ConvAct>(user)) {
        if (convActOp.getWeights() == constantOp->getResult(0)) {
          if (!globalName.empty()) {
            globalName += ";";
          }
          globalName +=
              cast<StringAttr>(convActOp->getAttr("onnx_node_name")).getValue();
          globalName += "_Weights";
        }
      }
    }

    mlir::ShapedType tensorType = cast<mlir::ShapedType>(constantOp.getType());
    mlir::MemRefType globalMemRefType = mlir::MemRefType::get(
        tensorType.getShape(), tensorType.getElementType());

    mlir::memref::GlobalOp globalOp =
        moduleBuilder.create<mlir::memref::GlobalOp>(module->getLoc(),
            globalName, moduleBuilder.getStringAttr("private"),
            globalMemRefType, attr, false, nullptr);

    mlir::OpBuilder replaceBuilder(constantOp);
    mlir::memref::GetGlobalOp getGlobalOp =
        replaceBuilder.create<mlir::memref::GetGlobalOp>(
            constantOp->getLoc(), globalMemRefType, globalOp.getName());

    constantOp->replaceAllUsesWith(getGlobalOp->getResults());
    constantOp->erase();
  }
}

std::unique_ptr<mlir::Pass> createGlobalizeMXConstantsPass() {
  return std::make_unique<GlobalizeMXConstantsPass>();
}

} // namespace onnx_mlir::nnay
