#include "src/Accelerators/NNAY/Pass/NNAYPasses.hpp"
#include "src/Accelerators/NNAY/Conversion/ONNXToNNAYHL/ONNXToNNAYHL.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

struct ONNXToNNAYHLPass : public mlir::PassWrapper<ONNXToNNAYHLPass,
                              mlir::OperationPass<mlir::ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToNNAYHLPass)

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    onnx_mlir::getONNXToNNAYHLPatterns(patterns);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(&getContext()))
          << "Failed to convert ONNX to NNAYHL";
    }
  }
};
} // namespace

namespace onnx_mlir {
namespace nnay {

std::unique_ptr<mlir::Pass> createONNXToNNAYHLPass() {
  return std::make_unique<ONNXToNNAYHLPass>();
}

} // namespace nnay
} // namespace onnx_mlir