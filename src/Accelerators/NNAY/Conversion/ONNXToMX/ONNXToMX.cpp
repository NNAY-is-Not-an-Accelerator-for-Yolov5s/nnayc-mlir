#include "ONNXToMX.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace onnx_mlir {
namespace nnay {

void populateConvertONNXTensorToMXPatterns(RewritePatternSet &patterns);

} // namespace nnay
} // namespace onnx_mlir

namespace {
struct ONNXToMXPass : public mlir::PassWrapper<ONNXToMXPass,
                          mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToMXPass)

  void runOnOperation() override;
};

void ONNXToMXPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  populateConvertONNXTensorToMXPatterns(patterns);

  if (mlir::failed(
          mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    mlir::emitError(mlir::UnknownLoc::get(&getContext()))
        << "Failed to convert ONNX to MX";
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<mlir::Pass> onnx_mlir::nnay::mx::createONNXToMXPass() {
  return std::make_unique<ONNXToMXPass>();
}

void onnx_mlir::nnay::mx::registerONNXToMXPasses() {
  PassRegistration<ONNXToMXPass>();
}
