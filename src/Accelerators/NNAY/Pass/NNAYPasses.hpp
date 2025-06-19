#ifndef ONNX_MLIR_NNAY_PASSES_H
#define ONNX_MLIR_NNAY_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace onnx_mlir {
namespace nnay {

std::unique_ptr<mlir::Pass> createONNXToNNAYHLPass();
std::unique_ptr<mlir::Pass> createRemoveUnusedConstantPass();
std::unique_ptr<mlir::Pass> createFoldConvActivationPass();
std::unique_ptr<mlir::Pass> createSimulatePass();
std::unique_ptr<mlir::Pass> createSplitFusePass();
std::unique_ptr<mlir::Pass> createConcatFusePass();
std::unique_ptr<mlir::Pass> createExportBinPass();

} // namespace nnay
} // namespace onnx_mlir

#endif // ONNX_MLIR_NNAY_PASSES_H
