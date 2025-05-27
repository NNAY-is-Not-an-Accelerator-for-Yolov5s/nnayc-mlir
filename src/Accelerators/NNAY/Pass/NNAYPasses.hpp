#ifndef ONNX_MLIR_NNAY_PASSES_H
#define ONNX_MLIR_NNAY_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace onnx_mlir {
namespace nnay {

std::unique_ptr<mlir::Pass> createONNXToNNAYHLPass();
std::unique_ptr<mlir::Pass> createRemoveUnusedConstantPass();
std::unique_ptr<mlir::Pass> createFoldConvActivationPass();

} // namespace nnay
} // namespace onnx_mlir

#endif // ONNX_MLIR_NNAY_PASSES_H
