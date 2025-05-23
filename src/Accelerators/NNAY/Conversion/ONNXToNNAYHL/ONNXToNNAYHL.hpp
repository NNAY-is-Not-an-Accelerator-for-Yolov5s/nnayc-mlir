#ifndef ONNX_MLIR_ONNX_TO_NNAYHL_H
#define ONNX_MLIR_ONNX_TO_NNAYHL_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace onnx_mlir {

// Exports ONNXtoNNAYHL patterns.
void getONNXToNNAYHLPatterns(mlir::RewritePatternSet &patterns);

// Exports ONNXtoNNAYHL dynamically legal checks.
void getONNXToNNAYHLDynamicallyLegal(mlir::ConversionTarget *target);

} // namespace onnx_mlir

#endif
