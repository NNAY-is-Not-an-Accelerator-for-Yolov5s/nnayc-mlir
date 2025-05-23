#ifndef NNAY_COMPILER_UTILS_HPP
#define NNAY_COMPILER_UTILS_HPP

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"

namespace onnx_mlir {
namespace nnay {

void addPassesNNAY(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt);

} // namespace nnay
} // namespace onnx_mlir

#endif // NNAY_COMPILER_UTILS_HPP
