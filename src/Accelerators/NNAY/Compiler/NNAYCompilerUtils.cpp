#include "NNAYCompilerUtils.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace nnay {

void addPassesNNAY(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt) {
}

} // namespace nnay
} // namespace onnx_mlir
