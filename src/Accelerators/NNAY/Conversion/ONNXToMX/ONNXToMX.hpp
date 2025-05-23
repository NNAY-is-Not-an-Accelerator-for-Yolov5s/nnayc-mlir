#ifndef ONNX_TO_MX_HPP
#define ONNX_TO_MX_HPP

#include "mlir/Pass/Pass.h"
#include <memory>

namespace onnx_mlir {
namespace nnay {
namespace mx {

std::unique_ptr<mlir::Pass> createONNXToMXPass();

void registerONNXToMXPasses();

} // namespace mx
} // namespace nnay
} // namespace onnx_mlir

#endif // ONNX_TO_MX_HPP
