#include "RegisterPasses.hpp"
#include "Conversion/NNAYHLToNNAYLL/NNAYHLToNNAYLL.hpp"
#include "Conversion/ONNXToMX/ONNXToMX.hpp"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "src/Accelerators/NNAY/Pass/NNAYPasses.hpp"
#include <memory>

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace onnx_mlir {
namespace nnay {

void registerPasses() {
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createONNXToNNAYHLPass(); });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createFoldConvActivationPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createSimulatePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mx::createONNXToMXPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createSplitFusePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConcatFusePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createGlobalizeMXConstantsPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createExportBinPass();
  });

  bufferization::registerBufferizationPasses();
}
} // namespace nnay
} // namespace onnx_mlir