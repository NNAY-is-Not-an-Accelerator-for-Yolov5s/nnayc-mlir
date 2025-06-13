#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "src/Accelerators/NNAY/Simulator/Simulator.hpp"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace {

struct SimulatePass
    : public PassWrapper<SimulatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimulatePass)

  void runOnOperation() override {
    auto module = getOperation();
    auto mainGraphFunc = module.lookupSymbol<mlir::func::FuncOp>("main_graph");
    if (!mainGraphFunc) {
      emitError(UnknownLoc::get(module.getContext()))
          << "main_graph function not found";
      return;
    }

    auto input = mainGraphFunc.getArgument(0);

    // Use the new simulator framework
    simulator::Simulator simulator(module);
    simulator.run(input);
  }

  mlir::StringRef getArgument() const override { return "nnay-simulate"; }
};

} // namespace

namespace onnx_mlir {
namespace nnay {

std::unique_ptr<mlir::Pass> createSimulatePass() {
  return std::make_unique<SimulatePass>();
}

} // namespace nnay
} // namespace onnx_mlir