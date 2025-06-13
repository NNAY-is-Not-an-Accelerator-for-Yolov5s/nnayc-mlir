#include "src/Accelerators/NNAY/Simulator/Simulator.hpp"
#include "llvm/Support/raw_ostream.h"

namespace onnx_mlir {
namespace nnay {
namespace simulator {

// Constructor
Simulator::Simulator(mlir::ModuleOp module) : module_(module) { initialize(); }

// Initialize simulator
void Simulator::initialize() {
  // Find main graph function
  auto mainGraphFunc = module_.lookupSymbol<mlir::func::FuncOp>("main_graph");
  if (!mainGraphFunc) {
    llvm::outs() << "Error: main_graph function not found\n";
    return;
  }
  mainGraphFunc_ = mainGraphFunc;

  llvm::outs() << "Explicitly registering operation handlers...\n";
}

// Run simulation
void Simulator::run(mlir::Value input) {
  if (!mainGraphFunc_) {
    llvm::outs() << "Error: main_graph function not initialized\n";
    return;
  }

  // Get entry block
  auto &entryBlock = mainGraphFunc_.getBody().front();

  // Process each operation
  for (auto &op : entryBlock) {
    llvm::outs() << "Processing operation: " << op << "\n";

    if (!processOperation(&op)) {
      llvm::outs() << "Warning: Failed to process operation: " << op << "\n";
    }
  }
}

// Process single operation
bool Simulator::processOperation(mlir::Operation *op) {
  // Get operation name
  std::string opName = op->getName().getStringRef().str();
}

} // namespace simulator
} // namespace nnay
} // namespace onnx_mlir