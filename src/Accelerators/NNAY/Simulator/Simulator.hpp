#ifndef NNAY_SIMULATOR_HPP
#define NNAY_SIMULATOR_HPP

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace onnx_mlir {
namespace nnay {
namespace simulator {

// Simulator class
class Simulator {
public:
  // Constructor
  Simulator(mlir::ModuleOp module);

  // Run simulation
  void run(mlir::Value input);

  // Get simulation context

private:
  // MLIR module
  mlir::ModuleOp module_;

  // Main graph function
  mlir::func::FuncOp mainGraphFunc_;


  // Process single operation
  bool processOperation(mlir::Operation *op);

  // Initialize simulator
  void initialize();
};

} // namespace simulator
} // namespace nnay
} // namespace onnx_mlir

#endif // NNAY_SIMULATOR_HPP