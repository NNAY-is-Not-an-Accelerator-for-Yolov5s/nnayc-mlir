#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "mlir/Pass/Pass.h"
#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLL.hpp"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace {

struct EmitNNAYAsmPass
    : public PassWrapper<EmitNNAYAsmPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    auto context = module->getContext();
    auto builder = OpBuilder(context);
  }
};

} // namespace

namespace onnx_mlir {
namespace nnay {} // namespace nnay
} // namespace onnx_mlir
