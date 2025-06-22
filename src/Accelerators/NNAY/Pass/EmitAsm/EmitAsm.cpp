#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.hpp"
#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLL.hpp"
#include "src/Accelerators/NNAY/Pass/EmitAsm/NNAYAsm.hpp"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace onnx_mlir::nnay {

struct EmitNNAYAsmPass
    : public PassWrapper<EmitNNAYAsmPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    auto nnayProgram = nnayasm::NNAYProgram();

    module->walk([&](mlir::memref::GlobalOp op) {
      if (auto value =
              dyn_cast<mx::MXBlockElementsAttr>(op.getInitialValueAttr())) {
        auto address = dyn_cast<mlir::IntegerAttr>(op->getAttr("address"));
        nnayProgram.appendWeightsBiasData(
            value.getRawData(), op.getName(), address.getInt());
      }
    });

    nnayProgram.emitMemoryBin("model.hwmdl");
  }

  llvm::StringRef getArgument() const override { return "emit-nnay-asm"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
  }
};

std::unique_ptr<mlir::Pass> createEmitNNAYAsmPass() {
  return std::make_unique<EmitNNAYAsmPass>();
}

} // namespace onnx_mlir::nnay
