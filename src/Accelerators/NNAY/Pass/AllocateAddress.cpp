#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace nnay {

struct AllocateAddressPass : public mlir::PassWrapper<AllocateAddressPass,
                                 mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    auto currentAddress = 0;

    module->walk([&](mlir::memref::GlobalOp op) {
      auto address = currentAddress;
      auto value = dyn_cast<mx::MXBlockElementsAttr>(op.getInitialValueAttr());
      op->setAttr(
          "address", mlir::IntegerAttr::get(
                         mlir::IntegerType::get(module->getContext(), 64), address));
      auto size = value.getRawData().size();
      currentAddress += size;
      currentAddress = (currentAddress + 7) & ~7;
    });
  }

  llvm::StringRef getArgument() const override { return "nnay-allocate-address"; }
};

std::unique_ptr<mlir::Pass> createAllocateAddressPass() {
  return std::make_unique<AllocateAddressPass>();
}

} // namespace nnay
} // namespace onnx_mlir
