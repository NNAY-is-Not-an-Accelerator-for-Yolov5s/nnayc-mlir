#ifndef ONNX_MLIR_NNAY_CONVERSION_NNAYHLTONNAYLL_HPP
#define ONNX_MLIR_NNAY_CONVERSION_NNAYHLTONNAYLL_HPP

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "mlir/Pass/Pass.h"
#include "src/Accelerators/NNAY/Dialect/MX/MXDialect.hpp"
#include "llvm/ADT/StringRef.h"

namespace onnx_mlir::nnay {

struct GlobalizeMXConstantsPass
    : public mlir::PassWrapper<GlobalizeMXConstantsPass,
          mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GlobalizeMXConstantsPass)

  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<::onnx_mlir::nnay::mx::MXDialect,
        mlir::memref::MemRefDialect, mlir::func::FuncDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "globalize-mx-constants";
  }

  llvm::StringRef getDescription() const override {
    return "Globalize MX constants to memref.GlobalOp";
  }
};

std::unique_ptr<mlir::Pass> createGlobalizeMXConstantsPass();

} // namespace onnx_mlir::nnay

#endif // ONNX_MLIR_NNAY_CONVERSION_NNAYHLTONNAYLL_HPP