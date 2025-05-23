#include "src/Accelerators/NNAY/NNAYAccelerator.hpp"
#include "Compiler/NNAYCompilerOptions.hpp"
#include "Conversion/ONNXToMX/ONNXToMX.hpp"
#include "Pass/NNAYPasses.hpp"
#include "mlir/Pass/PassRegistry.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Dialect/Mlir/VectorMachineSupport.hpp"
#include "src/Pass/Passes.hpp"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Accelerators/NNAY/Dialect/MX/MXDialect.hpp"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLDialect.hpp"

#define DEBUG_TYPE "nnay-accelerator"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace accel {

Accelerator *createNNAY() { return NNAYAccelerator::getInstance(); }

NNAYAccelerator *NNAYAccelerator::instance = nullptr;

NNAYAccelerator::NNAYAccelerator() : Accelerator(Accelerator::Kind::NNAY) {
  LLVM_DEBUG(llvm::dbgs() << "Creating an NNAY accelerator\n");

  // Add this accelerator to the list of available accelerators
  acceleratorTargets.push_back(this);

  // Add compiler dependencies if any
  // addCompilerConfig(CCM_SHARED_LIB_DEPS, {"RuntimeNNAY"}, true);
}

NNAYAccelerator::~NNAYAccelerator() {
  // No need to delete instance here
}

uint64_t NNAYAccelerator::getVersionNumber() const {
  // Return version number of the accelerator
  return 1;
}

void NNAYAccelerator::addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
    std::string outputNameNoExt) const {
  LLVM_DEBUG(llvm::dbgs() << "Adding passes for NNAY accelerator\n");
  // Here we add NNAY-specific passes
  if (nnayEmissionTarget >= EmitNNAYHLIR) {
    llvm::outs() << "Adding NNAYHLIR passes\n";
    // pm.addPass(onnx_mlir::nnay::mx::createONNXToMXPass());
    VectorMachineSupport::setGlobalVectorMachineSupport(march, mcpu, "");
    configureConstPropONNXToONNXPass(onnxConstPropRoundFPToInt,
        onnxConstPropExpansionBound, onnxConstPropDisablePatterns,
        disableConstantProp);
    configureOnnxToKrnlLoweringPass(optReport == OptReport::Parallel,
        enableParallel, parallelizeOps, optReport == OptReport::Simd,
        !disableSimdOption);
    addONNXToMLIRPasses(pm, /*target CPU*/ maccel.empty(),
        /*donotScrubDisposableElementsAttr*/ true);
    pm.addPass(onnx_mlir::nnay::createONNXToNNAYHLPass());
    pm.addPass(onnx_mlir::nnay::mx::createONNXToMXPass());
    emissionTarget = EmitMLIR;
  }
}

void NNAYAccelerator::registerDialects(mlir::DialectRegistry &registry) const {
  LLVM_DEBUG(llvm::dbgs() << "Registering dialects for NNAY accelerator\n");
  registry.insert<::onnx_mlir::nnay::mx::MXDialect>();
  registry.insert<::onnx_mlir::nnay::nnayhl::NNAYHLDialect>();
}

void NNAYAccelerator::registerPasses(int optLevel) const {
  LLVM_DEBUG(llvm::dbgs() << "Registering passes for NNAY accelerator\n");
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::nnay::mx::createONNXToMXPass();
  });
}

void NNAYAccelerator::configurePasses() const {
  LLVM_DEBUG(llvm::dbgs() << "Configuring passes for NNAY accelerator\n");
  // Configure NNAY passes
  // TODO: Implement pass configuration if needed
}

mlir::MemRefType NNAYAccelerator::convertTensorTypeToMemRefType(
    const mlir::TensorType tensorType) const {
  assert(tensorType.hasRank() && "expected only ranked shapes");
  // TODO: Implement NNAY specific tensor to memref type conversion
  return mlir::MemRefType::get(
      tensorType.getShape(), tensorType.getElementType());
}

int64_t NNAYAccelerator::getDefaultAllocAlignment(
    const mlir::TensorType tensorType) const {
  assert(tensorType.hasRank() && "expected only ranked shapes");
  // TODO: Implement NNAY specific alignment if needed
  return 64; // Default alignment for NNAY
}

void NNAYAccelerator::conversionTargetONNXToKrnl(
    mlir::ConversionTarget &target) const {}

void NNAYAccelerator::rewritePatternONNXToKrnl(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
    mlir::MLIRContext *ctx) const {}

void NNAYAccelerator::conversionTargetKrnlToLLVM(
    mlir::ConversionTarget &target) const {
  // TODO: Implement NNAY specific conversion target for LLVM
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp>();
}

void NNAYAccelerator::rewritePatternKrnlToLLVM(
    mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter,
    mlir::MLIRContext *ctx) const {
  // TODO: Implement NNAY specific rewrite patterns for LLVM
  // Example: populateNNAYToLLVMConversionPattern(patterns, typeConverter, ctx);
}

} // namespace accel
} // namespace onnx_mlir
