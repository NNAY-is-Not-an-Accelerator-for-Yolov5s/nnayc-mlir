#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "src/Compiler/CompilerDialects.hpp"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Compiler/DisposableGarbageCollector.hpp"
#include "src/Tools/nnayc-opt/RegisterPasses.hpp"
#include "src/Version/Version.hpp"

#include "src/Accelerators/NNAY/Dialect/MX/MXDialect.hpp"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLDialect.hpp"
#include "src/Accelerators/NNAY/Pass/NNAYPasses.hpp"

using namespace mlir;
using namespace onnx_mlir;

void scanAndSetMCPU(int argc, char **argv) {
  // Scan for (deprecated) --mcpu and add them to the mcpu option.
  for (int i = argc - 1; i > 0; --i) {
    std::string currStr(argv[i]);
    if (currStr.find("--mcpu=") == 0) {
      std::string cpuKind(&argv[i][7]); // Get the string starting 7 chars down.
      setTargetCPU(cpuKind);
      break;
    }
    if (currStr.find("-mcpu=") == 0) {
      std::string cpuKind(&argv[i][6]); // Get the string starting 6 chars down.
      setTargetCPU(cpuKind);
      break;
    }
  }
}

void scanAndSetMArch(int argc, char **argv) {
  // Scan --march and add them to the march option.
  for (int i = argc - 1; i > 0; --i) {
    std::string currStr(argv[i]);
    if (currStr.find("--march=") == 0) {
      std::string archKind(
          &argv[i][8]); // Get the string starting 8 chars down.
      setTargetArch(archKind);
      break;
    }
    if (currStr.find("-march=") == 0) {
      std::string archKind(
          &argv[i][7]); // Get the string starting 7 chars down.
      setTargetArch(archKind);
      break;
    }
  }
}

void scanAndSetMAccel(int argc, char **argv) {
  // Scan accelerators and add them to the maccel option.
  for (int i = argc - 1; i > 0; --i) {
    std::string currStr(argv[i]);
    if (currStr.find("--maccel=") == 0) {
      std::string accelKind(
          &argv[i][9]); // Get the string starting 9 chars down.
      setTargetAccel(accelKind);
      break;
    }
    if (currStr.find("-maccel=") == 0) {
      std::string accelKind(
          &argv[i][8]); // Get the string starting 8 chars down.
      setTargetAccel(accelKind);
      break;
    }
  }
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Scan Opt Level manually now as it is needed to register passes
  // before command line options are parsed.
  scanAndSetMCPU(argc, argv);
  scanAndSetMArch(argc, argv);
  scanAndSetMAccel(argc, argv);

  DialectRegistry registry = registerDialects(maccel);
  registry.insert<::onnx_mlir::nnay::nnayhl::NNAYHLDialect>();
  registry.insert<::onnx_mlir::nnay::mx::MXDialect>();

  bufferization::registerBufferizationPipelines();
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

  nnay::registerPasses();
  registerMLIRContextCLOptions();
  registerAsmPrinterCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();

  PassPipelineCLParser passPipeline("", "Compiler passes to run");

  if (!parseCustomEnvFlagsCommandLineOption(argc, argv, &llvm::errs()) ||
      !llvm::cl::ParseCommandLineOptions(argc, argv,
          getVendorName() + " - A modular optimizer driver\n", &llvm::errs(),
          customEnvFlags.c_str())) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  initCompilerConfig();

  std::string error_message;
  auto file = openInputFile(inputFilename, &error_message);
  if (!error_message.empty()) {
    llvm::errs() << "Failure to open file; " << error_message << "\n";
    return 1;
  }

  auto output = openOutputFile(outputBaseName, &error_message);
  if (!error_message.empty()) {
    llvm::errs() << "Failure to compile file; " << error_message << "\n";
    return 1;
  }

  configurePasses();
  for (auto *accel : accel::Accelerator::getAccelerators())
    accel->configurePasses();

  std::unique_ptr<llvm::ThreadPoolInterface> threadPoolPtr = nullptr;
  auto passManagerSetupFn = [&](PassManager &pm) {
    MLIRContext *ctx = pm.getContext();
    // Set number of threads in the MLIRContext
    if (compilationNumThreads > 0)
      ctx->disableMultithreading();
    if (compilationNumThreads > 1) {
      threadPoolPtr = std::make_unique<llvm::DefaultThreadPool>(
          llvm::hardware_concurrency(compilationNumThreads));
      ctx->setThreadPool(*threadPoolPtr);
    }

    // MlirOptMain constructed ctx with our registry so we just load all our
    // already registered dialects.
    ctx->loadAllAvailableDialects();
    pm.addInstrumentation(std::make_unique<DisposableGarbageCollector>(ctx));
    auto errorHandler = [ctx](const Twine &msg) {
      emitError(UnknownLoc::get(ctx)) << msg;
      return failure();
    };
    return passPipeline.addToPipeline(pm, errorHandler);
  };

  MlirOptMainConfig config;
  config.setPassPipelineSetupFn(passManagerSetupFn)
      .splitInputFile(split_input_file ? kDefaultSplitMarker : "")
      .verifyDiagnostics(verify_diagnostics)
      .verifyPasses(verify_passes)
      .allowUnregisteredDialects(allowUnregisteredDialects)
      .emitBytecode(false)
      .useExplicitModule(false);

  if (failed(MlirOptMain(output->os(), std::move(file), registry, config)))
    return 1;

  output->keep();
  return 0;
}