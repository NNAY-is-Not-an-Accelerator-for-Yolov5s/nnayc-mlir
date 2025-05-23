#include "src/Accelerators/NNAY/Compiler/NNAYCompilerOptions.hpp"

#define DEBUG_TYPE "NNAYCompilerOptions"

namespace onnx_mlir {

llvm::cl::opt<NNAYEmissionTargetType> nnayEmissionTarget(
    llvm::cl::desc("[Optional] Choose NNAY-related target to emit "
                   "(once selected it will cancel the other targets):"),
    llvm::cl::values(
        clEnumVal(EmitNNAYHLIR, "Lower model to NNAYHLIR (NNAYHL dialect)")),
    llvm::cl::init(EmitNNAYHLIR), llvm::cl::cat(OnnxMlirOptions));

} // namespace onnx_mlir
