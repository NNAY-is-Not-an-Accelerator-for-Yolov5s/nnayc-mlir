#include "src/Accelerators/NNAY/Dialect/MX/MXDialect.hpp"
#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.hpp"
#include "src/Accelerators/NNAY/Dialect/MX/MXOps.hpp"
#include "src/Accelerators/NNAY/Dialect/MX/MXTypes.hpp"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace onnx_mlir::nnay::mx;

#define GET_TYPEDEF_CLASSES
#include "src/Accelerators/NNAY/Dialect/MX/MXTypes.cpp.inc"

#include "src/Accelerators/NNAY/Dialect/MX/MXDialect.cpp.inc"

void onnx_mlir::nnay::mx::MXDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/Accelerators/NNAY/Dialect/MX/MXTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "src/Accelerators/NNAY/Dialect/MX/MXOps.cpp.inc"
      >();
}
