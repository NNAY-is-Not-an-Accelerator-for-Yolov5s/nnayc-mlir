#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLDialect.hpp"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"

#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLDialect.cpp.inc"

void onnx_mlir::nnay::nnayhl::NNAYHLDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLAttributes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.cpp.inc"
      >();
}
