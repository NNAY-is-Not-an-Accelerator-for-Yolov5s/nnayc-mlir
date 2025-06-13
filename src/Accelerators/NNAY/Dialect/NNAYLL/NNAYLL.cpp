#include "NNAYLL.hpp"

#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLLDialect.cpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLLOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLLTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLLAttributes.cpp.inc"

void onnx_mlir::nnay::nnayll::NNAYLLDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLLAttributes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLLOps.cpp.inc"
      >();
}