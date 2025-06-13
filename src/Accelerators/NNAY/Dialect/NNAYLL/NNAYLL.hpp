#ifndef ONNX_MLIR_NNAYLL_H
#define ONNX_MLIR_NNAYLL_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLLDialect.hpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLLOps.hpp.inc"

#define GET_TYPEDEF_CLASSES
#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLLTypes.hpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/NNAY/Dialect/NNAYLL/NNAYLLAttributes.hpp.inc"

#endif // ONNX_MLIR_NNAYLL_H