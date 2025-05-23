#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.hpp"

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.cpp.inc"