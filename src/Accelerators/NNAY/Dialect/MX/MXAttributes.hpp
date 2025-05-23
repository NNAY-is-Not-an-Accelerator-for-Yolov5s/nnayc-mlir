#ifndef MX_ATTRIBUTES_HPP
#define MX_ATTRIBUTES_HPP

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.hpp.inc"

#endif // MX_ATTRIBUTES_HPP
