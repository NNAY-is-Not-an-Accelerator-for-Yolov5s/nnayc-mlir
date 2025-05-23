#ifndef MX_OPS_HPP
#define MX_OPS_HPP

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "src/Accelerators/NNAY/Dialect/MX/MXTypes.hpp"
#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.hpp"

#define GET_OP_CLASSES
#include "src/Accelerators/NNAY/Dialect/MX/MXOps.hpp.inc"

#endif // MX_OPS_HPP
