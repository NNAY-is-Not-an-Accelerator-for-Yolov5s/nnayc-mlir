#ifndef NNAY_HL_OPS_HPP
#define NNAY_HL_OPS_HPP

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"


#define GET_OP_CLASSES
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp.inc"

#endif // NNAY_HL_OPS_HPP
