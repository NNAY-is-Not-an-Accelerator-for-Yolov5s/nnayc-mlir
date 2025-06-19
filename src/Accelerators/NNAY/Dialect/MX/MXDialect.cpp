#include "src/Accelerators/NNAY/Dialect/MX/MXDialect.hpp"
#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.hpp"
#include "src/Accelerators/NNAY/Dialect/MX/MXOps.hpp"
#include "src/Accelerators/NNAY/Dialect/MX/MXTypes.hpp"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

using namespace mlir;
using namespace onnx_mlir::nnay::mx;

#define GET_TYPEDEF_CLASSES
#include "src/Accelerators/NNAY/Dialect/MX/MXTypes.cpp.inc"

#include "src/Accelerators/NNAY/Dialect/MX/MXDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.cpp.inc"

MXBlockElementsAttr MXBlockElementsAttr::get(
    ShapedType type, ArrayRef<char> data) {
  return Base::get(type.getContext(), type, data, llvm::hash_value(data));
}

const char onnx_mlir::nnay::mx::detail::MXBlockElementsAttrStorage::kSplatTrue =
    1;
const char
    onnx_mlir::nnay::mx::detail::MXBlockElementsAttrStorage::kSplatFalse = 0;

namespace {
template <typename Printer>
void printAsHex(Printer &printer, ArrayRef<char> data) {
  printer << "0x";
  for (char c : data) {
    printer << llvm::format("%02x", static_cast<unsigned char>(c));
  }
}
} // namespace

void MXDialect::printAttribute(
    Attribute attr, DialectAsmPrinter &printer) const {
  if (auto mxBlockElementsAttr = dyn_cast<MXBlockElementsAttr>(attr)) {
    printer << "mx_block_elements";
    printer << "<";
    printer << mxBlockElementsAttr.getType().getElementType();
    printer << ",";
    printer << "\"";
    printAsHex(printer, mxBlockElementsAttr.getRawData());
    printer << "\"";
    printer << ">";
  } else {
    printer << attr;
  }
}

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
