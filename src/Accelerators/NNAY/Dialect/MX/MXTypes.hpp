#ifndef NNAY_MX_TYPES_HPP
#define NNAY_MX_TYPES_HPP

#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <vector>

#define GET_TYPEDEF_CLASSES
#include "src/Accelerators/NNAY/Dialect/MX/MXTypes.hpp.inc"

#include "MXDefines.hpp"

namespace onnx_mlir::nnay::mx {

inline llvm::SmallVector<char> getMXBlockData(llvm::ArrayRef<float> values) {
  auto result = llvm::SmallVector<char, 18>();
  std::vector<float> fp32_inputs(values.begin(), values.end());
  auto mx_values = quantize_fp32_to_mx_vector<8, 7>(fp32_inputs);
  result.push_back(static_cast<char>(mx_values[0].get_s_field()));
  char ss = 0;
  for (int i = 0; i != 8; i++) {
    ss |= (mx_values[2 * i].get_ss() ? 1 : 0) << i;
  }
  result.push_back(ss);
  for (int i = 0; i != 16; i++) {
    char m = mx_values[i].get_m_unsigned_value();
    m |= (mx_values[i].get_signbit() ? 1 : 0) << 7;
    result.push_back(m);
  }
  return result;
}

} // namespace onnx_mlir::nnay::mx

#endif // NNAY_MX_TYPES_HPP
