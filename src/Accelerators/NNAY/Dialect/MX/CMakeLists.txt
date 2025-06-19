add_onnx_mlir_dialect(MX mx)
add_onnx_mlir_dialect_doc(MX MX.td)

add_onnx_mlir_library(OMMXDialect
  MXDialect.cpp
  MXOps.cpp

  DEPENDS
  OMMXIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  OMNNAYCompilerOptions
  
  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNAY_INCLUDE_PATH}
)
