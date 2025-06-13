add_onnx_mlir_dialect(NNAYLL nnayll)
add_onnx_mlir_dialect_doc(NNAYLL NNAYLL.td)

add_onnx_mlir_library(OMNNAYLLDialect
  NNAYLL.cpp

  DEPENDS
  OMNNAYLLIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  OMMlirDialects
  MLIRSideEffectInterfaces
)