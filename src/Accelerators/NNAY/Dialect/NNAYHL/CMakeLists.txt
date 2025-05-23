add_onnx_mlir_dialect(NNAYHL nnayhl)
add_onnx_mlir_dialect_doc(NNAYHL NNAYHL.td)

add_onnx_mlir_library(OMNNAYHLDialect
  NNAYHLDialect.cpp
  NNAYHLOps.cpp

  DEPENDS
  OMNNAYHLIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  OMMlirDialects
  MLIRSideEffectInterfaces
)