add_onnx_mlir_library(OMNNAYHLToNNAYLL
  NNAYHLToNNAYLL.cpp

  LINK_LIBS PUBLIC
  OMNNAYHLDialect
  OMNNAYLLDialect
  OMMXDialect
  )
