add_onnx_mlir_rewriter(ONNXToMX)

add_onnx_mlir_library(OMONNXToMX
  ONNXToMX.cpp
  ConversionPatterns/ConvertONNXTensorToMX.cpp

  DEPENDS
  OMONNXONNXToMXIncGen
  )

add_onnx_mlir_rewriter(RewriteONNXForMX)
