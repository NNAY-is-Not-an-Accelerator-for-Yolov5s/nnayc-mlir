add_onnx_mlir_library(OMNNAYHLTransform
  SplitFuse.cpp
  ConcatFuse.cpp
  FoldConvSplit.cpp
  OptimizeSigmoidLayout.cpp

  EXCLUDE_FROM_OM_LIBS
)