get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

add_onnx_mlir_executable(nnayc-opt
  nnayc-opt.cpp
  RegisterPasses.cpp
  LINK_LIBS PRIVATE
  ${OMLibs}
  OMCompilerDialects
  OMCompilerOptions
  OMCompilerPasses
  OMAccelerator
  OMVersion
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIROpenMPToLLVM
  MLIROptLib
  MLIRSCFToOpenMP
  )