add_mlir_library(NNAYSimulator
  Simulator.cpp
  SimulatePass.cpp

  LINK_LIBS PUBLIC
  OMNNAYHLDialect
  OMMXDialect
  MLIRFuncDialect
)

target_include_directories(NNAYSimulator
  PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
)