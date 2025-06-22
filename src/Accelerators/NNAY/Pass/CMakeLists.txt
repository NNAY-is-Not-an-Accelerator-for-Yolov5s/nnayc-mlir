add_subdirectory(EmitAsm)

add_onnx_mlir_library(OMNNAYPasses
    NNAYPasses.cpp
    AllocateAddress.cpp

    DEPENDS
    OMONNXToNNAYHL
)
