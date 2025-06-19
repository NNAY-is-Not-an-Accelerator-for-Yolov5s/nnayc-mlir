find_package(flatbuffers CONFIG REQUIRED)

if (DEFINED ENV{CONDA_PREFIX})
    set(CONDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
    if (EXISTS ${CONDA_INCLUDE_DIR} AND IS_DIRECTORY ${CONDA_INCLUDE_DIR})
        message(STATUS "Adding Conda environment include directory: ${CONDA_INCLUDE_DIR}")
        include_directories(${CONDA_INCLUDE_DIR})
    else()
        message(WARNING "Conda environment include directory not found: ${CONDA_INCLUDE_DIR}")
        message(WARNING "Please ensure your Conda environment is activated before configuring CMake.")
    endif()
else()
    message(WARNING "CONDA_PREFIX environment variable not set. Conda include directory might be missing.")
    message(WARNING "Please activate your Conda environment before configuring CMake.")
endif()

flatbuffers_generate_headers(
    TARGET ModelFlatbuffersHeaders
    SCHEMAS ${CMAKE_CURRENT_SOURCE_DIR}/schemas/model_data.fbs
    BINARY_SCHEMAS_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_fbs
    FLAGS --gen-object-api
)

add_onnx_mlir_library(OMNNAYPassesEmitAsm
    EmitAsm.cpp
    ExportBin.cpp

    DEPENDS
    ModelFlatbuffersHeaders

    LINK_LIBS PUBLIC
    ModelFlatbuffersHeaders
)