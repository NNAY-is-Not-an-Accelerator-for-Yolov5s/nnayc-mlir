#include <cstdint>
#include <fstream>

#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "model_data_generated.h"
#include "llvm/ADT/StringRef.h"

#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.hpp"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace onnx_mlir::nnay {

struct ExportBinPass : public mlir::PassWrapper<ExportBinPass,
                           mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override { return "export-nnay-bin"; }
};

} // namespace onnx_mlir::nnay

void onnx_mlir::nnay::ExportBinPass::runOnOperation() {
  auto flatBufferBuilder = flatbuffers::FlatBufferBuilder(4 * 1024 * 1024);

  std::vector<flatbuffers::Offset<NNAYModelData::TensorDataBlock>> tensorsVec;

  auto tensorId = 0;
  for (auto &op : getOperation()) {
    if (auto memrefGlobalOp = dyn_cast<mlir::memref::GlobalOp>(op)) {
      if (auto tensorDataAttr = dyn_cast<mx::MXBlockElementsAttr>(
              memrefGlobalOp.getInitialValueAttr())) {
        auto tensorData = tensorDataAttr.getRawData();
        auto tensorDataOffset = flatBufferBuilder.CreateVector(
            reinterpret_cast<const int8_t *>(tensorData.data()),
            tensorData.size());
        auto tensorName =
            flatBufferBuilder.CreateString(memrefGlobalOp.getSymName().str());

        auto tensorBuilder =
            NNAYModelData::TensorDataBlockBuilder(flatBufferBuilder);
        tensorBuilder.add_tensor_id(tensorId++);
        tensorBuilder.add_name(tensorName);
        tensorBuilder.add_data_dst_addr(0x0);
        tensorBuilder.add_data(tensorDataOffset);
        auto tensorBlock = tensorBuilder.Finish();
        tensorsVec.push_back(tensorBlock);
      }
    }
  }

  auto tensorsOffset = flatBufferBuilder.CreateVector(tensorsVec);

  NNAYModelData::ModelBundleBuilder modelBundleBuilder(flatBufferBuilder);
  modelBundleBuilder.add_magic_number(0x4e4e4159);
  modelBundleBuilder.add_format_version(1);
  modelBundleBuilder.add_tensors(tensorsOffset);
  auto modelBundle = modelBundleBuilder.Finish();

  flatBufferBuilder.Finish(modelBundle);

  auto modelBundleBuffer = flatBufferBuilder.Release();
  auto modelBundleSize = modelBundleBuffer.size();

  std::ofstream modelBundleFile("model.hwmdl", std::ios::binary);
  modelBundleFile.write(
      reinterpret_cast<const char *>(modelBundleBuffer.data()),
      modelBundleSize);
}

namespace onnx_mlir::nnay {

std::unique_ptr<mlir::Pass> createExportBinPass() {
  return std::make_unique<ExportBinPass>();
}

} // namespace onnx_mlir::nnay