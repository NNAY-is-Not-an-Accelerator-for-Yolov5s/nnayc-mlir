#include <fstream>

#include "NNAYAsm.hpp"
#include "model_data_generated.h"

using namespace nnayasm;

namespace nnayasm {
struct ConvActInst : public impl::InstructionBase<ConvActInst> {
  auto emitInstruction(std::ostream &os) -> void override { os << "convact"; }

  auto emitInstructionAsm(std::ostream &os) -> void override {
    os << getName() << " ";
  }
};

void NNAYProgram::emitMemoryBin(std::string_view path) {
  auto flatBufferBuilder = flatbuffers::FlatBufferBuilder(4 * 1024 * 1024);
  std::vector<flatbuffers::Offset<NNAYModelData::TensorDataBlock>> tensorsVec;
  auto tensorId = 0;

  for (auto &[data, addr, name] : weightsBiasData) {
    auto tensorDataOffset = flatBufferBuilder.CreateVector(
        reinterpret_cast<const int8_t *>(data.data()), data.size());
    auto tensorName = flatBufferBuilder.CreateString(name);
    auto tensorBuilder =
        NNAYModelData::TensorDataBlockBuilder(flatBufferBuilder);
    tensorBuilder.add_tensor_id(tensorId++);
    tensorBuilder.add_name(tensorName);
    tensorBuilder.add_data_start_addr(addr);
    tensorBuilder.add_data_end_addr(addr + data.size());
    tensorBuilder.add_data(tensorDataOffset);
    auto tensorBlock = tensorBuilder.Finish();
    tensorsVec.push_back(tensorBlock);
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

  std::ofstream modelBundleFile(path.data(), std::ios::binary);
  modelBundleFile.write(
      reinterpret_cast<const char *>(modelBundleBuffer.data()),
      modelBundleSize);
}
} // namespace nnayasm
