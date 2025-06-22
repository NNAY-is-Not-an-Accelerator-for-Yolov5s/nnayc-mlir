#ifndef NNAY_ASM_HPP
#define NNAY_ASM_HPP

#include "mlir/IR/Value.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string_view>
#include <utility>

namespace nnayasm {

using OpcodeStoreType = uint8_t;
using FunctStoreType = uint8_t;
using BlockStoreType = uint64_t;

struct ConvActInst;

template <typename InstT>
struct InstTraits {
  static constexpr std::string_view name = "INVALID";
  static constexpr OpcodeStoreType opcode = 0xff;
  static constexpr FunctStoreType funct = 0xff;
  static constexpr size_t blockNum = 0;
};

#define DECLARE_INSTRUCTION_TRAITS(NAME, OPCODE, FUNCT, BLOCK_NUM)             \
  template <>                                                                  \
  struct InstTraits<NAME> {                                                    \
    static constexpr std::string_view name = #NAME;                            \
    static constexpr uint8_t opcode = OPCODE;                                  \
    static constexpr uint8_t funct = FUNCT;                                    \
    static constexpr size_t blockNum = BLOCK_NUM;                              \
  };
DECLARE_INSTRUCTION_TRAITS(ConvActInst, 0x00, 0x00, 1);

namespace impl {

struct InstructionI {
  virtual auto emitInstruction(std::ostream &os) -> void = 0;
  virtual auto emitInstructionAsm(std::ostream &os) -> void = 0;
  virtual ~InstructionI() = default;
};

template <typename Derived>
struct InstructionBase : public InstructionI {
  static constexpr auto getOpcode() -> OpcodeStoreType {
    static_assert(InstTraits<Derived>::opcode != 0xff,
        "Opcode is not set for instruction");
    return InstTraits<Derived>::opcode;
  }

  static constexpr auto getFunct() -> FunctStoreType {
    static_assert(
        InstTraits<Derived>::funct != 0xff, "Funct is not set for instruction");
    return InstTraits<Derived>::funct;
  }

  static constexpr auto getBlockNum() -> size_t {
    static_assert(InstTraits<Derived>::blockNum != 0,
        "Block number is not set for instruction");
    return InstTraits<Derived>::blockNum;
  }

  static constexpr auto getName() -> std::string_view {
    return InstTraits<Derived>::name;
  }

protected:
  void setBits(
      BlockStoreType &block, size_t start, size_t end, uint64_t value) {
    auto mask = (1 << (end - start + 1)) - 1;
    block |= (value & mask) << start;
  }

  auto getBlocks()
      -> std::array<BlockStoreType, InstTraits<Derived>::blockNum> & {
    return blocks;
  }

  auto getBlock(size_t index) -> BlockStoreType & { return blocks[index]; }

  std::array<BlockStoreType, InstTraits<Derived>::blockNum> blocks;
};

} // namespace impl

class NNAYProgram {
public:
  template <typename InstT, typename... Args>
  auto addInstruction(Args &&...args) -> InstT & {
    auto inst = std::make_unique<InstT>(std::forward<Args>(args)...);
    instructions.push_back(std::move(inst));
    return *inst;
  }

  auto emitProgram(std::ostream &os) -> void {
    for (auto &inst : instructions) {
      inst->emitInstructionAsm(os);
    }
  }

  auto emitMemoryBin(std::string_view path) -> void;

  auto appendWeightsBiasData(llvm::ArrayRef<char> data, std::string_view name, uint32_t addr) -> uint32_t {
    // auto addr = currentWeightsBiasTopAddr;
    weightsBiasData.push_back({data, addr, std::string(name)});
    // currentWeightsBiasTopAddr = (addr + data.size() + 7) & ~7;
    return addr;
  }

private:
  struct WeightsBiasDataInfo {
    llvm::ArrayRef<char> data;
    uint32_t addr;
    std::string name;
  };

private:
  std::vector<std::unique_ptr<impl::InstructionI>> instructions;
  std::vector<WeightsBiasDataInfo> weightsBiasData;
  uint32_t currentWeightsBiasTopAddr = 0;
};

} // namespace nnayasm

#endif // NNAY_ASM_HPP
