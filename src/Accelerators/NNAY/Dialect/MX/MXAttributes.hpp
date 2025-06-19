#ifndef MX_ATTRIBUTES_HPP
#define MX_ATTRIBUTES_HPP

#include "llvm-project/mlir/lib/IR/AttributeDetail.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include <cstdint>
#include <utility>

namespace onnx_mlir::nnay::mx {
namespace detail {
struct MXBlockElementsAttrStorage
    : public mlir::detail::DenseElementsAttributeStorage {
  MXBlockElementsAttrStorage(
      mlir::ShapedType ty, llvm::ArrayRef<char> data, bool isSplat = false)
      : mlir::detail::DenseElementsAttributeStorage(ty, isSplat), data(data) {}

  struct KeyTy {
    KeyTy(mlir::ShapedType type, llvm::ArrayRef<char> data,
        llvm::hash_code hashCode, bool isSplat = false)
        : type(type), data(data), hashCode(hashCode), isSplat(isSplat) {}

    mlir::ShapedType type;
    llvm::ArrayRef<char> data;
    llvm::hash_code hashCode;
    bool isSplat;
  };

  bool operator==(const KeyTy &key) const {
    return key.type == type && key.data == data;
  }

  static KeyTy getKey(
      mlir::ShapedType ty, llvm::ArrayRef<char> data, bool isKnownSplat) {
    // Handle an empty storage instance.
    if (data.empty())
      return KeyTy(ty, data, 0);

    // If the data is already known to be a splat, the key hash value is
    // directly the data buffer.
    bool isBoolData = ty.getElementType().isInteger(1);
    if (isKnownSplat) {
      if (isBoolData)
        return getKeyForSplatBoolData(ty, data[0] != 0);
      return KeyTy(ty, data, llvm::hash_value(data), isKnownSplat);
    }

    // Otherwise, we need to check if the data corresponds to a splat or not.

    // Handle the simple case of only one element.
    size_t numElements = ty.getNumElements();
    assert(numElements != 1 && "splat of 1 element should already be detected");

    // Handle boolean values directly as they are packed to 1-bit.
    if (isBoolData)
      return getKeyForBoolData(ty, data, numElements);

    size_t elementWidth =
        mlir::detail::getDenseElementBitWidth(ty.getElementType());
    // Non 1-bit dense elements are padded to 8-bits.
    size_t storageSize = llvm::divideCeil(elementWidth, CHAR_BIT);
    assert(((data.size() / storageSize) == numElements) &&
           "data does not hold expected number of elements");

    // Create the initial hash value with just the first element.
    auto firstElt = data.take_front(storageSize);
    auto hashVal = llvm::hash_value(firstElt);

    // Check to see if this storage represents a splat. If it doesn't then
    // combine the hash for the data starting with the first non splat element.
    for (size_t i = storageSize, e = data.size(); i != e; i += storageSize)
      if (memcmp(data.data(), &data[i], storageSize))
        return KeyTy(ty, data, llvm::hash_combine(hashVal, data.drop_front(i)));

    // Otherwise, this is a splat so just return the hash of the first element.
    return KeyTy(ty, firstElt, hashVal, /*isSplat=*/true);
  }

  /// Construct a key with a set of boolean data.
  static KeyTy getKeyForBoolData(
      mlir::ShapedType ty, llvm::ArrayRef<char> data, size_t numElements) {
    llvm::ArrayRef<char> splatData = data;
    bool splatValue = splatData.front() & 1;

    // Check the simple case where the data matches the known splat value.
    if (splatData ==
        llvm::ArrayRef<char>(splatValue ? kSplatTrue : kSplatFalse))
      return getKeyForSplatBoolData(ty, splatValue);

    // Handle the case where the potential splat value is 1 and the number of
    // elements is non 8-bit aligned.
    size_t numOddElements = numElements % CHAR_BIT;
    if (splatValue && numOddElements != 0) {
      // Check that all bits are set in the last value.
      char lastElt = splatData.back();
      if (lastElt != llvm::maskTrailingOnes<unsigned char>(numOddElements))
        return KeyTy(ty, data, llvm::hash_value(data));

      // If this is the only element, the data is known to be a splat.
      if (splatData.size() == 1)
        return getKeyForSplatBoolData(ty, splatValue);
      splatData = splatData.drop_back();
    }

    // Check that the data buffer corresponds to a splat of the proper mask.
    char mask = splatValue ? ~0 : 0;
    return llvm::all_of(splatData, [mask](char c) { return c == mask; })
               ? getKeyForSplatBoolData(ty, splatValue)
               : KeyTy(ty, data, llvm::hash_value(data));
  }

  /// Return a key to use for a boolean splat of the given value.
  static KeyTy getKeyForSplatBoolData(mlir::ShapedType type, bool splatValue) {
    const char &splatData = splatValue ? kSplatTrue : kSplatFalse;
    return KeyTy(type, splatData, llvm::hash_value(splatData),
        /*isSplat=*/true);
  }

  /// Hash the key for the storage.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.type, key.hashCode);
  }

  /// Construct a new storage instance.
  static MXBlockElementsAttrStorage *construct(
      mlir::AttributeStorageAllocator &allocator, KeyTy key) {
    // If the data buffer is non-empty, we copy it into the allocator with a
    // 64-bit alignment.
    llvm::ArrayRef<char> copy, data = key.data;
    if (!data.empty()) {
      char *rawData = reinterpret_cast<char *>(
          allocator.allocate(data.size(), alignof(uint64_t)));
      std::memcpy(rawData, data.data(), data.size());
      copy = llvm::ArrayRef<char>(rawData, data.size());
    }

    return new (allocator.allocate<MXBlockElementsAttrStorage>())
        MXBlockElementsAttrStorage(key.type, copy, key.isSplat);
  }

  llvm::ArrayRef<char> data;

  /// The values used to denote a boolean splat value.
  // This is not using constexpr declaration due to compilation failure
  // encountered with MSVC where it would inline these values, which makes it
  // unsafe to refer by reference in KeyTy.
  static const char kSplatTrue;
  static const char kSplatFalse;
};
} // namespace detail
} // namespace onnx_mlir::nnay::mx

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/NNAY/Dialect/MX/MXAttributes.hpp.inc"

#endif // MX_ATTRIBUTES_HPP
