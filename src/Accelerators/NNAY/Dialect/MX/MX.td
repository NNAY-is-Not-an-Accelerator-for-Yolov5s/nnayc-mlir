include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/AttrTypeBase.td"

include "mlir/Dialect/Index/IR/IndexDialect.td"
include "mlir/Dialect/Index/IR/IndexEnums.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def MX_Dialect : Dialect {
  let name = "mx";
  let cppNamespace = "::onnx_mlir::nnay::mx";
  let summary = "MX dialect";
  let description = "MX dialect";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 0;

  let extraClassDeclaration = [{
    void printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const override;
    // mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser, mlir::Type type) const override;
  }];
}

class MX_Attr<string name, string attrMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute">
    : AttrDef<MX_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
  let attrName = "mx." # attrMnemonic;
}

def MXBlockElementsAttr : MX_Attr<"MXBlockElements", "mx_block_elements", [ElementsAttrInterface], "::mlir::DenseElementsAttr"> {
  let summary = "MX BlockElements attribute";
  let description = "MX BlockElements attribute";

  let parameters = (ins
    AttributeSelfTypeParameter<"", "ShapedType">:$type,
    "ArrayRef<char>":$rawData
  );

  let extraClassDeclaration = [{
    using ::mlir::DenseElementsAttr::empty;
    using ::mlir::DenseElementsAttr::getNumElements;
    using ::mlir::DenseElementsAttr::getElementType;
    using ::mlir::DenseElementsAttr::getValues;
    using ::mlir::DenseElementsAttr::isSplat;
    using ::mlir::DenseElementsAttr::size;
    using ::mlir::DenseElementsAttr::value_begin;

    using ContiguousIterableTypesT = std::tuple<
      // Integer types.
      uint8_t, uint16_t, uint32_t, uint64_t,
      int8_t, int16_t, int32_t, int64_t,
      short, unsigned short, int, unsigned, long, unsigned long,
      std::complex<uint8_t>, std::complex<uint16_t>, std::complex<uint32_t>,
      std::complex<uint64_t>,
      std::complex<int8_t>, std::complex<int16_t>, std::complex<int32_t>,
      std::complex<int64_t>,
      // Float types.
      float, double, std::complex<float>, std::complex<double>
    >;
    using NonContiguousIterableTypesT = std::tuple<
      ::mlir::Attribute,
      // Integer types.
      ::mlir::APInt, bool, std::complex<::mlir::APInt>
    >;

    static MXBlockElementsAttr get(::mlir::ShapedType type, ::llvm::ArrayRef<char> data);
    template <typename T>
    auto try_value_begin_impl(OverloadToken<T>) const {
      return try_value_begin<T>();
    }

    void print(mlir::AsmPrinter &printer) const {
      printer << "mx_block_elements";
      printer << "<";
      printer << getType();
      printer << ",";
      printer << getRawData();
      printer << ">";
    }

  public:
  }];

  let genAccessors = 0;
  let genStorageClass = 0;
  let skipDefaultBuilders = 1;
}

def MXBlockType : TypeDef<MX_Dialect, "Block", [
  MemRefElementTypeInterface
]> {
  let cppClassName = "MXBlockType";
  let cppType = "::onnx_mlir::nnay::mx::MXBlockType";
  let typeName = "mx.Block";
  let mnemonic = "block";

  let summary = "MX BlockFP type";
  let description = "MX BlockFP type";

  let parameters = (ins
    "unsigned":$blockSize,
    "unsigned":$subblockSize,
    "unsigned":$sWidth, 
    "unsigned":$ssWidth,
    "unsigned":$mantissaWidth
  );

  let extraClassDeclaration = [{
    static MXBlockType getMX9(mlir::MLIRContext *context) {
      return MXBlockType::get(context, 16, 2, 8, 1, 7);
    }
    bool isMX9() const {
      return getBlockSize() == 16 && getSubblockSize() == 2 &&
             getSWidth() == 8 && getSsWidth() == 1 &&
             getMantissaWidth() == 7;
    }
  }];

  let assemblyFormat = "`<`$blockSize`,`$subblockSize`,`$sWidth`,`$ssWidth`,`$mantissaWidth`>`";
}

class MX_Op<string mnemonic, list<Trait> traits = []> :
    Op<MX_Dialect, mnemonic, traits> {
  let summary = "MX operation";
  let description = "MX operation";
}

def MX_ConstantOp : MX_Op<"Constant", [Pure]> {  // Implement ConstantLike interface
  let summary = "A constant operation";
  let description = "This operation returns a constant value.";
  let arguments = (ins
    AnyAttr:$value
  );

  let results = (outs
    AnyType:$output
  );

  let assemblyFormat = "attr-dict`:` type($output)";
}

def MX_PackTensorOp : MX_Op<"PackTensor", [Pure]> {
  let arguments = (ins
    TensorOf<[AnyType]>:$input
  );

  let results = (outs
    AnyType:$output
  );
}

def MX_ConvertPackedToMXBlockOp : MX_Op<"ConvertPackedToMXBlock", [Pure]> {
  let arguments = (ins
    TensorOf<[AnyType]>:$input
  );

  let results = (outs
    TensorOf<[MXBlockType]>:$output
  );
}
