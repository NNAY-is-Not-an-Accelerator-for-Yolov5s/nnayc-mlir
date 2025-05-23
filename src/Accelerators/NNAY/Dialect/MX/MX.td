include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

def MX_Dialect : Dialect {
  let name = "mx";
  let cppNamespace = "::onnx_mlir::nnay::mx";
  let summary = "MX dialect";
  let description = "MX dialect";

  let useDefaultTypePrinterParser = 1;
}

def MXBlockType : TypeDef<MX_Dialect, "Block"> {
  let cppClassName = "MXBlockType";
  let cppType = "::onnx_mlir::nnay::mx::MXBlockType";
  let typeName = "mx.Block";
  let mnemonic = "mxblock";

  let summary = "MX BlockFP type";
  let description = "MX BlockFP type";

  let parameters = (ins
    "unsigned":$blockSize,
    "unsigned":$subblockSize,
    "unsigned":$sWidth, 
    "unsigned":$ssWidth,
    "unsigned":$mantissaWidth
  );

  let assemblyFormat = "`<` `blockSize` $blockSize `,` `subblockSize` $subblockSize `,` `sWidth` $sWidth `,` `ssWidth` $ssWidth `,` `mantissaWidth` $mantissaWidth `>`";
}

class MX_Op<string mnemonic, list<Trait> traits = []> :
    Op<MX_Dialect, mnemonic, traits> {
  let summary = "MX operation";
  let description = "MX operation";
}

def MXBlockElementsTensorAttr : AttrDef<MX_Dialect, "BlockElementsTensorAttr"> {
  let mnemonic = "mx.block_elements_tensor";
  let cppClassName = "MXBlockElementsTensorAttr";
  // let parameters = (ins
  //   "::mlir::Type":$elementType,
  //   "::mlir::ArrayRef<int64_t>":$shape,
  //   "::mlir::StringRef":$rawData
  // );
  // let assemblyFormat = "`<` $elementType `,` $shape `,` $rawData `>`";
}

def MX_ConstantOp : MX_Op<"Constant", [ConstantLike, Pure]> {
  let arguments = (ins
    MXBlockElementsTensorAttr:$value
  );

  let results = (outs
    TensorOf<[MXBlockType]>:$output
  );

  let assemblyFormat = [{
    attr-dict `:` type($output)
  }];
}

def MX_PackInputTensorOp : MX_Op<"PackInputTensor", [Pure]> {
  let arguments = (ins
    TensorOf<[AnyType]>:$input
  );

  let results = (outs
    AnyType:$output
  );
}
