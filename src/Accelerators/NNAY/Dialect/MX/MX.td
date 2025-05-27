include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"

include "mlir/Dialect/Index/IR/IndexDialect.td"
include "mlir/Dialect/Index/IR/IndexEnums.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
    bool isMX9(MXBlockType type) {
      return type.getBlockSize() == 16 && type.getSubblockSize() == 2 &&
             type.getSWidth() == 8 && type.getSsWidth() == 1 &&
             type.getMantissaWidth() == 7;
    }
  }];

  let assemblyFormat = "`<` `b` $blockSize `,` `sb` $subblockSize `,` `s` $sWidth `,` `ss` $ssWidth `,` `m` $mantissaWidth `>`";
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

  let builders = [
    OpBuilder<(ins "mlir::Attribute":$value, "mlir::Type":$type), [{
      build($_builder, $_state, type, value);
    }]>
  ];

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
