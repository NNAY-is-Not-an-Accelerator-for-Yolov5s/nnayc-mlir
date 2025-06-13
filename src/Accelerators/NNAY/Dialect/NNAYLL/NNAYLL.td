include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"

def NNAYLL_Dialect : Dialect {
  let name = "nnayll";
  let cppNamespace = "::onnx_mlir::nnay::nnayll";
  let summary = "NNAYLL dialect";
  let description = "NNAYLL dialect";
}

class NNAYLL_Op<string mnemonic, list<Trait> traits = []> :
    Op<NNAYLL_Dialect, mnemonic, traits> {
  let summary = "NNAYLL operation";
  let description = "NNAYLL operation";

  let arguments = (ins OptionalAttr<Builtin_StringAttr>:$onnx_node_name);
}

def ConvAct : NNAYLL_Op<"ConvAct", [Pure]> {
  let summary = "convolution operation";
  let description = "convolution operation";

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    AnyType:$weights,
    AnyType:$bias,
    OptionalAttr<I64ArrayAttr>:$strides,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<Builtin_StringAttr>:$activation
  );

  let results = (outs AnyType:$output);
}