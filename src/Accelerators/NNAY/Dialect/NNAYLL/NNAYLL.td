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
    AnyType:$weights_bias,
    OptionalAttr<I64Attr>:$strides,
    OptionalAttr<I64Attr>:$pads,
    OptionalAttr<Builtin_StringAttr>:$activation,
    OptionalAttr<Builtin_StringAttr>:$onnx_node_name
  );

  let results = (outs Variadic<AnyType>:$outputs);
}

def TensorScalarMul : NNAYLL_Op<"TensorScalarMul", [Pure]> {
  let summary = "scalar multiplication operation";
  let description = "scalar multiplication operation";

  let arguments = (ins
    AnyType:$input,
    AnyAttr:$scalar,
    OptionalAttr<Builtin_StringAttr>:$onnx_node_name
  );

  let results = (outs AnyType:$output);
}

def TensorAdd : NNAYLL_Op<"TensorAdd", [Pure]> {
  let summary = "tensor addition operation";
  let description = "tensor addition operation";

  let arguments = (ins
    AnyType:$input1,
    AnyType:$input2,
    OptionalAttr<Builtin_StringAttr>:$onnx_node_name
  );

  let results = (outs AnyType:$output);
}

def TensorEltMul : NNAYLL_Op<"TensorEltMul", [Pure]> {
  let summary = "tensor element-wise multiplication operation";
  let description = "tensor element-wise multiplication operation";

  let arguments = (ins
    AnyType:$input1,
    AnyType:$input2,
    OptionalAttr<Builtin_StringAttr>:$onnx_node_name
  );

  let results = (outs AnyType:$output);
}

def TensorSquare : NNAYLL_Op<"TensorSquare", [Pure]> {
  let summary = "tensor square operation";
  let description = "tensor square operation";

  let arguments = (ins
    AnyType:$input,
    OptionalAttr<Builtin_StringAttr>:$onnx_node_name
  );

  let results = (outs AnyType:$output);
}
