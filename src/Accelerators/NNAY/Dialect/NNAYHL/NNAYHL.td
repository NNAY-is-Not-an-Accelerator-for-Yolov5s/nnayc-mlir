include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"

def NNAYHL_Dialect : Dialect {
  let name = "nnayhl";
  let cppNamespace = "::onnx_mlir::nnay::nnayhl";
  let summary = "NNAYHL dialect";
  let description = "NNAYHL dialect";
}

class NNAYHL_Op<string mnemonic, list<Trait> traits = []> :
    Op<NNAYHL_Dialect, mnemonic, traits> {
  let summary = "NNAYHL operation";
  let description = "NNAYHL operation";

  let arguments = (ins OptionalAttr<Builtin_StringAttr>:$onnx_node_name);
}

def Add : NNAYHL_Op<"Add", [MemoryEffects<[MemWrite]>]> {
  let summary = "add operation";
  let description = "add operation";

  let arguments = (ins AnyType:$input, AnyType:$other);
  let results = (outs AnyType:$output);
}

def SiLU : NNAYHL_Op<"SiLU", [Pure]> {
  let summary = "silu operation";
  let description = "silu operation";

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
}

def Conv : NNAYHL_Op<"Conv", [Pure]> {
  let summary = "convolution operation";
  let description = "convolution operation";

  let arguments = (ins
    AnyType:$input,
    AnyType:$weights,
    AnyType:$bias,
    OptionalAttr<I64ArrayAttr>:$strides,
    OptionalAttr<I64ArrayAttr>:$pads
  );

  let results = (outs AnyType:$output);
}

def Concat : NNAYHL_Op<"Concat", [Pure]> {
  let summary = "concat operation";
  let description = "concat operation";

  let arguments = (ins 
    Variadic<TensorOf<[AnyType]>>:$inputs,
    SI64Attr:$axis
  );

  let results = (outs TensorOf<[AnyType]>:$output);
}

def Split : NNAYHL_Op<"Split", [Pure]> {
  let summary = "split operation";
  let description = "split operation";

  let arguments = (ins 
    TensorOf<[AnyType]>:$input,
    SI64Attr:$axis,
    I64ArrayAttr:$split_sizes
  );

  let results = (outs Variadic<TensorOf<[AnyType]>>:$outputs);
}