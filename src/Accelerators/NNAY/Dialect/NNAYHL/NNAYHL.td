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

def ConvAct : NNAYHL_Op<"ConvAct", [Pure]> {
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

  // let results = (outs AnyType:$output);
  let results = (outs Variadic<AnyType>:$outputs);
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

def Pool : NNAYHL_Op<"Pool", [Pure]> {
  let summary = "pool operation";
  let description = "pool operation";

  let arguments = (ins
    AnyType:$input,
    OptionalAttr<I64ArrayAttr>:$kernel_size,
    OptionalAttr<I64ArrayAttr>:$stride,
    OptionalAttr<I64ArrayAttr>:$padding,
    OptionalAttr<Builtin_StringAttr>:$pooling_type,
    OptionalAttr<Builtin_StringAttr>:$ceil_mode
  );

  let results = (outs AnyType:$output);
}

def Mul : NNAYHL_Op<"Mul", [Pure]> {
  let summary = "mul operation";
  let description = "mul operation";

  let arguments = (ins AnyType:$input, AnyType:$other);
  let results = (outs AnyType:$output);
}

def Upsample : NNAYHL_Op<"Upsample", [Pure]> {
  let summary = "upsample operation";
  let description = "upsample operation";

  let arguments = (ins
    AnyType:$input,
    OptionalAttr<I64ArrayAttr>:$scale,
    OptionalAttr<Builtin_StringAttr>:$mode
  );

  let results = (outs AnyType:$output);
}