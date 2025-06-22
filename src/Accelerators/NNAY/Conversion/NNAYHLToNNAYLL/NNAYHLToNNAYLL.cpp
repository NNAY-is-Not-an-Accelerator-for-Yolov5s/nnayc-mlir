#include "src/Accelerators/NNAY/Conversion/NNAYHLToNNAYLL/NNAYHLToNNAYLL.hpp"

#include "Dialect/MX/MXAttributes.hpp"
#include "Dialect/MX/MXTypes.hpp"
#include "Dialect/NNAYLL/NNAYLL.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Accelerators/NNAY/Dialect/MX/MXOps.hpp"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include <deque>
#include <memory>

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace {
struct LowerConvActOpPattern : public mlir::OpRewritePattern<nnayhl::ConvAct> {
  using mlir::OpRewritePattern<nnayhl::ConvAct>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      nnayhl::ConvAct op, mlir::PatternRewriter &rewriter) const override {
    auto resultTypes = op.getResultTypes();
    llvm::SmallVector<Type> newResultTypes;
    for (auto resultType : resultTypes) {
      if (isa<RankedTensorType>(resultType)) {
        auto shape = cast<RankedTensorType>(resultType).getShape();
        if (shape.size() != 4)
          return failure();
        auto n = shape[0];
        auto c = shape[1];
        auto h = shape[2];
        auto w = shape[3];
        auto newShape = RankedTensorType::get(
            {n, h, c / 16, w}, mx::MXBlockType::getMX9(op->getContext()));
        newResultTypes.push_back(newShape);
      } else {
        newResultTypes.push_back(resultType);
      }
    }

    auto weightsConstantOp =
        op.getWeights().getDefiningOp<mlir::ONNXConstantOp>();
    if (!weightsConstantOp) {
      return failure();
    }
    auto biasConstantOp = op.getBias().getDefiningOp<mlir::ONNXConstantOp>();
    if (!biasConstantOp) {
      return failure();
    }

    auto weightsAttr =
        dyn_cast<DenseElementsAttr>(weightsConstantOp->getAttr("value"));
    auto biasAttr =
        dyn_cast<DenseElementsAttr>(biasConstantOp->getAttr("value"));

    auto weightsShape = weightsAttr.getType().getShape();
    auto biasShape = biasAttr.getType().getShape();

    auto outputChannels = weightsShape[0];
    auto inputChannels = weightsShape[1];
    auto height = weightsShape[2];
    auto width = weightsShape[3];

    if (outputChannels != biasShape[0]) {
      op->emitError() << "Output channels of weights and bias do not match";
      return failure();
    }

    auto weightsValues = weightsAttr.getValues<float>();
    auto biasValues = biasAttr.getValues<float>();

    auto ocGroupNum = (outputChannels + 15) / 16;
    auto lastGroupSize = outputChannels % 16;

    SmallVector<char> mxData;

    for (decltype(ocGroupNum) ocGroup = 0; ocGroup != ocGroupNum; ++ocGroup) {
      SmallVector<float, 16> biasBlockData;
      for (decltype(16) blockOffset = 0; blockOffset != 16; ++blockOffset) {
        if (lastGroupSize != 0 && ocGroup == ocGroupNum - 1 &&
            blockOffset >= lastGroupSize) {
          biasBlockData.push_back(0.0f);
        } else {
          auto oc = ocGroup * 16 + blockOffset;
          auto index = oc;
          biasBlockData.push_back(biasValues[index]);
        }
      }
      auto mxBiasBlockData = mx::getMXBlockData(biasBlockData);
      mxData.append(mxBiasBlockData.begin(), mxBiasBlockData.end());

      for (decltype(height) h = 0; h != height; ++h) {
        for (decltype(inputChannels) ic = 0; ic != inputChannels; ++ic) {
          for (decltype(width) w = 0; w != width; ++w) {
            SmallVector<float> blockData;
            for (decltype(16) blockOffset = 0; blockOffset != 16;
                ++blockOffset) {
              if (lastGroupSize != 0 && ocGroup == ocGroupNum - 1 &&
                  blockOffset >= lastGroupSize) {
                blockData.push_back(0.0f);
              } else {
                auto oc = ocGroup * 16 + blockOffset;
                auto index = oc * inputChannels * height * width +
                             ic * height * width + h * width + w;
                blockData.push_back(weightsValues[index]);
              }
            }
            auto mxBlockData = mx::getMXBlockData(blockData);
            mxData.append(mxBlockData.begin(), mxBlockData.end());
          }
        }
      }
    }

    auto newShape =
        RankedTensorType::get({ocGroupNum, height, inputChannels, width},
            mx::MXBlockType::getMX9(op->getContext()));

    auto newPackedAttr = mx::MXBlockElementsAttr::get(newShape, mxData);
    auto mxConstOp = rewriter.create<onnx_mlir::nnay::mx::ConstantOp>(
        op->getLoc(), newShape, newPackedAttr);

    SmallVector<Value> newConvOpOperands;
    for (auto input : op.getInputs()) {
      newConvOpOperands.push_back(input);
    }
    newConvOpOperands.push_back(mxConstOp.getResult());

    auto newConvOp = rewriter.create<nnayll::ConvAct>(
        op.getLoc(), newResultTypes, newConvOpOperands);
    auto strides =
        cast<mlir::IntegerAttr>(op.getStrides()->getValue().front()).getInt();
    newConvOp.setStrides(strides);
    auto pads =
        cast<mlir::IntegerAttr>(op.getPads()->getValue().front()).getInt();
    newConvOp.setPads(pads);
    newConvOp.setActivationAttr(op.getActivationAttr());
    newConvOp.setOnnxNodeNameAttr(
        cast<mlir::StringAttr>(op->getAttr("onnx_node_name")));

    rewriter.replaceOp(op, newConvOp.getOutputs());

    return success();
  }
};

class LowerAddOpPattern : public mlir::OpRewritePattern<nnayhl::Add> {
  using mlir::OpRewritePattern<nnayhl::Add>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      nnayhl::Add op, mlir::PatternRewriter &rewriter) const override {
    if (op->getOperands().size() != 2) {
      return failure();
    }

    auto operand2 = op.getOperand(1);
    if (auto onnxConstantOp = operand2.getDefiningOp<mlir::ONNXConstantOp>()) {
      auto constantAttr =
          dyn_cast<DenseElementsAttr>(onnxConstantOp->getAttr("value"));
      auto constantValues = constantAttr.getValues<float>();

      auto totalValueNums = constantAttr.getNumElements();
      if (totalValueNums % 16 != 0) {
        op.emitError() << "Add with constant operand has a last dimension that "
                          "is not a multiple of 16";
        return failure();
      }

      auto newShape = RankedTensorType::get(
          {totalValueNums / 16}, mx::MXBlockType::getMX9(op->getContext()));
      SmallVector<char> mxData;
      auto index = 0;
      while (true) {
        SmallVector<float, 16> blockData;
        for (decltype(16) blockOffset = 0; blockOffset != 16; ++blockOffset) {
          blockData.push_back(constantValues[index]);
        }
        auto mxBlockData = mx::getMXBlockData(blockData);
        mxData.append(mxBlockData.begin(), mxBlockData.end());
        index += 16;
        if (index >= totalValueNums) {
          break;
        }
      }

      auto newPackedAttr = mx::MXBlockElementsAttr::get(newShape, mxData);
      auto mxConstOp = rewriter.create<onnx_mlir::nnay::mx::ConstantOp>(
          op.getLoc(), newShape, newPackedAttr);

      SmallVector<Value> newAddOpOperands;
      newAddOpOperands.push_back(op.getOperand(0));
      newAddOpOperands.push_back(mxConstOp.getResult());
      auto newAddOp = rewriter.create<nnayll::TensorAdd>(
          op.getLoc(), op.getOperand(0).getType(), newAddOpOperands);
      newAddOp.setOnnxNodeNameAttr(
          cast<mlir::StringAttr>(op->getAttr("onnx_node_name")));

      rewriter.replaceOp(op, newAddOp.getResult());
      return success();
    }

    return failure();
  }
};

class LowerMulOpPattern : public mlir::OpRewritePattern<nnayhl::Mul> {
  using mlir::OpRewritePattern<nnayhl::Mul>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      nnayhl::Mul op, mlir::PatternRewriter &rewriter) const override {
    if (op->getOperands().size() != 2) {
      return failure();
    }

    auto operand2 = op.getOperand(1);
    if (auto onnxConstantOp = operand2.getDefiningOp<mlir::ONNXConstantOp>()) {
      auto constantAttr =
          dyn_cast<DenseElementsAttr>(onnxConstantOp->getAttr("value"));
      auto constantValues = constantAttr.getValues<float>();

      auto totalValueNums = constantAttr.getNumElements();
      if (totalValueNums == 1) {
        // Scalar constant
        auto value = constantValues[0];
        auto valueAttr = mlir::FloatAttr::get(
            mlir::BFloat16Type::get(op.getContext()), value);
        auto newMulOp = rewriter.create<nnayll::TensorScalarMul>(
            op.getLoc(), op.getOperand(0).getType(), op.getOperand(0));
        newMulOp.setScalarAttr(valueAttr);
        newMulOp.setOnnxNodeNameAttr(
            cast<mlir::StringAttr>(op->getAttr("onnx_node_name")));
        rewriter.replaceOp(op, newMulOp.getResult());
        return success();
      } else {
        if (totalValueNums % 16 != 0) {
          op.emitError()
              << "Mul with constant operand has a last dimension that "
                 "is not a multiple of 16";
          return failure();
        }

        auto newShape = RankedTensorType::get(
            {totalValueNums / 16}, mx::MXBlockType::getMX9(op->getContext()));

        SmallVector<char> mxData;
        auto index = 0;
        while (true) {
          SmallVector<float, 16> blockData;
          for (decltype(16) blockOffset = 0; blockOffset != 16; ++blockOffset) {
            blockData.push_back(constantValues[index]);
          }
          auto mxBlockData = mx::getMXBlockData(blockData);
          mxData.append(mxBlockData.begin(), mxBlockData.end());
          index += 16;
          if (index >= totalValueNums) {
            break;
          }
        }

        auto newPackedAttr = mx::MXBlockElementsAttr::get(newShape, mxData);
        auto mxConstOp = rewriter.create<onnx_mlir::nnay::mx::ConstantOp>(
            op.getLoc(), newShape, newPackedAttr);

        SmallVector<Value> newMulOpOperands;
        newMulOpOperands.push_back(op.getOperand(0));
        newMulOpOperands.push_back(mxConstOp.getResult());
        auto newMulOp = rewriter.create<nnayll::TensorEltMul>(
            op.getLoc(), op.getOperand(0).getType(), newMulOpOperands);
        newMulOp.setOnnxNodeNameAttr(
            cast<mlir::StringAttr>(op->getAttr("onnx_node_name")));

        rewriter.replaceOp(op, newMulOp.getResult());
      }
      return success();
    } else {
      op.emitWarning() << "Mul with non-constant operand is not supported";
    }

    return failure();
  }
};

class LowerSquareOpPattern : public mlir::OpRewritePattern<nnayhl::Square> {
  using mlir::OpRewritePattern<nnayhl::Square>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      nnayhl::Square op, mlir::PatternRewriter &rewriter) const override {
    auto newSquareOp = rewriter.create<nnayll::TensorSquare>(
        op.getLoc(), op.getType(), op.getOperand());
    newSquareOp.setOnnxNodeNameAttr(
        cast<mlir::StringAttr>(op->getAttr("onnx_node_name")));
    rewriter.replaceOp(op, newSquareOp.getResult());
    return success();
  }
};
} // namespace

namespace onnx_mlir::nnay {

class LowerConvActOpPass : public mlir::PassWrapper<LowerConvActOpPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<::onnx_mlir::nnay::mx::MXDialect,
        mlir::memref::MemRefDialect, mlir::func::FuncDialect>();
  }

  llvm::StringRef getArgument() const override { return "lower-conv-act-op"; }
};

class LowerToNNAYLLPass : public mlir::PassWrapper<LowerToNNAYLLPass,
                              mlir::OperationPass<mlir::ModuleOp>> {

  void runOnOperation() override {
    auto module = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerConvActOpPattern, LowerAddOpPattern, LowerMulOpPattern>(
        &getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<::onnx_mlir::nnay::mx::MXDialect,
        mlir::memref::MemRefDialect, mlir::func::FuncDialect>();
  }

  llvm::StringRef getArgument() const override { return "lower-to-nnayll"; }
};

void LowerConvActOpPass::runOnOperation() {
  auto func = getOperation();
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<LowerConvActOpPattern>(&getContext());
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

void GlobalizeMXConstantsPass::runOnOperation() {
  auto module = getOperation();

  std::deque<mx::ConstantOp> constantsToProcess;

  module->walk([&](mx::ConstantOp op) { constantsToProcess.push_back(op); });

  mlir::OpBuilder globalBuilder(module->getContext());
  globalBuilder.setInsertionPointToStart(module.getBody());
  mlir::OpBuilder replaceBuilder(module->getContext());
  module->walk([&](mlir::func::FuncOp funcOp) {
    replaceBuilder.setInsertionPointToStart(&funcOp.getBody().front());
  });

  for (mx::ConstantOp constantOp : constantsToProcess) {
    if (!constantOp.getOperation()) {
      continue;
    }

    mx::MXBlockElementsAttr attr =
        dyn_cast_or_null<mx::MXBlockElementsAttr>(constantOp.getValue());
    if (!attr) {
      constantOp->emitError() << "mx.Constant does not have a "
                                 "MXBlockElementsAttr value, skipping";
      continue;
    }

    SmallString<128> globalName;
    for (auto user : constantOp->getResult(0).getUsers()) {
      if (!globalName.empty()) {
        globalName += ";";
      }
      llvm::TypeSwitch<mlir::Operation *>(user)
          .Case<nnayll::ConvAct>([&](nnayll::ConvAct convActOp) {
            if (convActOp.getWeightsBias() == constantOp->getResult(0)) {
              globalName +=
                  cast<StringAttr>(convActOp->getAttr("onnx_node_name"))
                      .getValue();
              globalName += "_WeightsBias";
            }
          })
          .Case<nnayll::TensorAdd>([&](nnayll::TensorAdd tensorAddOp) {
            if (tensorAddOp.getOperand(1) == constantOp->getResult(0)) {
              globalName +=
                  cast<StringAttr>(tensorAddOp->getAttr("onnx_node_name"))
                      .getValue();
              globalName += "_Add";
            }
          })
          .Case<nnayll::TensorEltMul>([&](nnayll::TensorEltMul tensorEltMulOp) {
            if (tensorEltMulOp.getOperand(1) == constantOp->getResult(0)) {
              globalName +=
                  cast<StringAttr>(tensorEltMulOp->getAttr("onnx_node_name"))
                      .getValue();
              globalName += "_EltMul";
            }
          })
          .Default([](mlir::Operation *op) {
            op->emitError()
                << "MX constant is used by an unsupported operation";
            return;
          });
    }

    mlir::ShapedType tensorType = cast<mlir::ShapedType>(constantOp.getType());
    mlir::MemRefType globalMemRefType = mlir::MemRefType::get(
        tensorType.getShape(), tensorType.getElementType());

    mlir::memref::GlobalOp globalOp =
        globalBuilder.create<mlir::memref::GlobalOp>(module->getLoc(),
            globalName, globalBuilder.getStringAttr("private"),
            globalMemRefType, attr, false, nullptr);

    mlir::memref::GetGlobalOp getGlobalOp =
        replaceBuilder.create<mlir::memref::GetGlobalOp>(
            replaceBuilder.getUnknownLoc(), globalMemRefType,
            globalOp.getName());

    constantOp->replaceAllUsesWith(getGlobalOp->getResults());
    constantOp->erase();
  }
}

std::unique_ptr<mlir::Pass> createGlobalizeMXConstantsPass() {
  return std::make_unique<GlobalizeMXConstantsPass>();
}

std::unique_ptr<mlir::Pass> createLowerToNNAYLLPass() {
  return std::make_unique<LowerToNNAYLLPass>();
}
} // namespace onnx_mlir::nnay
