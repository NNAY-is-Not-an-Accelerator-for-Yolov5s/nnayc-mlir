#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Accelerators/NNAY/Dialect/NNAYHL/NNAYHLOps.hpp"

using namespace mlir;
using namespace onnx_mlir::nnay;

namespace {
struct FoldConvActivationPattern
    : public mlir::OpRewritePattern<nnayhl::ConvAct> {
  using mlir::OpRewritePattern<nnayhl::ConvAct>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(
      nnayhl::ConvAct op, mlir::PatternRewriter &rewriter) const override {
    if (op.getActivation().has_value() &&
        op.getActivation()->getValue() != "") {
      return mlir::failure();
    }

    if (op->getResults().size() != 1) {
      return mlir::failure();
    }

    // Check if Conv has exactly one user
    if (!op.getResult(0).hasOneUse())
      return mlir::failure();

    auto user = *op.getResult(0).getUsers().begin();
    if (auto actOp = dyn_cast<nnayhl::Act>(user)) {
      op.setActivationAttr(actOp.getActivation());
      actOp.getResult().replaceAllUsesWith(op.getResult(0));
      rewriter.eraseOp(actOp);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct FoldConvSplitPattern : public mlir::OpRewritePattern<nnayhl::ConvAct> {
  using mlir::OpRewritePattern<nnayhl::ConvAct>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(
      nnayhl::ConvAct op, mlir::PatternRewriter &rewriter) const override {
    if (op->getResults().size() != 1) {
      return mlir::failure();
    }

    // Check if Conv has exactly one user and it's a Split
    if (!op.getResult(0).hasOneUse())
      return mlir::failure();

    auto splitOp = dyn_cast<nnayhl::Split>(*op.getResult(0).getUsers().begin());
    if (!splitOp)
      return mlir::failure();

    // Get split sizes
    auto splitSizes = splitOp.getSplitSizes();
    if (splitSizes.size() != 2)
      return mlir::failure();

    // Check if both split results are used by Act
    auto user0 = *splitOp.getResult(0).getUsers().begin();
    auto user1 = *splitOp.getResult(1).getUsers().begin();
    bool hasAct0 = false, hasAct1 = false;
    if (splitOp.getResult(0).hasOneUse()) {
      hasAct0 = isa<nnayhl::Act>(user0);
    }
    if (splitOp.getResult(1).hasOneUse()) {
      hasAct1 = isa<nnayhl::Act>(user1);
    }

    auto user0Act = dyn_cast<nnayhl::Act>(user0);
    auto user1Act = dyn_cast<nnayhl::Act>(user1);

    // If both outputs are used by Act, we can fold the activation into Conv
    if (hasAct0 && hasAct1 &&
        user0Act.getActivation() == user1Act.getActivation()) {
      op.setActivationAttr(user0Act.getActivation());

      // Replace Act results with split results
      for (int i = 0; i < 2; i++) {
        if (auto siluOp = dyn_cast<nnayhl::Act>(
                *splitOp.getResult(i).getUsers().begin())) {
          siluOp.getResult().replaceAllUsesWith(splitOp.getResult(i));
          rewriter.eraseOp(siluOp);
        }
      }
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct FoldConvActivationPass : public mlir::PassWrapper<FoldConvActivationPass,
                                    mlir::OperationPass<mlir::ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldConvActivationPass)

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FoldConvActivationPattern>(patterns.getContext());
    patterns.add<FoldConvSplitPattern>(patterns.getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(&getContext()))
          << "Failed to fold conv activation";
    }
  }

  mlir::StringRef getArgument() const override {
    return "nnay-fold-conv-activation";
  }
};
} // namespace

namespace onnx_mlir {
namespace nnay {

std::unique_ptr<mlir::Pass> createFoldConvActivationPass() {
  return std::make_unique<FoldConvActivationPass>();
}

} // namespace nnay
} // namespace onnx_mlir