//===- TestSubgraphBuilder.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Pass to test subgraph builder utility.
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <xten/Dialect/XTenNN/IR/XTenNNBase.h>
#include <xten/Dialect/XTenNN/Utils/SubgraphBuilder.h>

using namespace mlir;

namespace amd::xten_nn {

namespace {

/// This rewrite pattern wraps each ONNX operation with a linalg::subgraphOp. If
/// \p cloneConstants is set, the defining operations of those operands that
/// are ONNXConstantOp will be also cloned into the subgraph region.
struct WrapArithOpInSubgraph : public RewritePattern {
  WrapArithOpInSubgraph(MLIRContext *context, bool cloneConstants)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        cloneConstants(cloneConstants) {}
  bool cloneConstants;
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    llvm::SmallVector<Operation *> ops;

    // Capture the defining operations of operands that are ONNXConstantOp.
    if (cloneConstants) {
      llvm::for_each(op->getOperands(), [&](Value value) {
        Operation *defOp = value.getDefiningOp();
        if (llvm::isa_and_present<tosa::ConstOp>(defOp))
          ops.push_back(defOp);
      });
    }

    ops.push_back(op);
    auto subgraph = createSubgraphOp(rewriter, ops);
    rewriter.replaceOp(op, subgraph.getResults());
    return success();
  }
};

/// This pass tests createSubgraphOp from SubgraphBuilder.
struct TestSubgraphBuilderPass
    : public PassWrapper<TestSubgraphBuilderPass,
                         OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSubgraphBuilderPass)

  TestSubgraphBuilderPass() = default;
  TestSubgraphBuilderPass(const TestSubgraphBuilderPass &pass)
      : PassWrapper(pass){};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<XTenNNDialect, tosa::TosaDialect>();
  }

  [[nodiscard]] StringRef getArgument() const final {
    return "test-subgraph-builder";
  }
  [[nodiscard]] StringRef getDescription() const final {
    return "test subgraph builder";
  }

  Option<bool> cloneConstants{
      *this, "clone-constants",
      llvm::cl::desc("Clone constant operations inside the region of the "
                     "created subgraph operation"),
      llvm::cl::init(false)};

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<amd::xten_nn::XTenNNDialect>();

    // A TOSA operation is only legal if it is wrapped by a linalg::subgraphOp.
    target.addDynamicallyLegalDialect<tosa::TosaDialect>([&](Operation *op) {
      return llvm::isa<amd::xten_nn::SubgraphOp>(op->getParentOp());
    });

    // Populate the patterns.
    patterns.add<WrapArithOpInSubgraph>(patterns.getContext(), cloneConstants);

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace test {
void registerTestSubgraphBuilder() {
  PassRegistration<TestSubgraphBuilderPass>();
}
} // namespace test

} // namespace amd::xten_nn
