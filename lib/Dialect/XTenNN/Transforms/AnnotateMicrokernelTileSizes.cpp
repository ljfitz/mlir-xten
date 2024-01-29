//===- AnnotateMicrokernelTilesSizesPass.cpp --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xten/Dialect/XTenNN/IR/XTenNNAttr.h"
#include "xten/Dialect/XTenNN/Transforms/Passes.h"

#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "xten-canonicalize"

namespace amd::xten_nn {
using namespace mlir;
#define GEN_PASS_DEF_ANNOTATEMICROKERNELTILESIZES
#include "xten/Dialect/XTenNN/Transforms/Passes.h.inc"
} // namespace amd::xten_nn

using namespace mlir;
using namespace amd::xten_nn;

namespace {

// Hardcoded tile sizes.
const TileSizesListType hardCodedTileSizes = {
    llvm::SmallVector<int64_t>{16, 64}, llvm::SmallVector<int64_t>{0, 0, 64},
    llvm::SmallVector<int64_t>{1, 1}, llvm::SmallVector<int64_t>{1, 1}};

/// Generate tiling strategy with Divide-by-Two on last dimension.
FailureOr<llvm::SmallVector<int64_t>> generateAutoTiling(Type ty) {
  auto shapedTy = dyn_cast<ShapedType>(ty);
  if (!shapedTy)
    return failure();

  // Get last dim information.
  auto lastDim = shapedTy.getRank() - 1;
  auto lastDimValue = shapedTy.getDimSize(lastDim);

  // Create a new vector with last dim divided by 2.
  llvm::SmallVector<int64_t> shape(shapedTy.getShape());
  shape[lastDim] = lastDimValue / 2;
  return shape;
}

/// Generate automatic tiling strategy for operands and return types according
/// to Divide-by-Two heuristic.
FailureOr<TileSizesListType> generateAutoTiling(Operation *op) {
  TileSizesListType tileSizes;
  for (auto ty : op->getOperandTypes()) {
    auto tileSize = generateAutoTiling(ty);
    if (failed(tileSize))
      continue;
    tileSizes.push_back(tileSize.value());
  }
  return tileSizes;
}

void annotateTileSizes(MLIRContext &ctx, Operation *op,
                       bool useHardCodedTiling) {

  TileSizesListType tileSizes;
  if (useHardCodedTiling) {
    tileSizes = hardCodedTileSizes;
  } else {
    auto autoTileSizes = generateAutoTiling(op);

    // If no strategy is found, return it.
    if (failed(autoTileSizes))
      return;
    tileSizes = autoTileSizes.value();
  }
  op->setAttr("lowering_config", LoweringConfigAttr::get(&ctx, tileSizes));
}

struct AnnotateMicrokernelTileSizes
    : public amd::xten_nn::impl::AnnotateMicrokernelTileSizesBase<
          AnnotateMicrokernelTileSizes> {
  using AnnotateMicrokernelTileSizesBase::AnnotateMicrokernelTileSizesBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Traverse the IR and collects all subgraphs.
    getOperation()->walk([&](amd::xten_nn::SubgraphOp sg) {
      sg.walk([&](linalg::LinalgOp linalgOp) {
        // Annotate all linalg operations.
        annotateTileSizes(*ctx, linalgOp,
                          this->tilingHeuristic == TilingHeuristic::HardCoded);
      });
    });

    // module
  }
};
} // namespace

std::unique_ptr<Pass> amd::xten_nn::createAnnotateMicrokernelTileSizes() {
  return std::make_unique<AnnotateMicrokernelTileSizes>();
}
