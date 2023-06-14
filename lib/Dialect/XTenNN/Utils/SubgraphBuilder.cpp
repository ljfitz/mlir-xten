//===- SubgraphBuilder.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Implements the XTenNN Subgraph Utilities.
//
//===----------------------------------------------------------------------===//

#include <xten/Dialect/XTenNN/IR/XTenNNOps.h>
#include <xten/Dialect/XTenNN/Interfaces/EnclaveOpInterfaces.h>
#include <xten/Dialect/XTenNN/Utils/SubgraphBuilder.h>

#include <cassert>

using namespace llvm;
using namespace mlir;

/// Fills \b uses with all(nested) operands of \b op that are not provided by
/// \b root.
static void collectUses(Operation *root, Operation *op,
                        llvm::SmallVectorImpl<Value> &uses) {
  // Check if the value has been defined locally inside the region.
  const auto isLocal = [&](Value value) {
    auto *source = value.getDefiningOp();
    if (auto barg = value.dyn_cast<BlockArgument>())
      source = barg.getOwner()->getParentOp();
    while (source != nullptr) {
      if (source == root)
        return true;
      source = source->getParentOp();
    }
    return false;
  };

  // Check if the operand has been defined locally or it has reference to the
  // outside of the region. It the latter is the case, it will be added to
  // list of uses if it has not been added before.
  llvm::for_each(op->getOperands(), [&](Value operand) {
    if (isLocal(operand))
      return;
    if (llvm::find(uses, operand) != uses.end())
      return;
    uses.push_back(operand);
  });

  // Collect the uses for all the operands of the all operations inside the
  // operation region.
  for (Region &region : op->getRegions()) {
    for (Operation &op : region.getOps()) {
      collectUses(root, &op, uses);
    }
  }
}

/// This helper function creates a subgraph and clone all the \p operations into
/// the region of the subgraph. The pass assumes that the last operation in \p
/// operations is the target operation whose results need to be returned. The
/// results are returned using an Yield operation. The target operation is
/// replaced with the created subgraph.
namespace amd::xten_nn {
SubgraphOp createSubgraphOp(
    PatternRewriter &rewriter,
    const llvm::SmallVector<Operation *> &operations) {
  assert(!operations.empty() &&
         "There should be at least one operation to create the subgraph.");

  // To isolate the subgraph operation from the above, all references
  // from inside the region of wrapped operations to outside region must be
  // cut and new operands must be added to the subgraph.
  SmallVector<Value> captures;
  llvm::for_each(operations,
                 [&](Operation *op) { collectUses(op, op, captures); });
  // Remove those operands that their defining operation is among cloned
  // operations.
  llvm::erase_if(captures, [&](Value value) {
    return llvm::any_of(
        operations, [&](Operation *op) { return value.getDefiningOp() == op; });
  });

  // The last operation is considered as the target operation.
  Operation *target = operations.back();
  Type resultType =
      target->getNumResults() != 0U ? target->getResult(0).getType() : Type{};
  // Create the Subgraph and replace the target

  // Gather all locations of ops that are to be placed inside the subgraph for a
  // FusedLoc
  SmallVector<Location, 4> locs;
  for (Operation *op : operations) {
    locs.emplace_back(op->getLoc());
  }

  auto subgraphLoc = FusedLoc::get(rewriter.getContext(), locs);
  auto subgraph = rewriter.create<amd::xten_nn::SubgraphOp>(
      subgraphLoc, resultType, ValueRange());

  auto *block = rewriter.createBlock(&subgraph.getRegion());
  // Clone original operations into the subgraph region
  rewriter.setInsertionPointToEnd(block);
  IRMapping mapping;
  subgraph.capture(captures, mapping);
  Operation *clonedOp = nullptr;
  for (const auto &originalOp : operations) {
    clonedOp = rewriter.clone(*originalOp, mapping);
  }
  rewriter.create<amd::xten_nn::OutputOp>(target->getLoc(),
                                          clonedOp->getResults());

  return subgraph;
}

} // namespace amd::xten_nn
