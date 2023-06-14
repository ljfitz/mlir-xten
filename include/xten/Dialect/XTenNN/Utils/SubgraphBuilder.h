//===- SubgraphBuilder.h ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Declares the XTenNN Subgraph Utilities.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

#include <xten/Dialect/XTenNN/IR/XTenNNOps.h>

namespace amd::xten_nn {

SubgraphOp
createSubgraphOp(mlir::PatternRewriter &rewriter,
                 const llvm::SmallVector<mlir::Operation *> &operations);

} // namespace amd::xten_nn
