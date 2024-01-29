//===- XTenNNOps.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XTENNNATTR_H
#define XTENNNATTR_H

#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace amd::xten_nn {
/// Typedef for tile sizes to use at different levels of tiling.
using TileSizesListType = llvm::SmallVector<llvm::SmallVector<int64_t>>;
using TileSizesListTypeRef = llvm::ArrayRef<llvm::SmallVector<int64_t>>;

enum TilingHeuristic { Auto, HardCoded };
} // namespace amd::xten_nn

#define GET_ATTRDEF_CLASSES
#include "xten/Dialect/XTenNN/IR/XTenNNAttr.h.inc"

#endif // XTENNNATTR_H
