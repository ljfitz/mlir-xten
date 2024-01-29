//===- XTenNNOps.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/TypeSwitch.h"

#include "xten/Dialect/XTenNN/IR/XTenNN.h"
#include "xten/Dialect/XTenNN/IR/XTenNNAttr.h"
#include "xten/Dialect/XTenNN/IR/XTenNNBase.h"
#include "xten/Dialect/XTenNN/IR/XTenNNOps.h"
#include "xten/Dialect/XTenNN/Interfaces/EnclaveOpInterfaces.h"

using namespace mlir;
using namespace amd::xten_nn;

LoweringConfigAttr LoweringConfigAttr::get(MLIRContext *context,
                                           TileSizesListTypeRef tileSizes,
                                           TileSizesListTypeRef tileInterchange) {
  SmallVector<LoweringConfigTilingLevelAttr> tilinglevels;
  for (auto [level, sizes] : llvm::enumerate(tileSizes)) {
    ArrayRef<int64_t> interchange = level < tileInterchange.size()
                                        ? tileInterchange[level]
                                        : ArrayRef<int64_t>{};
    tilinglevels.push_back(
        LoweringConfigTilingLevelAttr::get(context, sizes, interchange));
  }
  return get(context,
             LoweringConfigTilingLevelsAttr::get(context, tilinglevels));
}

TileSizesListType LoweringConfigAttr::getTileSizeVals() {
  TileSizesListType tileSizes;
  for (auto &level : getTilingLevels())
    tileSizes.push_back(SmallVector<int64_t>(level.getSizes()));
  return tileSizes;
}

SmallVector<int64_t> LoweringConfigAttr::getTileSizeVals(unsigned level) {
  auto levels = getTilingLevels();
  if (level >= levels.size())
    return {};
  return SmallVector<int64_t>(levels[level].getSizes());
}

SmallVector<int64_t>
LoweringConfigAttr::getTileInterchangeVals(unsigned level) {
  auto levels = getTilingLevels();
  if (level >= levels.size())
    return {};
  return SmallVector<int64_t>(levels[level].getInterchange());
}

LogicalResult
LoweringConfigAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           LoweringConfigTilingLevelsAttr levels) {
  if (!levels)
    return emitError() << "missing lowering config levels";
  return success();
}

void LoweringConfigTilingLevelAttr::print(mlir::AsmPrinter &printer) const {
  auto tileInterchange = getInterchange();
  auto printTileSizes = [&] {
    printer << '[';
    printer.printStrippedAttrOrType(getSizes());
    printer << ']';
  };
  if (tileInterchange.empty()) {
    printTileSizes();
  } else {
    printer << "{sizes = ";
    printTileSizes();
    printer << ", interchange = [";
    printer.printStrippedAttrOrType(tileInterchange);
    printer << "]}";
  }
}

Attribute LoweringConfigTilingLevelAttr::parse(mlir::AsmParser &parser,
                                               mlir::Type  /*type*/) {
  auto loc = parser.getCurrentLocation();
  auto parseListOfSizes = [&](bool prefixChecked =
                                  false) -> FailureOr<SmallVector<int64_t>> {
    if (!prefixChecked && parser.parseLSquare())
      return failure();
    if (parser.parseOptionalRSquare().succeeded()) {
      // Empty list.
      return SmallVector<int64_t>();
    }
    SmallVector<int64_t> sizes;
    auto listParse =
        parser.parseCommaSeparatedList(AsmParser::Delimiter::None, [&] {
          int64_t size = 0;
          if (parser.parseInteger(size))
            return failure();
          sizes.push_back(size);
          return success();
        });
    if (failed(listParse) || parser.parseRSquare())
      return failure();
    return sizes;
  };
  // {sizes = [0, 32, 16], interchange = [0, 1, 2]}
  if (parser.parseLBrace() || parser.parseKeyword("sizes") ||
      parser.parseEqual())
    return {};
  auto tileSizes = parseListOfSizes();
  if (failed(tileSizes) || parser.parseComma() ||
      parser.parseKeyword("interchange") || parser.parseEqual())
    return {};
  auto tileInterchange = parseListOfSizes();
  if (failed(tileInterchange) || parser.parseRBrace())
    return {};
  return parser.getChecked<LoweringConfigTilingLevelAttr>(
      loc, parser.getContext(), *tileSizes, *tileInterchange);
}

LogicalResult LoweringConfigTilingLevelAttr::verify(
    function_ref<InFlightDiagnostic()>  /*emitError*/, ArrayRef<int64_t>  /*tileSizes*/,
    ArrayRef<int64_t>  /*tileInterchange*/) {
  return success();
}

#define GET_ATTRDEF_CLASSES
#include "xten/Dialect/XTenNN/IR/XTenNNAttr.cpp.inc"

//===----------------------------------------------------------------------===//
// XTenNNDialect
//===----------------------------------------------------------------------===//

/// Print an attribute registered to this dialect.
void XTenNNDialect::printAttribute(Attribute attr,
                                   DialectAsmPrinter &printer) const {
  // generatedAttributePrinter is generated in ONNXAttributes.cpp.inc
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
}

void XTenNNDialect::registerAttributes() {
#define GET_ATTRDEF_LIST
  addAttributes<
#include "xten/Dialect/XTenNN/IR/XTenNNAttr.cpp.inc"
      >();
}
