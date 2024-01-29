//===-- XTenNNOps.td - XTenNN ops definitions *---------- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//===----------------------------------------------------------------------===//
//
//   Simplified version of IREE lowering_config attribute.
//
//===----------------------------------------------------------------------===//

#ifndef XTENNN_ATTR
#define XTENNN_ATTR

include "xten/Dialect/XTenNN/IR/XTenNNBase.td"
include "xten/Dialect/XTenNN/IR/XTenNNTypes.td"

include "mlir/IR/EnumAttr.td"

def IREECodegen_LoweringConfigTilingLevelAttr :
  AttrDef<XTenNN_Dialect, "LoweringConfigTilingLevel", []>
{
  let mnemonic = "lowering_config_level";
  let parameters = (ins
    ArrayRefParameter<"int64_t",
        "The tile sizes">:$sizes,
    OptionalArrayRefParameter<"int64_t",
        "The tile interchange">:$interchange);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def XTenNN_LoweringConfigTilingLevelsAttr :
  ArrayOfAttr<XTenNN_Dialect, "LoweringConfigTilingLevels",
    "lowering_config_levels", "LoweringConfigTilingLevelAttr", []>
{
}

def XTenNN_LoweringConfigAttr :
    AttrDef<XTenNN_Dialect, "LoweringConfig", []> {
  let mnemonic = "lowering_config";
  let summary = [{}];
  let description = [{
    TO BE DEFINED
  }];

  let assemblyFormat = [{
    `<` `tile_sizes` `=` $tilingLevels `>`
  }];

  let parameters = (ins
    AttrParameter<"LoweringConfigTilingLevelsAttr",
        "The lowering config at different levels">:$tilingLevels
  );
  let builders = [
    AttrBuilder<(ins "TileSizesListTypeRef":$tileSizes,
        CArg<"TileSizesListTypeRef", "{}">:$tileInterchange)>
  ];
  let extraClassDeclaration = [{
    // Returns the tile sizes for all levels set for the op.
    TileSizesListType getTileSizeVals();

    // Returns the tile sizes for a level set for the op.
    llvm::SmallVector<int64_t> getTileSizeVals(unsigned level);

    // Returns the tile interchange for a level set for the op.
    llvm::SmallVector<int64_t> getTileInterchangeVals(unsigned level);
  }];

  let genVerifyDecl = 1;
}

#endif // XTENNN_OPS
