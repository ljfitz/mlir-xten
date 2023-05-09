//===- XTenPasses.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_TRANSFORM_PASSES_H
#define XTEN_TRANSFORM_PASSES_H

#include "xten/Transform/ATenLoweringPass.h"
#include "xten/Transform/ATenOpReport.h"
#include "xten/Transform/ATenVisualGraph.h"
#include "xten/Transform/LowerToLibATenPass.h"
#include "xten/Transform/XTenCanonicalizePass.h"
#include "xten/Transform/XTenMinimizeLiveTensors.h"

namespace xilinx::xten {
void registerTransformPasses();
} // namespace xilinx::xten

#endif // XTEN_TRANSFORM_PASSES_H
