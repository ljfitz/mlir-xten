//===- Passes.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Declares the XTenNN pass entry points.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "xten/Dialect/XTenNN/Transforms/CanonicalizePass.h"
#include "xten/Dialect/XTenNN/Transforms/Simplify.h"

namespace amd::xten_nn {
#define GEN_PASS_DECL_ANNOTATEMICROKERNELTILESIZES
#include "xten/Dialect/XTenNN/Transforms/Passes.h.inc"
std::unique_ptr<mlir::Pass> createAnnotateMicrokernelTileSizes();
} // namespace amd::xten_nn

namespace amd::xten_nn {
#define GEN_PASS_REGISTRATION
#include "xten/Dialect/XTenNN/Transforms/Passes.h.inc"

} // namespace amd::xten_nn