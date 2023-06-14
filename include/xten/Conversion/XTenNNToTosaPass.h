//===- XTenNNToTosaPass.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTEN_NN_TO_TOSA_H
#define XTEN_NN_TO_TOSA_H

#include "mlir/Pass/Pass.h"

namespace amd {
namespace xten_nn {

std::unique_ptr<mlir::Pass> createXTenNNToTOSAPass();

} // namespace xten_nn
} // namespace amd

#endif
