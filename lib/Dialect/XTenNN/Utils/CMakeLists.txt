#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

add_mlir_dialect_library(XTenNNUtils
  SubgraphBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${XTEN_SOURCE_DIR}/include/xten/Dialect/XTenNN/Utils

  DEPENDS
  XTenNNIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  XTenNNDialect
)
