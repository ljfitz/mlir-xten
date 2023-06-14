//===- quantization.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s --xten-nn-to-tosa --split-input-file | FileCheck %s

module attributes{} {
// CHECK-LABEL:     func.func @explicit_case(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_2:.*]] = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_3:.*]] = "tosa.mul"(%[[VAL_0]], %[[VAL_1]]) {shift = 0 : i32} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_4:.*]] = "tosa.cast"(%[[VAL_3]]) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_5:.*]] = "tosa.cast"(%[[VAL_4]]) : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_6:.*]] = "tosa.mul"(%[[VAL_5]], %[[VAL_2]]) {shift = 0 : i32} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_6]] : tensor<1x3x4x4xf32>
// CHECK:           }
    func.func @explicit_case(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
        %0 = xten_nn.quantize(%arg0 : tensor<1x3x4x4xf32>) {shift = -5 : si32} -> tensor<1x3x4x4xsi8>
        %1 = xten_nn.dequantize(%0 : tensor<1x3x4x4xsi8>) {shift = -5 : si32} -> tensor<1x3x4x4xf32>
        return %1 : tensor<1x3x4x4xf32>
    }
}

// --

module attributes{} {
// CHECK-LABEL:     func.func @small_tensors(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<2x3xf32>) -> tensor<2x3xf32> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() {value = dense<1.250000e-01> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
// CHECK:             %[[VAL_2:.*]] = "tosa.const"() {value = dense<8.000000e+00> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
// CHECK:             %[[VAL_3:.*]] = "tosa.mul"(%[[VAL_0]], %[[VAL_1]]) {shift = 0 : i32} : (tensor<2x3xf32>, tensor<1x1xf32>) -> tensor<2x3xf32>
// CHECK:             %[[VAL_4:.*]] = "tosa.cast"(%[[VAL_3]]) : (tensor<2x3xf32>) -> tensor<2x3xi8>
// CHECK:             %[[VAL_5:.*]] = "tosa.clamp"(%[[VAL_4]]) {max_fp = 7.000000e+00 : f32, max_int = 7 : i64, min_fp = -8.000000e+00 : f32, min_int = -8 : i64} : (tensor<2x3xi8>) -> tensor<2x3xi8>
// CHECK:             %[[VAL_6:.*]] = "tosa.cast"(%[[VAL_5]]) : (tensor<2x3xi8>) -> tensor<2x3xf32>
// CHECK:             %[[VAL_7:.*]] = "tosa.mul"(%[[VAL_6]], %[[VAL_2]]) {shift = 0 : i32} : (tensor<2x3xf32>, tensor<1x1xf32>) -> tensor<2x3xf32>
// CHECK:             return %[[VAL_7]] : tensor<2x3xf32>
// CHECK:           }
    func.func @small_tensors(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %0 = xten_nn.quantize(%arg0 : tensor<2x3xf32>) {shift = 3 : si32} -> tensor<2x3xsi4>
        %1 = xten_nn.dequantize(%0 : tensor<2x3xsi4>) {shift = 3 : si32} -> tensor<2x3xf32>
        return %1 : tensor<2x3xf32>
    }
}

// --

module attributes{} {
// CHECK-LABEL:     func.func @quantize_case(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xsi8> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_2:.*]] = "tosa.mul"(%[[VAL_0]], %[[VAL_1]]) {shift = 0 : i32} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_3:.*]] = "tosa.cast"(%[[VAL_2]]) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_3]] : tensor<1x3x4x4xi8> to tensor<1x3x4x4xsi8>
// CHECK:             return %[[VAL_4]] : tensor<1x3x4x4xsi8>
// CHECK:           }
    func.func @quantize_case(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xsi8> {
        %0 = xten_nn.quantize(%arg0 : tensor<1x3x4x4xf32>) {shift = -5 : si32} -> tensor<1x3x4x4xsi8>
        return %0 : tensor<1x3x4x4xsi8>
    }
}

// --

module attributes{} {
// CHECK-LABEL:     func.func @dequantize_case(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<1x3x4x4xsi8>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : tensor<1x3x4x4xsi8> to tensor<1x3x4x4xi8>
// CHECK:             %[[VAL_3:.*]] = "tosa.cast"(%[[VAL_2]]) : (tensor<1x3x4x4xi8>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_4:.*]] = "tosa.mul"(%[[VAL_3]], %[[VAL_1]]) {shift = 0 : i32} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_4]] : tensor<1x3x4x4xf32>
// CHECK:           }
    func.func @dequantize_case(%arg0: tensor<1x3x4x4xsi8>) -> tensor<1x3x4x4xf32> {
        %0 = xten_nn.dequantize(%arg0 : tensor<1x3x4x4xsi8>) {shift = -5 : si32} -> tensor<1x3x4x4xf32>
        return %0 : tensor<1x3x4x4xf32>
    }
}

// --

module attributes{} {
// CHECK-LABEL:     func.func @i16_case(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_2:.*]] = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_3:.*]] = "tosa.mul"(%[[VAL_0]], %[[VAL_1]]) {shift = 0 : i32} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_4:.*]] = "tosa.cast"(%[[VAL_3]]) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi16>
// CHECK:             %[[VAL_5:.*]] = "tosa.cast"(%[[VAL_4]]) : (tensor<1x3x4x4xi16>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_6:.*]] = "tosa.mul"(%[[VAL_5]], %[[VAL_2]]) {shift = 0 : i32} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_6]] : tensor<1x3x4x4xf32>
// CHECK:           }
    func.func @i16_case(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
        %0 = xten_nn.quantize(%arg0 : tensor<1x3x4x4xf32>) {shift = -5 : si32} -> tensor<1x3x4x4xsi16>
        %1 = xten_nn.dequantize(%0 : tensor<1x3x4x4xsi16>) {shift = -5 : si32} -> tensor<1x3x4x4xf32>
        return %1 : tensor<1x3x4x4xf32>
    }
}

// --

module attributes{} {
// CHECK-LABEL:     func.func @i12_case(
// CHECK-SAME:                          %[[VAL_0:.*]]: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() {value = dense<3.200000e+01> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_2:.*]] = "tosa.const"() {value = dense<3.125000e-02> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
// CHECK:             %[[VAL_3:.*]] = "tosa.mul"(%[[VAL_0]], %[[VAL_1]]) {shift = 0 : i32} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_4:.*]] = "tosa.cast"(%[[VAL_3]]) : (tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xi16>
// CHECK:             %[[VAL_5:.*]] = "tosa.clamp"(%[[VAL_4]]) {max_fp = 2.047000e+03 : f32, max_int = 2047 : i64, min_fp = -2.048000e+03 : f32, min_int = -2048 : i64} : (tensor<1x3x4x4xi16>) -> tensor<1x3x4x4xi16>
// CHECK:             %[[VAL_6:.*]] = "tosa.cast"(%[[VAL_5]]) : (tensor<1x3x4x4xi16>) -> tensor<1x3x4x4xf32>
// CHECK:             %[[VAL_7:.*]] = "tosa.mul"(%[[VAL_6]], %[[VAL_2]]) {shift = 0 : i32} : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
// CHECK:             return %[[VAL_7]] : tensor<1x3x4x4xf32>
// CHECK:           }
    func.func @i12_case(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
        %0 = xten_nn.quantize(%arg0 : tensor<1x3x4x4xf32>) {shift = -5 : si32} -> tensor<1x3x4x4xsi12>
        %1 = xten_nn.dequantize(%0 : tensor<1x3x4x4xsi12>) {shift = -5 : si32} -> tensor<1x3x4x4xf32>
        return %1 : tensor<1x3x4x4xf32>
    }
}
