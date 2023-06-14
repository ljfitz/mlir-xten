// RUN: aten-opt %s -test-subgraph-builder=clone-constants=false -o - | FileCheck %s

//      CHECK: func.func @wrap_each_in_subgraph(%[[ARG0:.*]]: {{.*}})
func.func @wrap_each_in_subgraph(%arg0: tensor<1x4x224x224xf32>) -> tensor<1x64x112x112xf32> attributes {input_names = ["global_input_0"], output_names = ["global_outout_0"]} {
//      CHECK: %[[CONST_SUBGRAPH_1:.*]] = xten_nn.subgraph ()
// CHECK-NEXT: %[[CONSTANT_1:.*]] = "tosa.const"
// CHECK-NEXT: output %[[CONSTANT_1]]
  %1 = "tosa.const"() {value = dense<0.02> : tensor<64x4x7x7xf32>} : () -> tensor<64x4x7x7xf32>
//      CHECK: %[[CONST_SUBGRAPH_2:.*]] = xten_nn.subgraph ()
// CHECK-NEXT: %[[CONSTANT_2:.*]] = "tosa.const"
// CHECK-NEXT: output %[[CONSTANT_2]]
  %2 = "tosa.const"() {value = dense<0.01> : tensor<64xf32>} : () -> tensor<64xf32>
//      CHECK: %[[CONV_SUBGRAPH:.*]] = xten_nn.subgraph (
// CHECK-SAME: %[[CONV_ARG1:.*]] = %[[ARG0]]
// CHECK-SAME: %[[CONV_ARG2:.*]] = %[[CONST_SUBGRAPH_1]]
// CHECK-SAME: %[[CONV_ARG3:.*]] = %[[CONST_SUBGRAPH_2]]
// CHECK-NEXT: %[[CONV:.*]] = "tosa.conv2d"(%[[CONV_ARG1]], %[[CONV_ARG2]], %[[CONV_ARG3]])
// CHECK-NEXT: output %[[CONV]]
  %3 = "tosa.conv2d"(%arg0, %1, %2) {dilation = array<i64: 1, 1>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>} : (tensor<1x4x224x224xf32>, tensor<64x4x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
//      CHECK: %[[RELU_SUBGRAPH:.*]] = xten_nn.subgraph (
// CHECK-SAME: %[[RELU_ARG:.*]] = %[[CONV_SUBGRAPH]]
// CHECK-NEXT: %[[RELU:.*]] = "tosa.clamp"(%[[RELU_ARG]])
// CHECK-NEXT: output %[[RELU]]
  %4 = "tosa.clamp"(%3) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
//      CHECK: return %[[RELU_SUBGRAPH]]
  return %4 : tensor<1x64x112x112xf32>
}
