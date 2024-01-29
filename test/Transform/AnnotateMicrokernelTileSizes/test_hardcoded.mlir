// RUN: aten-opt %s -annotate-microkernel-tile-sizes="tiling-heuristic=hard-coded" -split-input-file | FileCheck %s

// CHECK: [[CONFIG:.+]] = #xten_nn.lowering_config<tile_sizes = {{\[\[}}16, 64], [0, 0, 64], [1, 1], [1, 1]]>

// CHECK-LABEL: forward
// CHECK: linalg.fill {lowering_config = [[CONFIG]]}
// CHECK: linalg.matmul {lowering_config = [[CONFIG]]}
module attributes {torch.debug_module_name = "_lambda"} {
  func.func @forward(%arg0: tensor<1x256x256xi8>) -> tensor<1x256x256xi32> {
    %0 = xten_nn.subgraph (%arg1 = %arg0: tensor<1x256x256xi8>, %arg2 = %arg0: tensor<1x256x256xi8>)  attributes {LayerName = "FXML-2215-#0", Operands = [{External = false, Port = "data_io.mat_a", l3_extend_end = dense<0> : vector<3xindex>, l3_tile_count = dense<[1, 256, 256]> : vector<3xindex>}, {External = false, Port = "data_io.mat_b", SubPort = "mat_b_data", l3_extend_end = dense<0> : vector<3xindex>, l3_tile_count = dense<[1, 256, 256]> : vector<3xindex>}], OutputName = "FXML-2215-#1", Reason = "Microkernel", Results = [{Port = "data_io.mat_c", l3_extend_end = dense<0> : vector<3xindex>, l3_tile_count = dense<[1, 256, 256]> : vector<3xindex>}], Specializes = "GemmI8I32", With = {config.act = 0 : ui32, config.activation = 0 : ui32, config.aie_arch = "aie-ml", config.bias_flag = 0 : ui32, config.dtype = "int8", config.transpose_enable = 0 : ui32}, l3_nhw_extend_end_matA = dense<0> : vector<3xindex>, l3_nhw_extend_end_matB = dense<0> : vector<3xindex>, l3_nhw_extend_end_output = dense<0> : vector<3xindex>} {
      %c0_i32 = arith.constant 0 : i32
      %collapsed = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<1x256x256xi8> into tensor<256x256xi8>
      %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<1x256x256xi8> into tensor<256x256xi8>
      %1 = tensor.empty() : tensor<256x256xi32>
      %2 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<256x256xi32>) -> tensor<256x256xi32>
      %3 = linalg.matmul ins(%collapsed, %collapsed_0 : tensor<256x256xi8>, tensor<256x256xi8>) outs(%2 : tensor<256x256xi32>) -> tensor<256x256xi32>
      %expanded = tensor.expand_shape %3 [[0, 1], [2]] : tensor<256x256xi32> into tensor<1x256x256xi32>
      xten_nn.output %expanded : tensor<1x256x256xi32>
    } -> tensor<1x256x256xi32>
    return %0 : tensor<1x256x256xi32>
  }
}

