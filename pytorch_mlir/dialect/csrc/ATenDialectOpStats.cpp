#include "ATenDialect.h"

namespace {

unsigned getTensorVolume(TensorType ty) {
  unsigned volume = 1;
  for (auto &d : ty.getShape())
    volume *= d;
  return volume;
}

unsigned getTensorVolume(MemRefType ty) {
  unsigned volume = 1;
  for (auto &d : ty.getShape())
    volume *= d;
  return volume;
}

unsigned getTensorVolume(Type ty) {
  if (auto t = ty.cast<TensorType>()) {
    return getTensorVolume(t);
  }
  else if (auto t = ty.cast<MemRefType>()) {
    return getTensorVolume(t);
  }
  else {
    return 0;
  }
}

} // namespace

namespace xilinx {
namespace aten {

// add
std::map<std::string, unsigned> AddOp::updateStatistics() {

  std::map<std::string, unsigned> toReturn;

  Type resultTy = getResult()->getType();
  unsigned ofm_volume = getTensorVolume(resultTy);
  toReturn["+"] = ofm_volume;
  toReturn["activation_out"] = ofm_volume;

  TensorType tensorResultTy = resultTy.cast<TensorType>();
  unsigned num_output_neurons = tensorResultTy.getShape()[1];

  // Find the size of the A and B operands
  Type aType = getOperand(0)->getType();
  unsigned a_volume = getTensorVolume(aType);

  Type bType = getOperand(1)->getType();
  unsigned b_volume = getTensorVolume(bType);

  toReturn["activation_in"] = a_volume + b_volume;
  return(toReturn);
}

// add_
std::map<std::string, unsigned> AddUnderOp::updateStatistics() {
  std::map<std::string, unsigned> toReturn;

  Type resultTy = getResult()->getType();
  unsigned ofm_volume = getTensorVolume(resultTy);

  toReturn["+"] = ofm_volume;
  toReturn["activation_out"] = ofm_volume;

  TensorType tensorResultTy = resultTy.cast<TensorType>();
  unsigned num_output_neurons = tensorResultTy.getShape()[1];

  // Find the size of the A and B operands
  Type aType = getOperand(0)->getType();
  unsigned a_volume = getTensorVolume(aType);

  Type bType = getOperand(1)->getType();
  unsigned b_volume = getTensorVolume(bType);

  toReturn["activation_in"] = a_volume + b_volume;
  return(toReturn);
}

// addmm
std::map<std::string, unsigned> AddmmOp::updateStatistics() {

  std::map<std::string, unsigned> toReturn;
  // For linear, we need the number of output neurons and the number of input neurons
  // Then the number of forward MACs is input * output
  // And the number of adds is output if there is bias

  Type resultTy = getResult()->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();
  unsigned num_output_neurons = tensorResultTy.getShape()[1];
  unsigned ofm_volume = getTensorVolume(tensorResultTy);

  // Use the weight tensor to find the number of input neurons
  Type wType = getOperand(2)->getType();
  TensorType wTy = wType.cast<TensorType>();
  unsigned num_input_neurons = wTy.getShape()[0];
  unsigned total_MACs = ofm_volume * num_input_neurons;
  unsigned weight_volume = getTensorVolume(wTy);

  Type ifmType = getOperand(1)->getType();
  TensorType txTy = ifmType.cast<TensorType>();
  unsigned ifm_volume = getTensorVolume(txTy);

  toReturn["MAC"] = total_MACs;
  toReturn["+"] = ofm_volume;   // Should be gated on whether there is bias at all
  toReturn["activation_in"] = ifm_volume;
  toReturn["activation_out"] = ofm_volume;
  toReturn["parameters_in"] = weight_volume + num_output_neurons;

  return(toReturn);
}

// batch_norm
std::map<std::string, unsigned>  BatchNormOp::updateStatistics() {
  std::map<std::string, unsigned> toReturn;
  Type resultTy = getResult(0)->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  unsigned op_volume = getTensorVolume(tensorResultTy);
  toReturn["activation_in"] = op_volume;
  toReturn["activation_out"] = op_volume;

  // There are 2x as many parameters are there are planes ...
  unsigned ifm_depth = tensorResultTy.getShape()[1];
  toReturn["parameters_in"] = ifm_depth * 2;

  // Now for the arithmetic.  Assume variance is calculated as sum of squares

  toReturn["+"] = op_volume;   // Add up for mean
  toReturn["*"] = op_volume;   // Square for variance
  toReturn["+"] += op_volume;  // Add up squares for variance

  toReturn["*"] += ifm_depth;   // Calc channel means
  toReturn["-"] += ifm_depth;   // Calc channel vars
  toReturn["*"] += ifm_depth;   // Calc channel vars

  toReturn["sqrt"] = ifm_depth;  // Convert to SD
  toReturn["/"] += ifm_depth;    // Get the reciprocal

  toReturn["+"] += op_volume;   // Subtract mean off each pixel
  toReturn["*"] += op_volume;   // Multiply by 1/SD for each pixel

  toReturn["+"] += op_volume;   // Bias
  toReturn["*"] += op_volume;   // Scale

  return(toReturn);
}

// _convolution
std::map<std::string, unsigned> ConvolutionOp::updateStatistics() {

  std::map<std::string, unsigned> toReturn;

  // For convolution, we need the OFM volume.
  // Then the number of forward MACs per pixel are kernel_width * kernel_height * ifm_depth / groups

  Type resultTy = getResult()->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  unsigned ofm_volume = getTensorVolume(tensorResultTy);
  unsigned ofm_depth = tensorResultTy.getShape()[1];

  // All the info we need for MACs is in the weight tensor
  Type wType = getOperand(1)->getType();
  TensorType wTy = wType.cast<TensorType>();

  unsigned ifm_depth = wTy.getShape()[1];
  unsigned kernel_width = wTy.getShape()[2];
  unsigned kernel_height = wTy.getShape()[3];
  unsigned groups = 1; // It's one of the operands, not sure which

  unsigned MACs_per_OFM = (ifm_depth/groups) * kernel_height * kernel_width;
  unsigned total_MACs = ofm_volume * MACs_per_OFM;

  Type ifmType = getOperand(0)->getType();
  TensorType txTy = ifmType.cast<TensorType>();
  unsigned ifm_volume = getTensorVolume(txTy);
  unsigned weight_volume = getTensorVolume(wTy);

  unsigned bias_volume = ofm_depth;  // See below
  toReturn["+"] = ofm_volume;        // Should be gated on whether there is bias at all

  toReturn["MAC"] = total_MACs;
  toReturn["activation_in"] = ifm_volume;
  toReturn["activation_out"] = ofm_volume;
  toReturn["parameters_in"] = weight_volume + ofm_depth;

  return(toReturn);
}

// max_pool2d
std::map<std::string, unsigned> MaxPool2dOp::updateStatistics() {

  auto unpack = [](auto &op, auto &v) -> void {
                  auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
                  DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
                  for (auto i : a.getIntValues())
                    v.push_back(i.getSExtValue());
                };

  std::map<std::string, unsigned>  toReturn;
  Type resultTy = getResult()->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  unsigned ofm_volume = getTensorVolume(tensorResultTy);
  toReturn["activation_out"] = ofm_volume;

  Type ifmType = getOperand(0)->getType();
  TensorType txTy = ifmType.cast<TensorType>();
  unsigned ifm_volume = getTensorVolume(txTy);
  toReturn["activation_in"] = ifm_volume;

  // To find the number of compares, we need the filter extent

  std::vector<uint64_t> kernel;
  mlir::Value *k = getOperand(1);
  unpack(k, kernel);

  unsigned aperture = kernel[0] * kernel[1];
  toReturn[">"] = ofm_volume * aperture;

  return(toReturn);
}

// max_pool2d_with_indices
std::map<std::string, unsigned> MaxPool2dWithIndicesOp::updateStatistics() {

  auto unpack = [](auto &op, auto &v) -> void {
                  auto co = cast<xilinx::aten::ConstantOp>(op->getDefiningOp());
                  DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
                  for (auto i : a.getIntValues())
                    v.push_back(i.getSExtValue());
                };

  std::map<std::string, unsigned>  toReturn;
  Type resultTy = getResult(0)->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  unsigned ofm_volume = getTensorVolume(tensorResultTy);
  toReturn["activation_out"] = ofm_volume;

  Type ifmType = getOperand(0)->getType();
  TensorType txTy = ifmType.cast<TensorType>();
  unsigned ifm_volume = getTensorVolume(txTy);
  toReturn["activation_in"] = ifm_volume;

  // To find the number of compares, we need the filter extent

  std::vector<uint64_t> kernel;
  mlir::Value *k = getOperand(1);
  unpack(k, kernel);

  unsigned aperture = kernel[0] * kernel[1];
  toReturn[">"] = ofm_volume * aperture;

  return(toReturn);
}

// mul
std::map<std::string, unsigned> MulOp::updateStatistics() {
  std::map<std::string, unsigned> toReturn;

  Type resultTy = getResult()->getType();
  unsigned ofm_volume = getTensorVolume(resultTy);
  toReturn["+"] = ofm_volume;
  toReturn["activation_out"] = ofm_volume;

  TensorType tensorResultTy = resultTy.cast<TensorType>();
  unsigned num_output_neurons = tensorResultTy.getShape()[1];

  // Find the size of the A and B operands
  Type aType = getOperand(0)->getType();
  unsigned a_volume = getTensorVolume(aType);

  Type bType = getOperand(1)->getType();
  unsigned b_volume = getTensorVolume(bType);

  toReturn["activation_in"] = a_volume + b_volume;
  return(toReturn);
}

// mul_
std::map<std::string, unsigned> MulUnderOp::updateStatistics() {
  std::map<std::string, unsigned> toReturn;

  Type resultTy = getResult()->getType();
  unsigned ofm_volume = getTensorVolume(resultTy);
  toReturn["+"] = ofm_volume;
  toReturn["activation_out"] = ofm_volume;

  TensorType tensorResultTy = resultTy.cast<TensorType>();
  unsigned num_output_neurons = tensorResultTy.getShape()[1];

  // Find the size of the A and B operands
  Type aType = getOperand(0)->getType();
  unsigned a_volume = getTensorVolume(aType);

  Type bType = getOperand(1)->getType();
  unsigned b_volume = getTensorVolume(bType);

  toReturn["activation_in"] = a_volume + b_volume;
  return(toReturn);
}

// native_batch_norm
std::map<std::string, unsigned> NativeBatchNormOp::updateStatistics() {
  std::map<std::string, unsigned>  toReturn;
  Type resultTy = getResult(0)->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  unsigned op_volume = getTensorVolume(tensorResultTy);
  toReturn["activation_in"] = op_volume;
  toReturn["activation_out"] = op_volume;

  // There are 2x as many parameters are there are planes ...
  unsigned ifm_depth = tensorResultTy.getShape()[1];
  toReturn["parameters_in"] = ifm_depth * 2;

  // Now for the arithmetic.  Assume variance is calculated as sum of squares

  toReturn["+"] = op_volume;   // Add up for mean
  toReturn["*"] = op_volume;   // Square for variance
  toReturn["+"] += op_volume;  // Add up squares for variance

  toReturn["*"] += ifm_depth;   // Calc channel means
  toReturn["-"] += ifm_depth;   // Calc channel vars
  toReturn["*"] += ifm_depth;   // Calc channel vars

  toReturn["sqrt"] = ifm_depth;  // Convert to SD
  toReturn["/"] += ifm_depth;    // Get the reciprocal

  toReturn["+"] += op_volume;   // Subtract mean off each pixel
  toReturn["*"] += op_volume;   // Multiply by 1/SD for each pixel

  toReturn["+"] += op_volume;   // Bias
  toReturn["*"] += op_volume;   // Scale

  return(toReturn);
}

// relu
std::map<std::string, unsigned> ReLUOp::updateStatistics() {
  std::map<std::string, unsigned> toReturn;
  Type resultTy = getResult()->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  unsigned op_volume = getTensorVolume(tensorResultTy);
  toReturn["activation_in"] = op_volume;
  toReturn["activation_out"] = op_volume;
  toReturn[">"] = op_volume;

  return(toReturn);
}

// relu_
std::map<std::string, unsigned> ReLUUnderOp::updateStatistics() {
  std::map<std::string, unsigned> toReturn;
  Type resultTy = getResult()->getType();
  TensorType tensorResultTy = resultTy.cast<TensorType>();

  unsigned op_volume = getTensorVolume(tensorResultTy);
  toReturn["activation_in"] = op_volume;
  toReturn["activation_out"] = op_volume;
  toReturn[">"] = op_volume;

  return(toReturn);
}

} // namespace aten
} // namespace xilinx