#include "AirOpWrapper.h"

namespace xilinx {
    namespace air {
        AbsOpWrapper::~AbsOpWrapper() {}

        Conv2dOpWrapper::Conv2dOpWrapper(Conv2dOp c) {
            conv = c;
        }

        Conv2dOpWrapper::~Conv2dOpWrapper() {}

        Operation* Conv2dOpWrapper::getUnderlyingOperation() {
            return conv.getOperation();
        }

        Value Conv2dOpWrapper::getWeights() {
            return this->conv.weight();
        }

        Value Conv2dOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value Conv2dOpWrapper::getInput() {
            return this->conv.input();
        }

        bool Conv2dOpWrapper::hasWeights() {
            return true;
        }

        Operation* Conv2dOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                            llvm::Optional<Value> bias, llvm::Optional<Value> partialIn, bool firstInPartialChain) {
            assert(weight.hasValue());
            assert(bias.hasValue());

            if(firstInPartialChain || partialIn.hasValue()) {
                Value chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
                return builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
                                                       returnType,
                                                       input,
                                                       chainIn,
                                                       weight.getValue(),
                                                       bias.getValue(),
                                                       this->conv.stride(),
                                                       this->conv.padding(),
                                                       this->conv.dilation(),
                                                       this->conv.transposed(),
                                                       this->conv.output_padding(),
                                                       this->conv.groups());
            } else {
                return builder.create<Conv2dOp>(builder.getUnknownLoc(),
                                                returnType,
                                                input,
                                                weight.getValue(),
                                                bias.getValue(),
                                                this->conv.stride(),
                                                this->conv.padding(),
                                                this->conv.dilation(),
                                                this->conv.transposed(),
                                                this->conv.output_padding(),
                                                this->conv.groups());
            }
        }

        PartialConv2dOpWrapper::PartialConv2dOpWrapper(PartialConv2dOp c) {
            conv = c;
        }

        PartialConv2dOpWrapper::~PartialConv2dOpWrapper() {}

        Operation* PartialConv2dOpWrapper::getUnderlyingOperation() {
            return conv.getOperation();
        }

        Value PartialConv2dOpWrapper::getWeights() {
            return this->conv.weight();
        }

        Value PartialConv2dOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value PartialConv2dOpWrapper::getInput() {
            return this->conv.input();
        }

        bool PartialConv2dOpWrapper::hasWeights() {
            return true;
        }

        Operation* PartialConv2dOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                                   llvm::Optional<Value> bias, llvm::Optional<Value> partialIn, bool firstInPartialChain) {
            assert(weight.hasValue());
            assert(bias.hasValue());

            Value chainIn;
            if(this->conv.PartialIn()) {
                assert(!partialIn.hasValue());
                chainIn = this->conv.PartialIn();
            } else {
                chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
            }

            return builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
                                                   returnType,
                                                   input,
                                                   chainIn,
                                                   weight.getValue(),
                                                   bias.getValue(),
                                                   this->conv.stride(),
                                                   this->conv.padding(),
                                                   this->conv.dilation(),
                                                   this->conv.transposed(),
                                                   this->conv.output_padding(),
                                                   this->conv.groups());
        }

        Conv2dReLUOpWrapper::Conv2dReLUOpWrapper(Conv2dReLUOp c) {
            conv = c;
        }

        Conv2dReLUOpWrapper::~Conv2dReLUOpWrapper() {}

        Operation* Conv2dReLUOpWrapper::getUnderlyingOperation() {
            return conv.getOperation();
        }

        Value Conv2dReLUOpWrapper::getWeights() {
            return this->conv.weight();
        }

        Value Conv2dReLUOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value Conv2dReLUOpWrapper::getInput() {
            return this->conv.input();
        }

        bool Conv2dReLUOpWrapper::hasWeights() {
            return true;
        }

        Operation* Conv2dReLUOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                                llvm::Optional<Value> bias, llvm::Optional<Value> partialIn, bool firstInPartialChain) {
            assert(weight.hasValue());
            assert(bias.hasValue());

            if(firstInPartialChain || partialIn.hasValue()) {
                Value chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
                return builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                           returnType,
                                                           input,
                                                           chainIn,
                                                           weight.getValue(),
                                                           bias.getValue(),
                                                           this->conv.stride(),
                                                           this->conv.padding(),
                                                           this->conv.dilation(),
                                                           this->conv.transposed(),
                                                           this->conv.output_padding(),
                                                           this->conv.groups());
            } else {
                return builder.create<Conv2dReLUOp>(builder.getUnknownLoc(),
                                                    returnType,
                                                    input,
                                                    weight.getValue(),
                                                    bias.getValue(),
                                                    this->conv.stride(),
                                                    this->conv.padding(),
                                                    this->conv.dilation(),
                                                    this->conv.transposed(),
                                                    this->conv.output_padding(),
                                                    this->conv.groups());
            }
        }

        PartialConv2dReLUOpWrapper::PartialConv2dReLUOpWrapper(PartialConv2dReLUOp c) {
            conv = c;
        }

        PartialConv2dReLUOpWrapper::~PartialConv2dReLUOpWrapper() {}

        Operation* PartialConv2dReLUOpWrapper::getUnderlyingOperation() {
            return conv.getOperation();
        }

        Value PartialConv2dReLUOpWrapper::getWeights() {
            return this->conv.weight();
        }

        Value PartialConv2dReLUOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value PartialConv2dReLUOpWrapper::getInput() {
            return this->conv.input();
        }

        bool PartialConv2dReLUOpWrapper::hasWeights() {
            return true;
        }

        Operation* PartialConv2dReLUOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input,
                                                       llvm::Optional<Value> weight, llvm::Optional<Value> bias,
                                                       llvm::Optional<Value> partialIn, bool firstInPartialChain) {
            assert(weight.hasValue());
            assert(bias.hasValue());

            Value chainIn;
            if(this->conv.PartialIn()) {
                assert(!partialIn.hasValue());
                chainIn = this->conv.PartialIn();
            } else {
                chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
            }

            return builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                       returnType,
                                                       input,
                                                       chainIn,
                                                       weight.getValue(),
                                                       bias.getValue(),
                                                       this->conv.stride(),
                                                       this->conv.padding(),
                                                       this->conv.dilation(),
                                                       this->conv.transposed(),
                                                       this->conv.output_padding(),
                                                       this->conv.groups());
        }


        MaxPool2dWithIndicesOpWrapper::MaxPool2dWithIndicesOpWrapper(mlir::NPCOMP::aten::MaxPool2dWithIndicesOp mp) {
            maxpool = mp;
        }

        MaxPool2dWithIndicesOpWrapper::~MaxPool2dWithIndicesOpWrapper() {}


        Operation* MaxPool2dWithIndicesOpWrapper::getUnderlyingOperation() {
            return maxpool.getOperation();
        }

        Value MaxPool2dWithIndicesOpWrapper::getWeights() {
            return Value();
        }

        Value MaxPool2dWithIndicesOpWrapper::getBiases() {
            return Value();
        }

        Value MaxPool2dWithIndicesOpWrapper::getInput() {
            return this->maxpool.self(); // TODO why self?!
        }

        bool MaxPool2dWithIndicesOpWrapper::hasWeights() {
            return false;
        }

        Operation* MaxPool2dWithIndicesOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input,
                                                          llvm::Optional<Value> weight, llvm::Optional<Value> bias,
                                                          llvm::Optional<Value> partialIn, bool firstInPartialChain) {
            assert(!weight.hasValue());
            assert(!bias.hasValue());
            assert(!firstInPartialChain);
            assert(!partialIn.hasValue());

            return builder.create<NPCOMP::aten::MaxPool2dWithIndicesOp>(builder.getUnknownLoc(), returnType, input,
                                                                        this->maxpool.kernel_size(),
                                                                        this->maxpool.stride(),
                                                                        this->maxpool.padding(),
                                                                        this->maxpool.dilation(),
                                                                        this->maxpool.ceil_mode());
        }
    }
}



