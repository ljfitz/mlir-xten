#ifndef AIR_OPS_SPLITTER
#define AIR_OPS_SPLITTER

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include "mlir/IR/PatternMatch.h"

#include "AIRDialect.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"


// NOTE this could be merged with the ops directly possibly, but would need to both NPCOMP and AIR
// But we will need the build function here anyway as each op will have different arguments and we
// don't want that in the pattern themselves
// So then not a big addition at the moment, especially with the NPCOMP dependency

// NOTE this may also be an easy to use that on top of ONNX or ATen transparently

// TODO whenever we generate something it will always be generating the partial version of the corresponding convolution?

namespace xilinx {
    namespace air {

        class AbsOpWrapper {
        public:
            virtual ~AbsOpWrapper() = 0;
            virtual Operation* getUnderlyingOperation() = 0;
            virtual Value getWeights() = 0;
            virtual Value getInput() = 0;
            virtual Value getBiases() = 0;
            virtual bool hasWeights() = 0;
            virtual bool isDepthWise() = 0;
            //virtual bool hasFusedBN(); // TODO
            //virtual Value getBNWeights();
            //virtual Value getBNBias();
            virtual Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                       llvm::Optional<Value> bias, llvm::Optional<Value> partialIn, bool firstInPartialChain) = 0;
        };

        class Conv2dOpWrapper : public AbsOpWrapper {
        private:
            Conv2dOp conv;
        public:
            Conv2dOpWrapper(Conv2dOp c);
            ~Conv2dOpWrapper();
            Operation* getUnderlyingOperation();
            Value getWeights();
            Value getInput();
            Value getBiases();
            bool hasWeights();
            bool isDepthWise();
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                               llvm::Optional<Value> bias,llvm::Optional<Value> partialIn, bool firstInPartialChain);
        };

        class PartialConv2dOpWrapper : public AbsOpWrapper {
        private:
            PartialConv2dOp conv;
        public:
            PartialConv2dOpWrapper(PartialConv2dOp c);
            ~PartialConv2dOpWrapper();
            Operation* getUnderlyingOperation();
            Value getWeights();
            Value getInput();
            Value getBiases();
            bool hasWeights();
            bool isDepthWise();
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                               llvm::Optional<Value> bias,llvm::Optional<Value> partialIn, bool firstInPartialChain);
        };

        class Conv2dReLUOpWrapper : public AbsOpWrapper {
        private:
            Conv2dReLUOp conv;
        public:
            Conv2dReLUOpWrapper(Conv2dReLUOp c);
            ~Conv2dReLUOpWrapper();
            Operation* getUnderlyingOperation();
            Value getWeights();
            Value getInput();
            Value getBiases();
            bool hasWeights();
            bool isDepthWise();
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                               llvm::Optional<Value> bias,llvm::Optional<Value> partialIn, bool firstInPartialChain);
        };

        class PartialConv2dReLUOpWrapper : public AbsOpWrapper {
        private:
            PartialConv2dReLUOp conv;
        public:
            PartialConv2dReLUOpWrapper(PartialConv2dReLUOp c);
            ~PartialConv2dReLUOpWrapper();
            Operation* getUnderlyingOperation();
            Value getWeights();
            Value getInput();
            Value getBiases();
            bool hasWeights();
            bool isDepthWise();
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                               llvm::Optional<Value> bias,llvm::Optional<Value> partialIn, bool firstInPartialChain);
        };

        class MaxPool2dWithIndicesOpWrapper : public AbsOpWrapper {
        private:
            mlir::NPCOMP::aten::MaxPool2dWithIndicesOp maxpool;
        public:
            MaxPool2dWithIndicesOpWrapper(mlir::NPCOMP::aten::MaxPool2dWithIndicesOp c);
            ~MaxPool2dWithIndicesOpWrapper();
            Operation* getUnderlyingOperation();
            Value getWeights();
            Value getInput();
            Value getBiases();
            bool hasWeights();
            bool isDepthWise();
            Operation* buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                               llvm::Optional<Value> bias,llvm::Optional<Value> partialIn, bool firstInPartialChain);
        };

    }
}

#endif
