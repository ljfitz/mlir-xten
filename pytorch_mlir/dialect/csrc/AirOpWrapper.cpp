#include "AirOpWrapper.h"
#include "AirDataflowConsts.h"

#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"

namespace xilinx {
    namespace air {

        void unpack_int_list(const Value &op, std::vector<int64_t> &v) {
            if (auto co = op.getDefiningOp<NPCOMP::aten::ConstantOp>()) {
                DenseElementsAttr a = co->template getAttrOfType<DenseElementsAttr>("value");
                for (auto i : a.getIntValues())
                    v.push_back(i.getSExtValue());
            }
            else if (auto co = op.getDefiningOp<NPCOMP::Basicpy::BuildListOp>()) {
                for (auto o : op.getDefiningOp()->getOperands())
                    v.push_back(o.template getDefiningOp<ConstantIntOp>().getValue());
            }
        }

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

        ArrayRef<Value> Conv2dOpWrapper::getBN() {
            return ArrayRef<Value>();
        }

        Value Conv2dOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value Conv2dOpWrapper::getInput() {
            return this->conv.input();
        }

        Value Conv2dOpWrapper::getPartialInput() {
            return Value();
        }

        unsigned int Conv2dOpWrapper::getKernelSize() {
            return this->conv.weight().getType().dyn_cast<ShapedType>().getShape()[F0_LOC];
        }

        bool Conv2dOpWrapper::hasWeights() {
            return true;
        }

        bool Conv2dOpWrapper::hasBN() {
            return false;
        }

        bool Conv2dOpWrapper::isDepthWise() {
            unsigned int groups = this->conv.groups().getDefiningOp<ConstantIntOp>().getValue();
            ShapedType aShape = this->conv.input().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();

            int64_t C = aShapeAR[C_LOC];

            return groups != C;
        }

        double Conv2dOpWrapper::getKernelEfficiency(AbsArchitecture* arch) {
            if(dynamic_cast<AIEv1*>(arch)) {
                return 0.95;
            } else {
                return 0.8; // If not traced then return estimate
            }
        }

        Operation* Conv2dOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                            llvm::Optional<Value> bias, llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                            llvm::Optional<ArrayRef<Value>> bn) {
            assert(weight.hasValue());
            assert(bias.hasValue());

            Operation* op = this->getUnderlyingOperation();
            if(firstInPartialChain || partialIn.hasValue()) {
                Value chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
                Operation* nOp =  builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
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

                nOp->setAttrs(op->getAttrs());

                return nOp;
            } else {
                Operation* nOp = builder.create<Conv2dOp>(builder.getUnknownLoc(),
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

                nOp->setAttrs(op->getAttrs());

                return nOp;
            }
        }

        Operation* Conv2dOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) {
            Operation* op;

            if(resTypes.hasValue()) {
                op =  builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
                                                      resTypes.getValue(),
                                                      this->getInput(),
                                                      Value(),
                                                      this->getWeights(),
                                                      this->getBiases(),
                                                      this->conv.stride(),
                                                      this->conv.padding(),
                                                      this->conv.dilation(),
                                                      this->conv.transposed(),
                                                      this->conv.output_padding(),
                                                      this->conv.groups());

            } else {
                op = builder.create<Conv2dOp>(builder.getUnknownLoc(),
                                              this->getUnderlyingOperation()->getResultTypes(),
                                              this->getInput(),
                                              this->getWeights(),
                                              this->getBiases(),
                                              this->conv.stride(),
                                              this->conv.padding(),
                                              this->conv.dilation(),
                                              this->conv.transposed(),
                                              this->conv.output_padding(),
                                              this->conv.groups());
            }

            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
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

        ArrayRef<Value> PartialConv2dOpWrapper::getBN() {
            return ArrayRef<Value>();
        }

        Value PartialConv2dOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value PartialConv2dOpWrapper::getInput() {
            return this->conv.input();
        }

        Value PartialConv2dOpWrapper::getPartialInput() {
            return this->conv.PartialIn();
        }

        unsigned int PartialConv2dOpWrapper::getKernelSize() {
            return this->conv.weight().getType().dyn_cast<ShapedType>().getShape()[F0_LOC];
        }

        bool PartialConv2dOpWrapper::hasWeights() {
            return true;
        }

        bool PartialConv2dOpWrapper::hasBN() {
            return false;
        }

        bool PartialConv2dOpWrapper::isDepthWise() {
            unsigned int groups = this->conv.groups().getDefiningOp<ConstantIntOp>().getValue();
            ShapedType aShape = this->conv.input().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();

            int64_t C = aShapeAR[C_LOC];

            return groups != C;
        }

        double PartialConv2dOpWrapper::getKernelEfficiency(AbsArchitecture* arch) {
            if(dynamic_cast<AIEv1*>(arch)) {
                return 0.95;
            } else {
                return 0.8; // If not traced then return estimate
            }
        }

        Operation* PartialConv2dOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                                   llvm::Optional<Value> bias, llvm::Optional<Value> partialIn,
                                                   bool firstInPartialChain, llvm::Optional<ArrayRef<Value>> bn) {
            assert(weight.hasValue());
            assert(bias.hasValue());

            Value chainIn;
            if(this->conv.PartialIn()) {
                assert(!partialIn.hasValue());
                chainIn = this->conv.PartialIn();
            } else {
                chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
            }

            Operation* op = this->getUnderlyingOperation();

            Operation* nOp =  builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
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

            nOp->setAttrs(op->getAttrs());
            return nOp;
        }

        Operation* PartialConv2dOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) {
            Operation* op;

            if(resTypes.hasValue()) {
                op = builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
                                                     resTypes.getValue(),
                                                     this->getInput(),
                                                     this->conv.PartialIn(),
                                                     this->getWeights(),
                                                     this->getBiases(),
                                                     this->conv.stride(),
                                                     this->conv.padding(),
                                                     this->conv.dilation(),
                                                     this->conv.transposed(),
                                                     this->conv.output_padding(),
                                                     this->conv.groups());
            } else {
                op = builder.create<PartialConv2dOp>(builder.getUnknownLoc(),
                                                     this->getUnderlyingOperation()->getResultTypes(),
                                                     this->getInput(),
                                                     this->conv.PartialIn(),
                                                     this->getWeights(),
                                                     this->getBiases(),
                                                     this->conv.stride(),
                                                     this->conv.padding(),
                                                     this->conv.dilation(),
                                                     this->conv.transposed(),
                                                     this->conv.output_padding(),
                                                     this->conv.groups());
            }


            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
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

        ArrayRef<Value> Conv2dReLUOpWrapper::getBN() {
            return ArrayRef<Value>();
        }

        Value Conv2dReLUOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value Conv2dReLUOpWrapper::getInput() {
            return this->conv.input();
        }

        Value Conv2dReLUOpWrapper::getPartialInput() {
            return Value();
        }

        unsigned int Conv2dReLUOpWrapper::getKernelSize() {
            return this->conv.weight().getType().dyn_cast<ShapedType>().getShape()[F0_LOC];
        }

        bool Conv2dReLUOpWrapper::hasWeights() {
            return true;
        }

        bool Conv2dReLUOpWrapper::hasBN() {
            return false;
        }

        bool Conv2dReLUOpWrapper::isDepthWise() {
            unsigned int groups = this->conv.groups().getDefiningOp<ConstantIntOp>().getValue();
            ShapedType aShape = this->conv.input().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();

            int64_t C = aShapeAR[C_LOC];

            return groups != C;
        }

        double Conv2dReLUOpWrapper::getKernelEfficiency(AbsArchitecture* arch) {
            if(dynamic_cast<AIEv1*>(arch)) {
                return 0.95;
            } else {
                return 0.8; // If not traced then return estimate
            }
        }

        Operation* Conv2dReLUOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input, llvm::Optional<Value> weight,
                                                llvm::Optional<Value> bias, llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                                llvm::Optional<ArrayRef<Value>> bn) {
            assert(weight.hasValue());
            assert(bias.hasValue());

            Operation* op = this->getUnderlyingOperation();

            if(firstInPartialChain || partialIn.hasValue()) {
                Value chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
                Operation* nOp =  builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
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

                nOp->setAttrs(op->getAttrs());
                return nOp;
            } else {
                Operation* nOp = builder.create<Conv2dReLUOp>(builder.getUnknownLoc(),
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

                nOp->setAttrs(op->getAttrs());
                return nOp;
            }
        }

        Operation* Conv2dReLUOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) {
            Operation* op;

            if(resTypes.hasValue()) {
                op =  builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                          resTypes.getValue(),
                                                          this->getInput(),
                                                          Value(),
                                                          this->getWeights(),
                                                          this->getBiases(),
                                                          this->conv.stride(),
                                                          this->conv.padding(),
                                                          this->conv.dilation(),
                                                          this->conv.transposed(),
                                                          this->conv.output_padding(),
                                                          this->conv.groups());
            } else {
                op = builder.create<Conv2dReLUOp>(builder.getUnknownLoc(),
                                                  this->getUnderlyingOperation()->getResultTypes(),
                                                  this->getInput(),
                                                  this->getWeights(),
                                                  this->getBiases(),
                                                  this->conv.stride(),
                                                  this->conv.padding(),
                                                  this->conv.dilation(),
                                                  this->conv.transposed(),
                                                  this->conv.output_padding(),
                                                  this->conv.groups());
            }


            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
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

        ArrayRef<Value> PartialConv2dReLUOpWrapper::getBN() {
            return ArrayRef<Value>();
        }

        Value PartialConv2dReLUOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value PartialConv2dReLUOpWrapper::getInput() {
            return this->conv.input();
        }

        Value PartialConv2dReLUOpWrapper::getPartialInput() {
            return this->conv.PartialIn();
        }

        unsigned int PartialConv2dReLUOpWrapper::getKernelSize() {
            return this->conv.weight().getType().dyn_cast<ShapedType>().getShape()[F0_LOC];
        }

        bool PartialConv2dReLUOpWrapper::hasWeights() {
            return true;
        }

        bool PartialConv2dReLUOpWrapper::hasBN() {
            return false;
        }

        bool PartialConv2dReLUOpWrapper::isDepthWise() {
            unsigned int groups = this->conv.groups().getDefiningOp<ConstantOp>();
            ShapedType aShape = this->conv.input().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();

            int64_t C = aShapeAR[C_LOC];

            return groups != C;
        }

        double PartialConv2dReLUOpWrapper::getKernelEfficiency(AbsArchitecture* arch) {
            if(dynamic_cast<AIEv1*>(arch)) {
                return 0.95;
            } else {
                return 0.8; // If not traced then return estimate
            }
        }

        Operation* PartialConv2dReLUOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input,
                                                       llvm::Optional<Value> weight, llvm::Optional<Value> bias,
                                                       llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                                       llvm::Optional<ArrayRef<Value>> bn) {
            assert(weight.hasValue());
            assert(bias.hasValue());

            Value chainIn;
            if(this->conv.PartialIn()) {
                assert(!partialIn.hasValue());
                chainIn = this->conv.PartialIn();
            } else {
                chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
            }

            Operation* op = this->getUnderlyingOperation();
            Operation* nOp = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
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
            nOp->setAttrs(op->getAttrs());
            return nOp;
        }

        Operation* PartialConv2dReLUOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) {
            Operation* op;

            if(resTypes.hasValue()) {
                op = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                         resTypes.getValue(),
                                                         this->getInput(),
                                                         this->conv.PartialIn(),
                                                         this->getWeights(),
                                                         this->getBiases(),
                                                         this->conv.stride(),
                                                         this->conv.padding(),
                                                         this->conv.dilation(),
                                                         this->conv.transposed(),
                                                         this->conv.output_padding(),
                                                         this->conv.groups());
            } else {
                op = builder.create<PartialConv2dReLUOp>(builder.getUnknownLoc(),
                                                         this->getUnderlyingOperation()->getResultTypes(),
                                                         this->getInput(),
                                                         this->conv.PartialIn(),
                                                         this->getWeights(),
                                                         this->getBiases(),
                                                         this->conv.stride(),
                                                         this->conv.padding(),
                                                         this->conv.dilation(),
                                                         this->conv.transposed(),
                                                         this->conv.output_padding(),
                                                         this->conv.groups());
            }



            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
        }

        PartialConv2dBatchNormReLUOpWrapper::PartialConv2dBatchNormReLUOpWrapper(PartialConv2dBatchNormReLUOp c) {
            conv = c;
        }

        PartialConv2dBatchNormReLUOpWrapper::~PartialConv2dBatchNormReLUOpWrapper() {}

        Operation* PartialConv2dBatchNormReLUOpWrapper::getUnderlyingOperation() {
            return conv.getOperation();
        }

        Value PartialConv2dBatchNormReLUOpWrapper::getWeights() {
            return this->conv.weight();
        }

        ArrayRef<Value> PartialConv2dBatchNormReLUOpWrapper::getBN() {
            return ArrayRef<Value>({this->conv.bn_weight(), this->conv.bn_bias(), this->conv.running_mean(), this->conv.running_var()});
        }

        Value PartialConv2dBatchNormReLUOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value PartialConv2dBatchNormReLUOpWrapper::getInput() {
            return this->conv.input();
        }

        Value PartialConv2dBatchNormReLUOpWrapper::getPartialInput() {
            return this->conv.PartialIn();
        }

        unsigned int PartialConv2dBatchNormReLUOpWrapper::getKernelSize() {
            return this->conv.weight().getType().dyn_cast<ShapedType>().getShape()[F0_LOC];
        }

        bool PartialConv2dBatchNormReLUOpWrapper::hasWeights() {
            return true;
        }

        bool PartialConv2dBatchNormReLUOpWrapper::hasBN() {
            return true;
        }

        bool PartialConv2dBatchNormReLUOpWrapper::isDepthWise() {
            unsigned int groups = this->conv.groups().getDefiningOp<ConstantOp>();
            ShapedType aShape = this->conv.input().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();

            int64_t C = aShapeAR[C_LOC];

            return groups != C;
        }

        double PartialConv2dBatchNormReLUOpWrapper::getKernelEfficiency(AbsArchitecture* arch) {
            if(dynamic_cast<AIEv1*>(arch)) {
                return 0.95;
            } else {
                return 0.8; // If not traced then return estimate
            }
        }

        Operation* PartialConv2dBatchNormReLUOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input,
                                                                llvm::Optional<Value> weight, llvm::Optional<Value> bias,
                                                                llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                                                llvm::Optional<ArrayRef<Value>> bn) {
            assert(weight.hasValue());
            assert(bias.hasValue());
            assert(bn.hasValue());

            Value chainIn;
            if(this->conv.PartialIn()) {
                assert(!partialIn.hasValue());
                chainIn = this->conv.PartialIn();
            } else {
                chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
            }

            Operation* op = this->getUnderlyingOperation();
            Operation* nOp = builder.create<PartialConv2dBatchNormReLUOp>(builder.getUnknownLoc(),
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
                                                                          this->conv.groups(),
                                                                          bn.getValue()[0],
                                                                          bn.getValue()[1],
                                                                          bn.getValue()[2],
                                                                          bn.getValue()[3],
                                                                          this->conv.training(),
                                                                          this->conv.momentum(),
                                                                          this->conv.eps());

            nOp->setAttrs(op->getAttrs());
            return nOp;
        }

        Operation* PartialConv2dBatchNormReLUOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) {
            Operation* op;
            if(resTypes.hasValue()) {
                op = builder.create<PartialConv2dBatchNormReLUOp>(builder.getUnknownLoc(),
                                                                  resTypes.getValue(),
                                                                  this->getInput(),
                                                                  this->conv.PartialIn(),
                                                                  this->getWeights(),
                                                                  this->getBiases(),
                                                                  this->conv.stride(),
                                                                  this->conv.padding(),
                                                                  this->conv.dilation(),
                                                                  this->conv.transposed(),
                                                                  this->conv.output_padding(),
                                                                  this->conv.groups(),
                                                                  this->conv.bn_weight(),
                                                                  this->conv.bn_bias(),
                                                                  this->conv.running_mean(),
                                                                  this->conv.running_var(),
                                                                  this->conv.training(),
                                                                  this->conv.momentum(),
                                                                  this->conv.eps());
            } else {
                op = builder.create<PartialConv2dBatchNormReLUOp>(builder.getUnknownLoc(),
                                                                  this->getUnderlyingOperation()->getResultTypes(),
                                                                  this->getInput(),
                                                                  this->conv.PartialIn(),
                                                                  this->getWeights(),
                                                                  this->getBiases(),
                                                                  this->conv.stride(),
                                                                  this->conv.padding(),
                                                                  this->conv.dilation(),
                                                                  this->conv.transposed(),
                                                                  this->conv.output_padding(),
                                                                  this->conv.groups(),
                                                                  this->conv.bn_weight(),
                                                                  this->conv.bn_bias(),
                                                                  this->conv.running_mean(),
                                                                  this->conv.running_var(),
                                                                  this->conv.training(),
                                                                  this->conv.momentum(),
                                                                  this->conv.eps());
            }



            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            /*auto lines = op->getAttr("line").dyn_cast<ArrayAttr>().getValue();

            if(lines.size() == 1) {
                unsigned int lines0 = lines[0].dyn_cast<IntegerAttr>().getValue().getZExtValue();

                auto attr = builder.getI32ArrayAttr({static_cast<int>(lines0 + into)});
                op->setAttr(llvm::StringRef("line"), attr);
            } else {
                llvm::outs() << "Got line size of: " << lines.size() << "\n";
                unsigned int lines0 = lines[0].dyn_cast<IntegerAttr>().getValue().getZExtValue();
                unsigned int lines1 = lines[1].dyn_cast<IntegerAttr>().getValue().getZExtValue();

                auto attr = builder.getI32ArrayAttr({static_cast<int>(lines0 + into), static_cast<int>(lines1 + into)});
                op->setAttr(llvm::StringRef("line"), attr);
                }*/

            return op;
        }


        Conv2dBatchNormReLUOpWrapper::Conv2dBatchNormReLUOpWrapper(Conv2dBatchNormReLUOp c) {
            conv = c;
        }

        Conv2dBatchNormReLUOpWrapper::~Conv2dBatchNormReLUOpWrapper() {}

        Operation* Conv2dBatchNormReLUOpWrapper::getUnderlyingOperation() {
            return conv.getOperation();
        }

        Value Conv2dBatchNormReLUOpWrapper::getWeights() {
            return this->conv.weight();
        }

        ArrayRef<Value> Conv2dBatchNormReLUOpWrapper::getBN() {
            return ArrayRef<Value>({this->conv.bn_weight(), this->conv.bn_bias(), this->conv.running_mean(), this->conv.running_var()});
        }

        Value Conv2dBatchNormReLUOpWrapper::getBiases() {
            return this->conv.bias();
        }

        Value Conv2dBatchNormReLUOpWrapper::getInput() {
            return this->conv.input();
        }

        Value Conv2dBatchNormReLUOpWrapper::getPartialInput() {
            return Value();
        }

        unsigned int Conv2dBatchNormReLUOpWrapper::getKernelSize() {
            return this->conv.weight().getType().dyn_cast<ShapedType>().getShape()[F0_LOC];
        }

        bool Conv2dBatchNormReLUOpWrapper::hasWeights() {
            return true;
        }

        bool Conv2dBatchNormReLUOpWrapper::hasBN() {
            return true;
        }

        bool Conv2dBatchNormReLUOpWrapper::isDepthWise() {
            unsigned int groups = this->conv.groups().getDefiningOp<ConstantOp>();
            ShapedType aShape = this->conv.input().getType().dyn_cast<ShapedType>();
            ArrayRef<int64_t> aShapeAR = aShape.getShape();

            int64_t C = aShapeAR[C_LOC];

            return groups != C;
        }

        double Conv2dBatchNormReLUOpWrapper::getKernelEfficiency(AbsArchitecture* arch) {
            if(dynamic_cast<AIEv1*>(arch)) {
                return 0.95;
            } else {
                return 0.8; // If not traced then return estimate
            }
        }

        Operation* Conv2dBatchNormReLUOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input,
                                                                llvm::Optional<Value> weight, llvm::Optional<Value> bias,
                                                                llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                                                llvm::Optional<ArrayRef<Value>> bn) {
            assert(weight.hasValue());
            assert(bias.hasValue());
            assert(bn.hasValue());

            Operation* op = this->getUnderlyingOperation();
            if(firstInPartialChain || partialIn.hasValue()) {
                Value chainIn = (partialIn.hasValue()) ? partialIn.getValue() : Value();
                Operation* nOp =  builder.create<PartialConv2dBatchNormReLUOp>(builder.getUnknownLoc(),
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
                                                                               this->conv.groups(),
                                                                               bn.getValue()[0],
                                                                               bn.getValue()[1],
                                                                               bn.getValue()[2],
                                                                               bn.getValue()[3],
                                                                               this->conv.training(),
                                                                               this->conv.momentum(),
                                                                               this->conv.eps());
                nOp->setAttrs(op->getAttrs());

                return nOp;
            } else {
                Operation* nOp = builder.create<Conv2dBatchNormReLUOp>(builder.getUnknownLoc(),
                                                                       returnType,
                                                                       input,
                                                                       weight.getValue(),
                                                                       bias.getValue(),
                                                                       this->conv.stride(),
                                                                       this->conv.padding(),
                                                                       this->conv.dilation(),
                                                                       this->conv.transposed(),
                                                                       this->conv.output_padding(),
                                                                       this->conv.groups(),
                                                                       bn.getValue()[0],
                                                                       bn.getValue()[1],
                                                                       bn.getValue()[2],
                                                                       bn.getValue()[3],
                                                                       this->conv.training(),
                                                                       this->conv.momentum(),
                                                                       this->conv.eps());

                nOp->setAttrs(op->getAttrs());

                return nOp;
            }
        }

        Operation* Conv2dBatchNormReLUOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> resTypes) {
            Operation* op;

            if(resTypes.hasValue()) {
                op =  builder.create<PartialConv2dBatchNormReLUOp>(builder.getUnknownLoc(),
                                                                   resTypes.getValue(),
                                                                   this->getInput(),
                                                                   Value(),
                                                                   this->getWeights(),
                                                                   this->getBiases(),
                                                                   this->conv.stride(),
                                                                   this->conv.padding(),
                                                                   this->conv.dilation(),
                                                                   this->conv.transposed(),
                                                                   this->conv.output_padding(),
                                                                   this->conv.groups(),
                                                                   this->conv.bn_weight(),
                                                                   this->conv.bn_bias(),
                                                                   this->conv.running_mean(),
                                                                   this->conv.running_var(),
                                                                   this->conv.training(),
                                                                   this->conv.momentum(),
                                                                   this->conv.eps());
            } else {
                op = builder.create<Conv2dBatchNormReLUOp>(builder.getUnknownLoc(),
                                                           this->getUnderlyingOperation()->getResultTypes(),
                                                           this->getInput(),
                                                           this->getWeights(),
                                                           this->getBiases(),
                                                           this->conv.stride(),
                                                           this->conv.padding(),
                                                           this->conv.dilation(),
                                                           this->conv.transposed(),
                                                           this->conv.output_padding(),
                                                           this->conv.groups(),
                                                           this->conv.bn_weight(),
                                                           this->conv.bn_bias(),
                                                           this->conv.running_mean(),
                                                           this->conv.running_var(),
                                                           this->conv.training(),
                                                           this->conv.momentum(),
                                                           this->conv.eps());
            }



            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
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

        unsigned int MaxPool2dWithIndicesOpWrapper::getKernelSize() {
            Value ks = this->maxpool.kernel_size();
            std::vector<int64_t> kernel_size;
            unpack_int_list(ks, kernel_size);

            return kernel_size.at(0);
        }

        Value MaxPool2dWithIndicesOpWrapper::getInput() {
            return this->maxpool.self();
        }

        Value MaxPool2dWithIndicesOpWrapper::getPartialInput() {
            return Value();
        }

        ArrayRef<Value> MaxPool2dWithIndicesOpWrapper::getBN() {
            return ArrayRef<Value>();
        }

        bool MaxPool2dWithIndicesOpWrapper::hasWeights() {
            return false;
        }

        bool MaxPool2dWithIndicesOpWrapper::isDepthWise() {
            return true;
        }

        bool MaxPool2dWithIndicesOpWrapper::hasBN() {
            return false;
        }

        double MaxPool2dWithIndicesOpWrapper::getKernelEfficiency(AbsArchitecture* arch) {
            if(dynamic_cast<AIEv1*>(arch)) {
                return 0.15;
            } else {
                return 0.8; // If not traced then return estimate
            }
        }

        Operation* MaxPool2dWithIndicesOpWrapper::buildOp(OpBuilder &builder, TypeRange returnType, Value input,
                                                          llvm::Optional<Value> weight, llvm::Optional<Value> bias,
                                                          llvm::Optional<Value> partialIn, bool firstInPartialChain,
                                                          llvm::Optional<ArrayRef<Value>> bn) {
            assert(!weight.hasValue());
            assert(!bias.hasValue());
            assert(!firstInPartialChain);
            assert(!partialIn.hasValue());

            Operation* op = this->getUnderlyingOperation();
            Operation* nOp =  builder.create<NPCOMP::aten::MaxPool2dWithIndicesOp>(builder.getUnknownLoc(), returnType, input,
                                                                                   this->maxpool.kernel_size(),
                                                                                   this->maxpool.stride(),
                                                                                   this->maxpool.padding(),
                                                                                   this->maxpool.dilation(),
                                                                                   this->maxpool.ceil_mode());

            nOp->setAttrs(op->getAttrs());
            return nOp;
        }

        Operation* MaxPool2dWithIndicesOpWrapper::wCopy(OpBuilder &builder, unsigned int into, llvm::Optional<TypeRange> typeRes) {
            assert(!typeRes.hasValue());

            Operation* op =  builder.create<NPCOMP::aten::MaxPool2dWithIndicesOp>(builder.getUnknownLoc(),
                                                                                  this->getUnderlyingOperation()->getResultTypes(),
                                                                                  this->getInput(),
                                                                                  this->maxpool.kernel_size(),
                                                                                  this->maxpool.stride(),
                                                                                  this->maxpool.padding(),
                                                                                  this->maxpool.dilation(),
                                                                                  this->maxpool.ceil_mode());

            op->setAttrs(this->getUnderlyingOperation()->getAttrs());

            auto ty = IntegerType::get(builder.getContext(), 32);
            auto attr = IntegerAttr::get(ty, into);
            op->setAttr(llvm::StringRef("locW"), attr);

            return op;
        }

    }
}

