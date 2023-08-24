/// Implements the DLNN dialect enclave interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#include "dlnn-mlir/Dialect/DLNN/Interfaces/EnclaveOpInterface.h"

#include "dlnn-mlir/Dialect/DLNN/IR/DLNN.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/STLExtras.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include <set>

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;

void mlir::dlnn::enclave_interface_defaults::populateCaptureMap(
    Operation* op,
    BlockAndValueMapping &result)
{
    auto self = cast<EnclaveOp>(op);
    auto captures = self.getCaptures();
    auto operands =
        MutableArrayRef<OpOperand>(captures.getBase(), captures.size());

    for (auto &capture : operands)
        result.map(capture.get(), self.lookupArgument(capture));
}

void mlir::dlnn::enclave_interface_defaults::capture(
    Operation* op,
    ValueRange values,
    BlockAndValueMapping &result)
{
    auto self = cast<EnclaveOp>(op);
    auto &body = self.getEnclaveBody();
    auto unknownLoc = UnknownLoc::get(op->getContext());

    for (auto value : values) {
        // Try to find an existing capture.
        const auto capture = find_if(op->getOpOperands(), [&](auto &op) {
            return op.get() == value;
        });
        if (capture != op->getOpOperands().end()) {
            // Return the existing capture.
            result.map(value, self.lookupArgument(*capture));
            continue;
        }

        assert(
            !body.getParent()->isAncestor(value.getParentRegion())
            && "capture value declared inside enclave");

        // Append both an operand and a block argument.
        op->insertOperands(op->getNumOperands(), ValueRange(value));
        result.map(
            value,
            body.insertArgument(
                body.getNumArguments(),
                value.getType(),
                unknownLoc));
    }
}

void mlir::dlnn::enclave_interface_defaults::uncapture(
    Operation* op,
    ArrayRef<BlockArgument> args)
{
    auto self = cast<EnclaveOp>(op);
    auto &body = self.getEnclaveBody();

    // Use a small vector as a sorted set.
    SmallVector<unsigned> indices;
    const auto insert_index = [&](unsigned idx) {
        indices.insert(
            std::lower_bound(indices.begin(), indices.end(), idx),
            idx);
    };

    // Find all unique argument indices to erase and order them.
    for (auto arg : args) {
        assert(arg.getOwner() == &body && "unrelated block argument");
        assert(arg.use_empty() && "capture still has uses");

        insert_index(arg.getArgNumber());
    }

    for (auto idx : reverse(indices)) {
        // Erase both the operand and the block argument.
        op->eraseOperand(idx);
        body.eraseArgument(idx);
    }
}

LogicalResult mlir::dlnn::enclave_interface_defaults::verify(Operation* op)
{
    auto self = cast<EnclaveOp>(op);

    if (self.getEnclaveBody().empty()
        || !isRegionReturnLike(&self.getEnclaveBody().back()))
        return op->emitOpError() << "enclave is missing terminator";
    auto terminator = self.getTerminator();

    // The number of results must be equal to the enclave.
    if (terminator->getNumOperands() != op->getNumResults())
        return terminator->emitOpError()
               << "number of operands (" << terminator->getNumResults()
               << ") does not match number of enclave results ("
               << op->getNumResults() << ")";

    // The types of the results must match.
    for (auto [idx, resTy] : enumerate(op->getResultTypes()))
        if (terminator->getOperandTypes()[idx] != resTy)
            return terminator->emitOpError()
                   << "type of operand #" << idx << " ("
                   << terminator->getOperandTypes()[idx]
                   << ") does not match enclave result type (" << resTy << ")";

    return success();
}

//===- Generated implementation -------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/Interfaces/EnclaveOpInterface.cpp.inc"

//===----------------------------------------------------------------------===//
