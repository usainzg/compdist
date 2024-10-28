#include "lib/Transform/Affine/AffineDistributeToMPI.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_DEF_AFFINEDISTRIBUTETOMPI
#include "lib/Transform/Affine/Passes.h.inc"

using mlir::affine::AffineForOp;

// A pass that manually walks the IR
struct AffineDistributeToMPI
    : impl::AffineDistributeToMPIBase<AffineDistributeToMPI> {
  using AffineDistributeToMPIBase::AffineDistributeToMPIBase;

  // TODO: change to work with funcOp instead of affineForOp?
  void runOnOperation() {
    getOperation()->walk([&](AffineForOp op) {
      OpBuilder builder(op.getContext());

      builder.setInsertionPoint(op);

      // add mpi init
      auto retvalType = builder.getType<mpi::RetvalType>();
      auto initOp = builder.create<mpi::InitOp>(op.getLoc(), retvalType);

      // get mpi rank
      auto i32Type = builder.getI32Type();
      auto rankOp =
          builder.create<mpi::CommRankOp>(op.getLoc(), retvalType, i32Type);

      // create constants
      auto c0 = builder.create<arith::ConstantOp>(op.getLoc(), i32Type,
                                                  builder.getI32IntegerAttr(0));
      auto c1 = builder.create<arith::ConstantOp>(op.getLoc(), i32Type,
                                                  builder.getI32IntegerAttr(1));

      // create comparison for rank
      auto cmpOp = builder.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::eq, rankOp.getRank(), c0);

      // create if-else structure
      auto ifOp = builder.create<scf::IfOp>(op.getLoc(), cmpOp);

      // process rank 0
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      processRankZero(builder, op, c1, c0);

      // process rank 1
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      processRankOne(builder, op);

      // remove original loop
      op.erase();
    });
  }

  void processRankZero(OpBuilder &builder, affine::AffineForOp forOp,
                       Value dest, Value tag) {
    // send first half of data to rank 1
    auto loc = forOp.getLoc();
    auto retvalType = builder.getType<mpi::RetvalType>();
    auto i32Type = builder.getI32Type();

    // TODO: for (auto arg : funcOp.getArguments()) { mpi_send }

    // send first half of data to rank 1
    // get all memref operands from the loop body
    SmallVector<Value, 4> memrefOperands;
    forOp.walk([&](Operation *op) {
      if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
        if (!llvm::is_contained(memrefOperands, loadOp.getMemref()))
          memrefOperands.push_back(loadOp.getMemref());
      }
      if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
        if (!llvm::is_contained(memrefOperands, storeOp.getMemref()))
          memrefOperands.push_back(storeOp.getMemref());
      }
    });

    // send each memref to rank 1
    for (auto memref : memrefOperands) {
      builder.create<mpi::SendOp>(loc, retvalType, memref, dest, tag);
    }

    // create affine loop for the second half
    auto upperBound = forOp.getUpperBound();
    auto lowerBound = getHalfPoint(builder, forOp);

    // insert new loop
    
    // receive processed first half
    // only receive the result memref (assumed to be the last operand)
    if (!memrefOperands.empty()) {
      auto resultMemref = memrefOperands.back();
      builder.create<mpi::RecvOp>(loc, retvalType, resultMemref, dest, tag);
    }
  }

  void processRankOne(OpBuilder &builder, affine::AffineForOp forOp) {
    // allocate local buffers
    // receive data from rank 0
    // create affine loop for the first half
    // send result back to rank 0
    // cleanup local buffers (memrefs dealloc)
  }

  // helper function to get the midpoint of the loop range
  AffineMap getHalfPoint(OpBuilder &builder, AffineForOp forOp) {
    auto context = builder.getContext();
    auto upperMap = forOp.getUpperBoundMap();
    auto upperBound = upperMap.getResult(0);

    // create an affine map that divides the upper bound by 2
    auto halfExpr = upperBound.floorDiv(2);
    return AffineMap::get(upperMap.getNumDims(), upperMap.getNumSymbols(),
                          halfExpr, context);
  }
};

} // namespace tutorial
} // namespace mlir
