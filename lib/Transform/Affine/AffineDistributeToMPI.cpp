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

  // NOTE: change to work with funcOp instead of affineForOp?
  // -> not necessary?
  void runOnOperation() {
    // print number of ranks
    llvm::errs() << "n_ranks=" << n_ranks << "\n";

    // capture affineForOp and walk the IR
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

      // create constants for 0 and 1
      // NOTE:here create constants for all the ranks that we have
      // e.g., n ranks -> 0, 1, 2, ..., n-1 constants?
      auto c0 = builder.create<arith::ConstantOp>(op.getLoc(), i32Type,
                                                  builder.getI32IntegerAttr(0));
      auto c1 = builder.create<arith::ConstantOp>(op.getLoc(), i32Type,
                                                  builder.getI32IntegerAttr(1));

      // create comparison for rank
      auto cmpOp = builder.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::eq, rankOp.getRank(), c0);

      // create if-else structure
      // NOTE:the last boolean param represents withElseBlock
      auto ifOp = builder.create<scf::IfOp>(op.getLoc(), cmpOp, true);

      // process rank 0
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      processRankZero(builder, op, c1, c0);

      // process rank 1
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      processRankOne(builder, op, c0, c1);

      // remove original loop
      op.erase();
    });
  }

  void processRankZero(OpBuilder &builder, affine::AffineForOp forOp,
                       Value dest, Value tag) {
    auto loc = forOp.getLoc();
    auto retvalType = builder.getType<mpi::RetvalType>();

    // NOTE: for (auto arg : funcOp.getArguments()) { mpi_send }

    // get all memref operands from the loop body
    // TODO: only send what is used by the other node?
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
    // new bound for the new loop
    auto upperBoundMap = forOp.getUpperBoundMap();
    auto upperBoundOperands = forOp.getUpperBoundOperands();
    auto lowerBoundMap = getHalfPoint(builder, forOp);
    auto lowerBoundOperands = forOp.getLowerBoundOperands();

    // insert new loop
    auto newLoop = builder.create<affine::AffineForOp>(
        loc, lowerBoundOperands, lowerBoundMap, upperBoundOperands,
        upperBoundMap);

    // clone the original loop body into the new loop
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), newLoop.getInductionVar());

    // get the original loop body
    Block &originalBody = forOp.getRegion().front();

    // clone operations from original body to new loop body, excluding the
    // terminator
    builder.setInsertionPointToStart(newLoop.getBody());
    for (auto &op : originalBody.without_terminator()) {
      builder.clone(op, mapping);
    }

    // only receive the result memref (assumed to be the last operand)
    builder.setInsertionPointAfter(newLoop);
    if (!memrefOperands.empty()) {
      auto resultMemref = memrefOperands.back();
      builder.create<mpi::RecvOp>(loc, retvalType, resultMemref, dest, tag);
    }
  }

  void processRankOne(OpBuilder &builder, affine::AffineForOp forOp, Value dest,
                      Value tag) {
    auto loc = forOp.getLoc();
    auto retvalType = builder.getType<mpi::RetvalType>();

    // collect all memref operands from the loop body
    SmallVector<Value, 4> memrefOperands;
    SmallVector<Value, 4> allocatedMemrefs;
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

    // allocate local buffers with same types as original memrefs
    for (auto memref : memrefOperands) {
      auto memrefType = mlir::cast<MemRefType>(memref.getType());
      auto allocated = builder.create<memref::AllocOp>(loc, memrefType);
      allocatedMemrefs.push_back(allocated);
    }

    // receive data from rank 0
    for (auto localMemref : allocatedMemrefs) {
      builder.create<mpi::RecvOp>(loc, retvalType, localMemref, dest, tag);
    }

    // get bounds for the first half
    auto upperBoundMap = getHalfPoint(builder, forOp);
    auto upperBoundOperands = forOp.getUpperBoundOperands();
    auto lowerBoundMap = forOp.getLowerBoundMap();
    auto lowerBoundOperands = forOp.getLowerBoundOperands();

    // create affine loop for the first half
    auto newLoop = builder.create<affine::AffineForOp>(
        loc, lowerBoundOperands, lowerBoundMap, upperBoundOperands,
        upperBoundMap);

    // clone the original loop body into the new loop
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), newLoop.getInductionVar());

    // map the original memrefs to the local allocated ones
    for (auto [origMemref, localMemref] :
         llvm::zip(memrefOperands, allocatedMemrefs)) {
      mapping.map(origMemref, localMemref);
    }

    // clone operations from original body to new loop body
    builder.setInsertionPointToStart(newLoop.getBody());
    Block &originalBody = forOp.getRegion().front();
    for (auto &op : originalBody.without_terminator()) {
      builder.clone(op, mapping);
    }

    // send result back to rank 0
    builder.setInsertionPointAfter(newLoop);
    if (!allocatedMemrefs.empty()) {
      auto resultMemref = allocatedMemrefs.back();
      builder.create<mpi::SendOp>(loc, retvalType, resultMemref, dest, tag);
    }

    // cleanup: deallocate all local buffers
    for (auto memref : allocatedMemrefs) {
      builder.create<memref::DeallocOp>(loc, memref);
    }
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
