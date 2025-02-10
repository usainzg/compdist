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
      auto mpiRetvalType = builder.getType<mpi::RetvalType>();
      auto mpiInitOp = builder.create<mpi::InitOp>(op.getLoc(), mpiRetvalType);

      // get mpi rank
      auto i32Type = builder.getI32Type();
      auto mpiRankOp =
          builder.create<mpi::CommRankOp>(op.getLoc(), mpiRetvalType, i32Type);

      // create constants for 0 and 1
      // NOTE:here create constants for all the ranks that we have
      // e.g., n ranks -> 0, 1, 2, ..., n-1 constants?
      auto c0 = builder.create<arith::ConstantOp>(op.getLoc(), i32Type,
                                                  builder.getI32IntegerAttr(0));
      auto c1 = builder.create<arith::ConstantOp>(op.getLoc(), i32Type,
                                                  builder.getI32IntegerAttr(1));

      // create comparison for rank
      auto cmpOp = builder.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::eq, mpiRankOp.getRank(), c0);

      // create if-else structure
      // NOTE:the last boolean param represents withElseBlock
      auto ifOp = builder.create<scf::IfOp>(op.getLoc(), cmpOp, true);

      // process mpi rank 0 (master)
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      
      // process other mpi ranks
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

      // remove original loop
      op.erase();
    });
  }

  // NOTE:this function creates subviews using the memref
  Value createSubview(Value memref, Value offset, Value size,
                      OpBuilder &builder, Location loc) {
    // Create a subview of the memref for the current rank's chunk
    SmallVector<OpFoldResult, 1> offsets{builder.getIndexAttr(0)};
    SmallVector<OpFoldResult, 1> sizes{size};
    SmallVector<OpFoldResult, 1> strides{builder.getIndexAttr(1)};

    return builder.create<memref::SubViewOp>(loc, memref, offsets, sizes,
                                             strides);
  }

  // helper function to get the midpoint of the loop range
  AffineMap getHalfPoint(OpBuilder &builder, AffineForOp forOp,
                         bool isUpper = false) {
    auto context = builder.getContext();
    auto boundMap =
        isUpper ? forOp.getUpperBoundMap() : forOp.getLowerBoundMap();
    auto bound = boundMap.getResult(0);

    // create an affine map that divides the bound by 2
    auto halfExpr = bound.floorDiv(2);
    return AffineMap::get(boundMap.getNumDims(), boundMap.getNumSymbols(),
                          halfExpr, context);
  }
};

} // namespace tutorial
} // namespace mlir
