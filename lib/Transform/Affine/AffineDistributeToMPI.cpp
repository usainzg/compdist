#include "lib/Transform/Affine/AffineDistributeToMPI.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
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
      processRankZero(builder, op);

      // process rank 1
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      processRankOne(builder, op);

      // remove original loop
      op.erase();
    });
  }

  void processRankZero(OpBuilder &builder, affine::AffineForOp forOp) {
    // send first half of data to rank 1
    auto loc = forOp.getLoc();
    auto retvalType = builder.getType<mpi::RetvalType>();
    auto i32Type = builder.getI32Type();

    // TODO: for (auto arg : funcOp.getArguments()) { mpi_send }

    // create affine loop for the second half
    // receive processed first half
  }

  void processRankOne(OpBuilder &builder, affine::AffineForOp forOp) {
    // allocate local buffers
    // receive data from rank 0
    // create affine loop for the first half
    // send result back to rank 0
    // cleanup local buffers (memrefs dealloc)
  }
};

} // namespace tutorial
} // namespace mlir
