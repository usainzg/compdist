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

      // remove original loop
      op.erase();
    });
  }
};

} // namespace tutorial
} // namespace mlir
