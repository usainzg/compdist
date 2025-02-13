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

      auto i32Type = builder.getI32Type();
      // create constants for n_ranks
      SmallVector<Value, 4> rankConstants;
      for (int i = 0; i < n_ranks; ++i) {
        rankConstants.push_back(builder.create<arith::ConstantOp>(
            op.getLoc(), i32Type, builder.getI32IntegerAttr(i)));
      }

      // add mpi retval + init
      auto mpiRetvalType = builder.getType<mpi::RetvalType>();
      builder.create<mpi::InitOp>(op.getLoc(), mpiRetvalType);

      // get mpi rank
      auto mpiRankOp =
          builder.create<mpi::CommRankOp>(op.getLoc(), mpiRetvalType, i32Type);

      // create comparison for rank
      auto cmpOp =
          builder.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::eq,
                                        mpiRankOp.getRank(), rankConstants[0]);

      // create if-else structure
      auto ifOp =
          builder.create<scf::IfOp>(op.getLoc(), cmpOp, /*withElseBlock=*/true);

      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      // TODO: process mpi rank 0 (master)
      SmallVector<Value, 4> loadMemrefs;
      SmallVector<Value, 4> storeMemrefs;
      op.walk([&](memref::LoadOp loadOp) {
        loadMemrefs.push_back(loadOp.getMemRef());
      });
      op.walk([&](memref::StoreOp storeOp) {
        storeMemrefs.push_back(storeOp.getMemRef());
      });

      SmallVector<Value, 4> loadSubviews;
      for (auto memref : loadMemrefs) {
        auto chunkSize =
            cast<MemRefType>(memref.getType()).getShape()[0] / n_ranks;
        auto memrefType = cast<MemRefType>(memref.getType());
        auto [strides, offsets] = memrefType.getStridesAndOffset();
        auto subViewOp = builder.create<memref::SubViewOp>(
            op.getLoc(), memref, offsets, chunkSize, strides);
        loadSubviews.push_back(subViewOp);
      }
      for (auto subview : loadSubviews) {
        // send a subview to each rank except rank 0
        for (int i = 1; i < n_ranks; ++i) {
          builder.create<mpi::SendOp>(op.getLoc(), mpiRetvalType, subview,
                                      rankConstants[0], rankConstants[i]);
        }
      }

      // create a new affine loop with the chunk size
      // TODO: change getHalfPoint
      auto upperBoundMapHalf = getHalfPoint(builder, op, /*isUpper=*/true);
      auto upperBoundOperandsHalf = op.getUpperBoundOperands();
      auto lowerBoundMapHalf = op.getLowerBoundMap();
      auto lowerBoundOperandsHalf = op.getLowerBoundOperands();

      // insert new loop
      auto newLoop = builder.create<affine::AffineForOp>(
          op->getLoc(), lowerBoundOperandsHalf, lowerBoundMapHalf,
          upperBoundOperandsHalf, upperBoundMapHalf);

      // clone the original loop body into the new loop
      IRMapping mapping;
      mapping.map(op.getInductionVar(), newLoop.getInductionVar());

      // get the original loop body
      Block &originalBody = op.getRegion().front();

      // clone operations from original body to new loop body, excluding the
      // terminator
      builder.setInsertionPointToStart(newLoop.getBody());
      for (auto &op : originalBody.without_terminator()) {
        builder.clone(op, mapping);
      }

      SmallVector<Value, 4> storeSubviews;
      builder.setInsertionPointAfter(newLoop);
      for (auto memref : storeMemrefs) {
        auto chunkSize =
            cast<MemRefType>(memref.getType()).getShape()[0] / n_ranks;
        auto memrefType = cast<MemRefType>(memref.getType());
        auto [strides, offsets] = memrefType.getStridesAndOffset();
        auto subViewOp = builder.create<memref::SubViewOp>(
            op.getLoc(), memref, offsets, chunkSize, strides);
        storeSubviews.push_back(subViewOp);
      }
      for (auto subview : storeSubviews) {
        // receive a subview from other ranks
        for (int i = 1; i < n_ranks; ++i) {
          builder.create<mpi::RecvOp>(op.getLoc(), mpiRetvalType, subview,
                                      rankConstants[i], rankConstants[i]);
        }
      }

      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      // TODO: process other mpi ranks

      for (int i = 1; i < n_ranks; ++i) {
        auto cmpOpOtherRanks = builder.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::eq, mpiRankOp.getRank(),
            rankConstants[i]);

        auto ifOpRank = builder.create<scf::IfOp>(op.getLoc(), cmpOpOtherRanks,
                                                  /*withElseBlock=*/false);

        builder.setInsertionPointToStart(&ifOpRank.getThenRegion().front());

        // Allocate memory for the subview
        MemRefType memRefType = MemRefType::get(
            {cast<MemRefType>(loadMemrefs[0].getType()).getShape()[0] /
             n_ranks},
            cast<MemRefType>(loadMemrefs[0].getType()).getElementType());
        auto allocOp = builder.create<memref::AllocOp>(op.getLoc(), memRefType);

        // Receive the subview from rank 0
        builder.create<mpi::RecvOp>(op.getLoc(), mpiRetvalType, allocOp,
                                    rankConstants[0], rankConstants[0]);

        // Compute the chunk of the loop
        auto upperBoundMapHalfRank =
            getHalfPoint(builder, op, /*isUpper=*/true);
        auto upperBoundOperandsHalfRank = op.getUpperBoundOperands();
        auto lowerBoundMapHalfRank = op.getLowerBoundMap();
        auto lowerBoundOperandsHalfRank = op.getLowerBoundOperands();

        // Insert the new loop for each rank
        auto newLoopRank = builder.create<affine::AffineForOp>(
            op->getLoc(), lowerBoundOperandsHalfRank, lowerBoundMapHalfRank,
            upperBoundOperandsHalfRank, upperBoundMapHalfRank);

        // Clone the original loop body into the new loop for the rank
        IRMapping rankMapping;
        rankMapping.map(op.getInductionVar(), newLoopRank.getInductionVar());

        builder.setInsertionPointToStart(newLoopRank.getBody());
        for (auto &originalOp : originalBody.without_terminator()) {
          builder.clone(originalOp, rankMapping);
        }

        builder.setInsertionPointAfter(newLoopRank);

        // Send the result back to rank 0
        // Create a subview from the allocated memory
        auto [strides, offsets] = memRefType.getStridesAndOffset();
        auto subViewOpRank = builder.create<memref::SubViewOp>(
            op.getLoc(), allocOp, offsets,
            cast<MemRefType>(loadMemrefs[0].getType()).getShape()[0] / n_ranks,
            strides);
        builder.create<mpi::SendOp>(op.getLoc(), mpiRetvalType, subViewOpRank,
                                    rankConstants[i], rankConstants[0]);

        builder.create<memref::DeallocOp>(op.getLoc(), allocOp);
      }

      // remove original loop
      op.erase();
    });
  }

  // helper function to get the midpoint of the loop range
  // TODO: change to get
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
