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

      // process rank 0
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      processRankZero(builder, op, c1, c0);

      /*SmallVector<Value, 4> memrefOperands;*/
      /*op.walk([&](Operation *op) {*/
      /*  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {*/
      /*    if (!llvm::is_contained(memrefOperands, loadOp.getMemref()))*/
      /*      memrefOperands.push_back(loadOp.getMemref());*/
      /*  }*/
      /*  if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {*/
      /*    if (!llvm::is_contained(memrefOperands, storeOp.getMemref()))*/
      /*      memrefOperands.push_back(storeOp.getMemref());*/
      /*  }*/
      /*});*/
      /**/
      /*auto nRanks =*/
      /*    builder.create<arith::ConstantIndexOp>(op.getLoc(), n_ranks);*/
      /**/
      /*auto arraySize = builder.create<memref::DimOp>(*/
      /*    op.getLoc(),*/
      /*    memrefOperands[0], // Use first memref*/
      /*    builder.create<arith::ConstantIndexOp>(op.getLoc(),*/
      /*                                           0) // First dimension*/
      /*);*/

      

      /*processRankZero_2(builder, op, c1, c0);*/

      // process rank 1
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      processRankOne(builder, op, c0, c1);

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

  // TODO:send/recv only a chunk (subview) of the data to/from rank 1
  // dynamic offset is needed
  void processRankZero_2(OpBuilder &builder, affine::AffineForOp forOp,
                         Value dest, Value tag) {
    auto loc = forOp.getLoc();
    auto retvalType = builder.getType<mpi::RetvalType>();

    // collect all memref operands from the loop body that are used by rank 1
    // TODO:change to input/output operands
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

    // send only the necessary subview of each memref to rank 1
    // TODO:send only input operands?
    for (auto memref : memrefOperands) {
      auto memrefType = mlir::cast<MemRefType>(memref.getType());
      auto subviewType = MemRefType::get({memrefType.getShape()[0] / 2},
                                         memrefType.getElementType());
      auto subview = builder.create<memref::SubViewOp>(
          loc, subviewType, memref, ValueRange{tag},
          ValueRange{builder.create<arith::ConstantIndexOp>(
              loc, memrefType.getShape()[0] / 2)},
          ValueRange{builder.create<arith::ConstantIndexOp>(loc, 1)});
      builder.create<mpi::SendOp>(loc, retvalType, subview, dest, tag);
    }

    // create affine loop for the second half
    // new bound for the new loop
    /*auto upperBoundMap = forOp.getUpperBoundMap();*/
    /*auto upperBoundOperands = forOp.getUpperBoundOperands();*/
    /*auto lowerBoundMap = getHalfPoint(builder, forOp, isUpper=false);*/
    /*auto lowerBoundOperands = forOp.getLowerBoundOperands();*/

    /*// insert new loop*/
    /*auto newLoop = builder.create<affine::AffineForOp>(*/
    /*    loc, lowerBoundOperands, lowerBoundMap, upperBoundOperands,*/
    /*    upperBoundMap);*/
    /**/
    /*// clone the original loop body into the new loop*/
    /*IRMapping mapping;*/
    /*mapping.map(forOp.getInductionVar(), newLoop.getInductionVar());*/
    /**/
    /*// get the original loop body*/
    /*Block &originalBody = forOp.getRegion().front();*/
    /**/
    /*// clone operations from original body to new loop body, excluding the*/
    /*// terminator*/
    /*builder.setInsertionPointToStart(newLoop.getBody());*/
    /*for (auto &op : originalBody.without_terminator()) {*/
    /*  builder.clone(op, mapping);*/
    /*}*/
    /**/
    /*// receive only the necessary subview of the memrefs from rank 1*/
    /*// TODO:receive only the output operand?*/
    /*builder.setInsertionPointAfter(newLoop);*/
    /*for (auto memref : memrefOperands) {*/
    /*  auto memrefType = mlir::cast<MemRefType>(memref.getType());*/
    /*  auto subviewType = MemRefType::get({memrefType.getShape()[0] / 2},*/
    /*                                     memrefType.getElementType());*/
    /*  auto subview = builder.create<memref::SubViewOp>(*/
    /*      loc, subviewType, memref, ValueRange{tag},*/
    /*      ValueRange{builder.create<arith::ConstantIndexOp>(*/
    /*          loc, memrefType.getShape()[0] / 2)},*/
    /*      ValueRange{builder.create<arith::ConstantIndexOp>(loc, 1)});*/
    /*  builder.create<mpi::RecvOp>(loc, retvalType, subview, dest, tag);*/
    /*}*/
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
    auto lowerBoundMap = getHalfPoint(builder, forOp);
    auto lowerBoundOperands = forOp.getUpperBoundOperands();
    auto upperBoundMap = forOp.getLowerBoundMap();
    auto upperBoundOperands = forOp.getLowerBoundOperands();

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
