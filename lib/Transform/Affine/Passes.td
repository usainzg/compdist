#ifndef LIB_TRANSFORM_AFFINE_PASSES_TD_
#define LIB_TRANSFORM_AFFINE_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AffineDistributeToMPI : Pass<"affine-distribute-to-mpi"> {
  let summary = "Distribute affine loops to MPI processes";
  let description = [{
    Distribute affine loops to MPI processes.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::mpi::MPIDialect",
    "mlir::memref::MemRefDialect",
  ];
}

#endif  // LIB_TRANSFORM_AFFINE_PASSES_TD_
