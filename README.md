# compdist - project report

## main goal

- explore the intersection betweenc compilers and distributed computing, trying to enrich compilers
with domain specific information about the target system, e.g., topology, nodes capacity, etc.

## progress

- the basic idea is to have a pass to optimize affine -> mpi, starting with
a basic partition (A+B) in two ranks, processing a half of the data on each
process []

## tasks

- clean mlir-tutorial code []
- create a basic pass using cpp mechanism []
- update basic version with more ranks (e.g., -n-ranks=10) []
- create a new version where the number of ranks is passed as an option
and the pass distributes the data uniformly to every rank []

## Examples

### A + B (partition in halfs)

Input:

```mlir
func.func @add_arrays(%A: memref<100xf32>, %B: memref<100xf32>, %C: memref<100xf32>) {
  affine.for %i = 0 to 100 {
    %a = memref.load %A[%i] : memref<100xf32>
    %b = memref.load %B[%i] : memref<100xf32>
    %sum = arith.addf %a, %b : f32
    memref.store %sum, %C[%i] : memref<100xf32>
  }
  return
}
```

Output:

```mlir
func.func @add_arrays_mpi(%A: memref<100xf32>, %B: memref<100xf32>, %C: memref<100xf32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c50 = arith.constant 50 : index  // Half of 100

    // Get MPI rank
    %init_err = mpi.init : !mpi.retval
    %rank_err, %rank = mpi.comm_rank : !mpi.retval, i32

    // Process based on rank
    %zero = arith.constant 0 : i32
    %is_rank_zero = arith.cmpi eq, %rank, %zero : i32
    scf.if %is_rank_zero {
      // Process 0 sends first half of data
      %send_err1 = mpi.send(%A, %c1, %c0) : memref<100xf32>, i32, i32 -> !mpi.retval
      %send_err2 = mpi.send(%B, %c1, %c0) : memref<100xf32>, i32, i32 -> !mpi.retval

      // Process local half
      affine.for %i = 50 to 100 {
        %a = memref.load %A[%i] : memref<100xf32>
        %b = memref.load %B[%i] : memref<100xf32>
        %sum = arith.addf %a, %b : f32
        memref.store %sum, %C[%i] : memref<100xf32>
      }

      // Receive processed first half
      %recv_err = mpi.recv(%C, %c1, %c0) : memref<100xf32>, i32, i32 -> !mpi.retval

    } else {
      // Process 1 receives and processes first half
      %local_a = memref.alloc() : memref<100xf32>
      %local_b = memref.alloc() : memref<100xf32>
      %local_result = memref.alloc() : memref<100xf32>

      %recv_err1 = mpi.recv(%local_a, %c0, %c0) : memref<100xf32>, i32, i32 -> !mpi.retval
      %recv_err2 = mpi.recv(%local_b, %c0, %c0) : memref<100xf32>, i32, i32 -> !mpi.retval

      affine.for %i = 0 to 50 {
        %a = memref.load %local_a[%i] : memref<100xf32>
        %b = memref.load %local_b[%i] : memref<100xf32>
        %sum = arith.addf %a, %b : f32
        memref.store %sum, %local_result[%i] : memref<100xf32>
      }

      %send_err = mpi.send(%local_result, %c0, %c0) : memref<100xf32>, i32, i32 -> !mpi.retval

      memref.dealloc %local_a : memref<100xf32>
      memref.dealloc %local_b : memref<100xf32>
      memref.dealloc %local_result : memref<100xf32>
    }

    return
}
```
