func.func @add_arrays(%A: memref<100xf32>, %B: memref<100xf32>, %C: memref<100xf32>) {
  affine.for %i = 0 to 100 {
    %a = memref.load %A[%i] : memref<100xf32>
    %b = memref.load %B[%i] : memref<100xf32>
    %sum = arith.addf %a, %b : f32
    memref.store %sum, %C[%i] : memref<100xf32>
  }
  return
}
