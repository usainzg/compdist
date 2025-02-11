add_mlir_library(AffineDistributeToMPI
    AffineDistributeToMPI.cpp
    AffineDistributeToMPI.cpp

    ${PROJECT_SOURCE_DIR}/lib/Transform/Affine/
    ADDITIONAL_HEADER_DIRS

    DEPENDS
    MLIRAffineDistributeToMPIPasses

    LINK_LIBS PUBLIC
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Affine)
add_public_tablegen_target(MLIRAffineDistrbuteToMPIPasses)
add_mlir_doc(Passes AffinePasses ./ -gen-pass-doc)
