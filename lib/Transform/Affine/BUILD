# Passes that work with the Affine dialect

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Affine",
            ],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "AffinePasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
  name = "AffineDistributeToMPI",
  srcs = ["AffineDistributeToMPI.cpp"],
  hdrs = [
      "AffineDistributeToMPI.h",
      "Passes.h",
  ],
  deps = [
      ":pass_inc_gen",
      "@llvm-project//mlir:AffineDialect",
      "@llvm-project//mlir:MPIDialect",
      "@llvm-project//mlir:SCFDialect",
      "@llvm-project//mlir:FuncDialect",
      "@llvm-project//mlir:MemRefDialect",
      "@llvm-project//mlir:Pass",
      "@llvm-project//mlir:Transforms",
  ],
)

cc_library(
    name = "Passes",
    hdrs = ["Passes.h"],
    deps = [
        "AffineDistributeToMPI",
        ":pass_inc_gen",
    ],
)
