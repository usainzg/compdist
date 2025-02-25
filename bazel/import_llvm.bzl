"""Provides the repository macro to import LLVM."""

load(
    "@bazel_tools//tools/build_defs/repo:git.bzl",
    "new_git_repository",
)

def import_llvm(name):
    """Imports LLVM."""

    # 2025-02-22
    # LLVM_COMMIT = "16d4453f2f5d9554ce39507fda0f33ce9066007b"
    LLVM_COMMIT = "31824b2a11a8aa3e1d5c699cd0786ad666999abf"

    new_git_repository(
        name = name,
        # this BUILD file is intentionally empty, because the LLVM project
        # internally contains a set of bazel BUILD files overlaying the project.
        build_file_content = "# empty",
        commit = LLVM_COMMIT,
        init_submodules = False,
        remote = "https://github.com/llvm/llvm-project.git",
    )
