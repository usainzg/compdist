#include "lib/Transform/Affine/Passes.h"
#include "mlir/include/mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/include/mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/include/mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/include/mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/include/mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/include/mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/include/mlir/Dialect/Linalg/Passes.h"
#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/include/mlir/InitAllPasses.h"
#include "mlir/include/mlir/Pass/PassManager.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/include/mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::tutorial::registerAffinePasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
