// RUN: triton-opt %s -split-input-file -convert-proton-amd-gpu-to-llvm="gfx-arch=gfx1250-strict" --verify-diagnostics | FileCheck %s

// gfx1250-strict used to fail chipset parsing.

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_read_counter
  llvm.func @convert_read_counter() -> i32 {
    // CHECK: llvm.call_intrinsic "llvm.readcyclecounter"() : () -> i64
    // CHECK: llvm.trunc %{{.*}} : i64 to i32
    %1 = proton_gpu.read_counter : i32
    llvm.return %1 : i32
  }
}
