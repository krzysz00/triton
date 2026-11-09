// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx950" | FileCheck %s --check-prefixes=CHECK,DOT2
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942" | FileCheck %s --check-prefixes=CHECK,NODOT2

// Targets with v_dot2_f32_bf16 (dot12-insts) keep bf16 FMA operands; others
// upcast them to f32.

// CHECK-LABEL: fma_dot_bf16
// DOT2: tt.dot {{.*}} : tensor<2x64xbf16, {{.*}}> * tensor<64x64xbf16, {{.*}}> -> tensor<2x64xf32, {{.*}}>
// NODOT2: tt.dot {{.*}} : tensor<2x64xf32, {{.*}}> * tensor<64x64xf32, {{.*}}> -> tensor<2x64xf32, {{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fma_dot_bf16(
      %arg0: tensor<2x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<2x64x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.0> : tensor<2x64xf32, #blocked>
    %1 = tt.dot %arg0, %arg1, %cst : tensor<2x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<2x64xf32, #blocked>
    tt.store %arg2, %1 : tensor<2x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
