// RUN: triton-opt %s -split-input-file --tritonamdgpu-coalesce-async-copy=gfx-arch=gfx90a | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "hip:gfx90a", "ttg.threads-per-warp" = 64 : i32} {
// CHECK: #[[$NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: async_copy_1d
tt.func @async_copy_1d(%input: tensor<1024x!tt.ptr<f32>, #blocked> {tt.contiguity = 4 : i32, tt.divisibility = 16 : i32},
    %view: !ttg.memdesc<1024xf32, #shared, #smem, mutable>) {
  // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<1024x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
  // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<1024x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
  %token = ttg.async_copy_global_to_local %input, %view: tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
  tt.return
}
}
