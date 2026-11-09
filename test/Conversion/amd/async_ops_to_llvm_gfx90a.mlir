// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx90a --verify-diagnostics | FileCheck %s

// gfx90a has the dword LDS DMA loads (vmem-to-lds-load-insts).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx90a", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy
  tt.func public @async_copy(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg2: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>) {
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    // CHECK-COUNT-8: rocdl.global.load.async.lds {{.*}} <alias_scopes =
    // CHECK-NOT: rocdl.global.load.async.lds
    %2 = ttg.async_copy_global_to_local %1, %arg2 : tensor<32x64x!tt.ptr<f32>, #blocked> -> <32x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx90a", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: buffer_load_to_local_simple
  tt.func public @buffer_load_to_local_simple(%arg1: !tt.ptr<f32>,
                                %arg2: tensor<32x64xi32, #blocked>,
                                %arg3: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>) {
    // CHECK: rocdl.make.buffer.rsrc
    // CHECK-NOT: rocdl.make.buffer.rsrc
    // CHECK-COUNT-8: rocdl.raw.ptr.buffer.load.async.lds {{.*}} <alias_scopes =
    // CHECK-NOT: rocdl.raw.ptr.buffer.load.async.lds
    %65 = amdg.buffer_load_to_local %arg1[%arg2] into %arg3 : !tt.ptr<f32>[tensor<32x64xi32, #blocked>] -> <32x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "hip:gfx90a", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @buffer_load_to_local_vectorized_8xf16(%arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !ttg.memdesc<256x8xf16, #shared, #smem, mutable>) {
    %cst = arith.constant dense<8> : tensor<256x1xi32, #blocked>
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %3 = arith.muli %2, %cst : tensor<256x1xi32, #blocked>
    %4 = tt.broadcast %3 : tensor<256x1xi32, #blocked> -> tensor<256x8xi32, #blocked>
    %5 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<1x8xi32, #blocked> -> tensor<256x8xi32, #blocked>
    %7 = arith.addi %4, %6 : tensor<256x8xi32, #blocked>
    // The dwordx4 LDS DMA load is gfx950-only.
    // expected-error@+2 {{cannot lower 'amdg.buffer_load_to_local' to a direct-to-LDS copy}}
    // expected-error@+1 {{failed to legalize operation 'amdg.buffer_load_to_local' that was explicitly marked illegal}}
    %8 = amdg.buffer_load_to_local %arg1[%7] into %arg2 : !tt.ptr<f16>[tensor<256x8xi32, #blocked>]  -> <256x8xf16, #shared, #smem, mutable>
    tt.return
  }
}
