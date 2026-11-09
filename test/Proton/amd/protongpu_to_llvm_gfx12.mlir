// RUN: triton-opt %s -split-input-file -convert-proton-amd-gpu-to-llvm="gfx-arch=gfx1200" --convert-builtin-func-to-llvm | FileCheck %s

// gfx11+ has no s_memrealtime.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, ttg.profile_scratch_memory_alignment = 128 : i32, ttg.profile_scratch_memory_size = 384 : i32} {
  // CHECK-LABEL: convert_smem_finalize
  // CHECK-NOT: s.memrealtime
  // CHECK: %[[MSG:.*]] = llvm.mlir.constant(131 : i32) : i32
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.sendmsg.rtn.i64"(%[[MSG]]) : (i32) -> i64
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.s.sendmsg.rtn.i64"(%[[MSG]]) : (i32) -> i64
  // CHECK-NOT: s.memrealtime
  // CHECK: llvm.return
  llvm.func @convert_smem_finalize(%arg: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    %1 = ttg.global_scratch_alloc {alignment = 128 : i32, third_party_allocation, nbytes = 384 : i32, ttg.global_scratch_memory_offset = 0 : i32} : !tt.ptr<i32>
    %2 = proton_gpu.segment_alloc %0 : !ttg.memdesc<512xi32, #shared, #smem, mutable> -> !proton_gpu.segment<2048, #smem, warp>
    proton_gpu.finalize %2, %1 : !proton_gpu.segment<2048, #smem, warp>, !tt.ptr<i32>
    llvm.return
  }
}
