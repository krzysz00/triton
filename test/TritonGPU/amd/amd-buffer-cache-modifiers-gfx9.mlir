// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx90a | FileCheck %s --check-prefixes=CHECK,GFX90A
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s --check-prefixes=CHECK,GFX942

// gfx90a uses GLC (bit 0) and SLC (bit 1); gfx940+ reuses them as SC0 and NT
// and adds SC1 (bit 4).

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: buffer_cache_modifiers
  tt.func @buffer_cache_modifiers(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<64xi32, #blocked> {tt.divisibility = 16 : i32}) {
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 0 : f32
    %ca = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = ca, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<64xf32, #blocked>
    // GFX90A: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 1 : f32
    // GFX942: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 3 : f32
    %cg = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<64xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 3 : f32
    %cs = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cs, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<64xf32, #blocked>
    // GFX90A: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 1 : f32
    // GFX942: rocdl.raw.ptr.buffer.load {{.*}}, {{.*}}, {{.*}}, 17 : f32
    %cv = amdg.buffer_load %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cv, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<64xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, 0 : f32
    amdg.buffer_store %ca, %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = wb, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<64xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, 0 : f32
    amdg.buffer_store %cg, %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<64xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, 3 : f32
    amdg.buffer_store %cs, %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = cs, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<64xf32, #blocked>
    // GFX90A: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, 3 : f32
    // GFX942: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, {{.*}}, {{.*}}, 17 : f32
    amdg.buffer_store %cv, %arg0[%offset] {cachePolicy = #tt.cache_policy<cache_modifier = wt, eviction_policy = evict_normal>} : !tt.ptr<f32> -> tensor<64xf32, #blocked>
    tt.return
  }
}
