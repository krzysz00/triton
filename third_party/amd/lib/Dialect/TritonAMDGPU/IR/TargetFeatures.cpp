#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::amdgpu {

using namespace llvm::AMDGPU;

namespace {
::mlir::ROCDL::TargetInfo parseTargetInfo(StringRef arch) {
  // Passes may run without a target.
  if (arch.empty())
    return {};
  FailureOr<::mlir::ROCDL::TargetInfo> info =
      ::mlir::ROCDL::TargetInfo::get(arch);
  return succeeded(info) ? *info : ::mlir::ROCDL::TargetInfo();
}

ISAFamily computeISAFamily(const ::mlir::ROCDL::TargetInfo &info);
} // namespace

TargetFeatures::TargetFeatures(std::optional<StringRef> arch)
    : TargetFeatures(arch.value_or("")) {}

TargetFeatures::TargetFeatures(StringRef arch)
    : arch(arch.str()), info(parseTargetInfo(arch)),
      isaFamily(computeISAFamily(info)) {}

TargetFeatures TargetFeatures::fromModuleOp(ModuleOp moduleOp) {
  auto targetAttr =
      moduleOp->getAttrOfType<StringAttr>(triton::gpu::AttrTargetName);
  if (!targetAttr)
    return TargetFeatures(StringRef());

  StringRef targetName = targetAttr.getValue();
  assert(targetName.starts_with(kTargetPrefix) &&
         "expected target attribute to be prefixed with \"hip:\"");

  return TargetFeatures(targetName.drop_front(sizeof(kTargetPrefix) - 1));
}

StringRef TargetFeatures::getArch() const { return arch; }

ISAFamily TargetFeatures::getISAFamily() const { return isaFamily; }

namespace {
ISAFamily computeISAFamily(const ::mlir::ROCDL::TargetInfo &info) {
  // Classify by feature so generic and future targets land in the right family.
  // gfx13 takes the gfx1250 paths until it needs its own.
  if (info.has(FEAT_GFX1250_INSTS))
    return ISAFamily::CDNA5;
  if (info.isGeneration(9)) {
    if (info.has(FEAT_GFX950_INSTS))
      return ISAFamily::CDNA4;
    if (info.has(FEAT_GFX940_INSTS))
      return ISAFamily::CDNA3;
    if (info.has(FEAT_GFX90A_INSTS))
      return ISAFamily::CDNA2;
    return ISAFamily::Unknown;
  }
  if (info.isGeneration(11)) {
    // gfx11.7 carries the gfx12-style 128-bit-operand WMMA instructions.
    if (info.has(FEAT_WMMA_128B_INSTS))
      return ISAFamily::RDNA4m;
    return ISAFamily::RDNA3;
  }
  if (info.isGeneration(12))
    return ISAFamily::RDNA4;
  return ISAFamily::Unknown;
}
} // namespace

bool TargetFeatures::isCDNA() const {
  return mlir::triton::amdgpu::isCDNA(getISAFamily());
}

bool TargetFeatures::isRDNA() const {
  return mlir::triton::amdgpu::isRDNA(getISAFamily());
}

bool TargetFeatures::isCDNA3() const {
  return getISAFamily() == ISAFamily::CDNA3;
}

bool TargetFeatures::isCDNA4() const {
  return getISAFamily() == ISAFamily::CDNA4;
}

bool TargetFeatures::isCDNA5() const {
  return getISAFamily() == ISAFamily::CDNA5;
}

bool TargetFeatures::isGFX1250Strict() const {
  return getGPUKind() == llvm::AMDGPU::GK_GFX1250_STRICT;
}

int TargetFeatures::getWarpSize() const {
  return info.getWavefrontSize().value_or(32);
}

bool TargetFeatures::supportsWaveId() const {
  // LLVM reads the wave ID from TTMP8, which needs gfx12's architected SGPRs.
  return has(FEAT_GFX12_INSTS);
}

int TargetFeatures::getSharedMemorySize() const {
  return info.getMaxAddressableLocalMemorySize().value_or(64 * 1024);
}

int TargetFeatures::getSharedMemoryBanks() const {
  return info.getLDSBankCount().value_or(32);
}

size_t TargetFeatures::getSharedMemoryPartitionSize() const {
  // TODO(llvm): Query the LDS partition size once LLVM exposes it.
  if (has(FEAT_GFX1250_INSTS))
    return 64 * 1024;
  // No partitioning on other targets.
  return 0;
}

std::optional<TargetFeatures::LDSTransLoadParams>
TargetFeatures::queryLDSTransLoadParams(int bitWidth) const {
  enum class TransLoadKind { None, Gfx950, Gfx1250 };
  struct TransConfig {
    TransLoadKind kind;
    int bitWidth;
    unsigned instBitWidth;
    // addr basis order:
    //   leading reg bases
    //   leading lane bases
    //   remaining reg bases
    //   remaining lane bases
    unsigned leadingRegBases;
    unsigned leadingLaneBases;
  };

  static constexpr TransConfig configs[] = {
      {TransLoadKind::Gfx950, 16, 64, 0, 2},
      {TransLoadKind::Gfx950, 8, 64, 0, 1},
      {TransLoadKind::Gfx950, 4, 64, 0, 0},
      {TransLoadKind::Gfx1250, 16, 128, 0, 0},
      {TransLoadKind::Gfx1250, 8, 64, 2, 1},
      {TransLoadKind::Gfx1250, 4, 64, 3, 1},
  };

  // TODO(llvm): Query a transposing-LDS-load feature once LLVM has one.
  TransLoadKind kind = TransLoadKind::None;
  if (has(FEAT_GFX1250_INSTS))
    kind = TransLoadKind::Gfx1250;
  else if (has(FEAT_GFX950_INSTS))
    kind = TransLoadKind::Gfx950;

  for (const auto &config : configs) {
    if (config.kind == kind && config.bitWidth == bitWidth) {
      return LDSTransLoadParams{
          config.instBitWidth, config.instBitWidth / config.bitWidth,
          config.leadingRegBases, config.leadingLaneBases};
    }
  }
  return std::nullopt;
}

bool TargetFeatures::supportsDirectToLdsScatter() const {
  // Only the async LDS loads take a per-lane LDS address.
  return has(FEAT_ASYNC_LOAD_TO_LDS_INSTS);
}

bool TargetFeatures::supportsDirectToLdsLoadBitWidth(int bitWidth) const {
  if (has(FEAT_ASYNC_LOAD_TO_LDS_INSTS)) {
    // Disable 8, 16 bits because they get extended to 32 bit and therefore
    // overwrite. 96 is not a pow2 and generally not useful in Triton.
    return llvm::is_contained({128, 64, /*96, */ 32, /*16, 8*/}, bitWidth);
  }
  if (has(FEAT_VMEM_TO_LDS_LOAD_INSTS)) {
    // 8/16 bits get extended to 32 and 96 to 128, so they are disabled.
    if (bitWidth == 128)
      return has(FEAT_GFX950_INSTS);
    return llvm::is_contained({32, /*16, 8*/}, bitWidth);
  }
  return false;
}

bool TargetFeatures::supportsDirectFromLdsStoreBitWidth(int bitWidth) const {
  if (has(FEAT_ASYNC_STORE_FROM_LDS_INSTS))
    return llvm::is_contained({128, 64, 32, 8}, bitWidth);
  return false;
}

bool TargetFeatures::supportsBufferLoadToLocal() const {
  return has(FEAT_VMEM_TO_LDS_LOAD_INSTS);
}

// The LDS DMA loads write LDS through M0, which alias analysis cannot see, and
// complete under vmcnt alongside ordinary loads.
bool TargetFeatures::requiresAliasInfoForAsyncOps() const {
  return has(FEAT_VMEM_TO_LDS_LOAD_INSTS) && !has(FEAT_ASYNCCNT);
}

bool TargetFeatures::useAsyncMarks() const {
  // TODO: Use asyncmarks for the asynccnt-based loads of gfx1250+ as well.
  return has(FEAT_VMEM_TO_LDS_LOAD_INSTS) && !has(FEAT_ASYNCCNT);
}

bool TargetFeatures::supportsTDM() const { return has(FEAT_TDM_INSTS); }

bool TargetFeatures::supportsMultiCTALaunch() const {
  return has(FEAT_CLUSTERS);
}

bool TargetFeatures::supportsMulticast() const {
  // LLVM forces the gfx1250-strict multicast mask to zero.
  return has(FEAT_MCAST_LOAD_INSTS) && !isGFX1250Strict();
}

unsigned TargetFeatures::getMaxMulticastMaskPopcount() const {
  return supportsMulticast() ? 5 : 1;
}

bool TargetFeatures::supportsClusterLoadBitWidth(int bitWidth) const {
  if (supportsMulticast()) {
    return llvm::is_contained({32, 64, 128}, bitWidth);
  }
  return false;
}

bool TargetFeatures::supportsBufferAtomicRMW() const {
  // Every gfx9+ target has the integer buffer atomics, but the fences and
  // cache-policy bits the buffer atomic lowering emits follow the gfx942 and
  // gfx11+ memory models, so gfx90a and older keep the generic lowering.
  return has(FEAT_GFX940_INSTS) || has(FEAT_GFX11_INSTS);
}

bool TargetFeatures::supportsBufferAtomicFadd(Type elementType) const {
  // Atomics with users need the returning form.
  if (elementType.isF32())
    return has(FEAT_ATOMIC_FADD_RTN_INSTS);
  if (elementType.isF16())
    return has(FEAT_ATOMIC_BUFFER_GLOBAL_PK_ADD_F16_INSTS);
  if (elementType.isBF16())
    return has(FEAT_ATOMIC_BUFFER_PK_ADD_BF16_INST);
  if (elementType.isF64())
    return has(FEAT_FLAT_BUFFER_GLOBAL_FADD_F64_INST);
  return false;
}

bool TargetFeatures::supportsBufferAtomicFMinMax(Type elementType) const {
  // LLVM gates the buffer forms on the global forms' features.
  if (elementType.isF32())
    return has(FEAT_ATOMIC_FMIN_FMAX_GLOBAL_F32);
  if (elementType.isF64())
    return has(FEAT_ATOMIC_FMIN_FMAX_GLOBAL_F64);
  return false;
}

int32_t TargetFeatures::getBufferAtomicCachePolicy(bool hasUsers) const {
  const int sc0Bit = 0b1;          // TH_ATOMIC_RETURN (cpol bit 0)
  const int scopeDevBit = 0b10000; // SCOPE_DEV = 2 << 3 (cpol bits [4:3])
  int32_t aux = 0;
  if (hasUsers)
    aux |= sc0Bit;
  // gfx12 agent-scope atomics need SCOPE_DEV.
  if (has(FEAT_GFX12_INSTS))
    aux |= scopeDevBit;
  return aux;
}

bool TargetFeatures::supportMaximumMinimum() const {
  // TODO(llvm): Query IEEEMinimumMaximumInsts/Minimum3Maximum3F32 once exposed.
  return has(FEAT_GFX950_INSTS) || has(FEAT_GFX12_INSTS) ||
         getISAFamily() == ISAFamily::RDNA4m;
}

bool TargetFeatures::supportDppBroadcast() const {
  return info.isGeneration(9);
}

bool TargetFeatures::supportsPermlane16Swap() const {
  return has(FEAT_PERMLANE16_SWAP);
}

bool TargetFeatures::supportsPermlane32Swap() const {
  return has(FEAT_PERMLANE32_SWAP);
}

bool TargetFeatures::supportsCvtPkScalePk8() const {
  // LLVM gates v_cvt_scale_pk8_* on gfx1250-insts.
  return has(FEAT_GFX1250_INSTS);
}

bool TargetFeatures::supportsCvtPkScalePk8Upcast() const {
  return supportsCvtPkScalePk8();
}

bool TargetFeatures::supportsCvtPkScalePk8Block16() const {
  return supportsCvtPkScalePk8() && has(FEAT_BLOCK16_CVT_SCALE_INSTS);
}

SmallVector<StringRef, 3> TargetFeatures::getUnsupportedWmmaFeatures() const {
  SmallVector<StringRef, 3> unsupported;
  // Only the gfx1250 WMMA table entries carry feature tags.
  if (has(FEAT_GFX1250_INSTS)) {
    if (!has(FEAT_WMMA_F4_INSTS))
      unsupported.push_back(kWmmaF4InstsFeature);
    if (!has(FEAT_WMMA_16X16X128FP8_INSTS))
      unsupported.push_back(kWmma16x16x128Fp8InstsFeature);
  }
  // TODO(llvm): gfx1250-strict also forbids the 16x16x32 f16/bf16 and 16x16x64
  // fp8/bf8/iu8 WMMAs, which LLVM still lists for it.
  if (isGFX1250Strict())
    unsupported.push_back(kWmmaRestrictedInstsFeature);
  return unsupported;
}

bool TargetFeatures::hasScaledPkCvtInsts() const {
  return has(FEAT_FP8_CVT_SCALE_INSTS) && has(FEAT_BF8_CVT_SCALE_INSTS) &&
         has(FEAT_FP4_CVT_SCALE_INSTS);
}

bool TargetFeatures::supportsHwScaledUpcast() const {
  return hasScaledPkCvtInsts() || supportsCvtPkScalePk8Upcast();
}

bool TargetFeatures::supportsHwScaledDowncast() const {
  return hasScaledPkCvtInsts() || supportsCvtPkScalePk8();
}

bool TargetFeatures::supportBitwidth16Elementwise() const { return true; }

bool TargetFeatures::supportBitwidth32Elementwise() const {
  // TODO(llvm): Query PackedFP32Ops once exposed.
  return has(FEAT_GFX90A_INSTS) || has(FEAT_GFX1250_INSTS);
}

bool isCDNA(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
  case ISAFamily::CDNA5:
    return true;
  default:
    return false;
  }
}

bool isRDNA(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::RDNA3:
  case ISAFamily::RDNA4m:
  case ISAFamily::RDNA4:
    return true;
  default:
    return false;
  }
}

} // namespace mlir::triton::amdgpu
