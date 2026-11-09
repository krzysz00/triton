#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_TARGETCHIPSET_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_TARGETCHIPSET_H_

// Kept apart from TargetFeatures.h: declaring mlir::amdgpu there would make
// `amdgpu` ambiguous in the many files that use both mlir and mlir::triton.

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"

namespace mlir::triton::amdgpu {

// For upstream patterns that still take a Chipset. Generic targets give their
// oldest covered ISA version, which those patterns' version checks tolerate.
inline FailureOr<::mlir::amdgpu::Chipset>
getChipset(const TargetFeatures &features) {
  const ::mlir::ROCDL::TargetInfo &info = features.getTargetInfo();
  if (info.isUnknown())
    return failure();
  llvm::AMDGPU::IsaVersion version = info.getIsaVersion();
  return ::mlir::amdgpu::Chipset(version.Major, version.Minor,
                                 version.Stepping);
}

} // namespace mlir::triton::amdgpu

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_IR_TARGETCHIPSET_H_
