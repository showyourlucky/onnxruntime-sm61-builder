import os
import sys

print("--- Starting ONNX Runtime sm_61 / sm_80 Isolation Patch ---")

# 1. Patch CMake source filters to isolate cutlass_fmha into SM80 object library
filter_path = os.path.join('onnxruntime-src', 'cmake', 'onnxruntime_cuda_source_filters.cmake')
if os.path.exists(filter_path):
    with open(filter_path, 'r', encoding='utf-8') as f:
        c = f.read()
    if '/bert/flash_attention/' in c:
        c = c.replace('/bert/flash_attention/', '/bert/(flash_attention|cutlass_fmha)/')
        with open(filter_path, 'w', encoding='utf-8') as f:
            f.write(c)
        print(f"Successfully patched {filter_path} to isolate cutlass_fmha into SM80 library.")
    else:
        print(f"Warning: '/bert/flash_attention/' not found in {filter_path}")
else:
    print(f"Error: {filter_path} does not exist!")

# 2. Patch fmha_sm80.cu directly with pre-sm80 device guard and fallback stub
fmha_path = os.path.join('onnxruntime-src', 'onnxruntime', 'contrib_ops', 'cuda', 'bert', 'cutlass_fmha', 'fmha_sm80.cu')
if os.path.exists(fmha_path):
    fmha_content = """// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_MEMORY_EFFICIENT_ATTENTION

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)

#include "contrib_ops/cuda/bert/cutlass_fmha/fmha_launch_template.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

void run_memory_efficient_attention_sm80(const MemoryEfficientAttentionParams& params) {
  if (params.is_half) {
    DispatchBlockSize<cutlass::half_t, cutlass::arch::Sm80>(params);
  } else if (params.is_bf16) {
    DispatchBlockSize<cutlass::bfloat16_t, cutlass::arch::Sm80>(params);
  } else {
    DispatchBlockSize<float, cutlass::arch::Sm80>(params);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#else  // Fallback stub for pre-sm80 device passes

namespace onnxruntime {
namespace contrib {
namespace cuda {

void run_memory_efficient_attention_sm80(const MemoryEfficientAttentionParams& params) {
  // Empty stub for pre-sm80 device passes
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif

#endif  // USE_MEMORY_EFFICIENT_ATTENTION
"""
    with open(fmha_path, 'w', encoding='utf-8') as f:
        f.write(fmha_content)
    print(f"Successfully patched {fmha_path} with pre-sm80 fallback stub.")
else:
    print(f"Error: {fmha_path} does not exist!")

print("--- Patch completed successfully ---")
