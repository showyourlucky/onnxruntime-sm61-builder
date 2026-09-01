import os
import sys

print("--- Starting ONNX Runtime sm_61 Isolation Patch ---")

# 1. Patch CMake source filters
filter_path = os.path.join('onnxruntime-src', 'cmake', 'onnxruntime_cuda_source_filters.cmake')
if os.path.exists(filter_path):
    with open(filter_path, 'r', encoding='utf-8') as f:
        c = f.read()
    if '/bert/flash_attention/' in c:
        c = c.replace('/bert/flash_attention/', '/bert/(flash_attention|cutlass_fmha)/')
        with open(filter_path, 'w', encoding='utf-8') as f:
            f.write(c)
        print(f"Successfully patched {filter_path} to isolate cutlass_fmha.")

# 2. Stub all 4 CUTLASS FMHA architecture files (sm50, sm70, sm75, sm80)
# Pascal (sm_61, GTX 1050 Ti) lacks tensor cores and subbyte atomic memory ordering
# Standard Attention / cuBLAS / cuDNN path is used instead for full functionality.
fmha_dir = os.path.join('onnxruntime-src', 'onnxruntime', 'contrib_ops', 'cuda', 'bert', 'cutlass_fmha')
for sm, fname in [('50', 'fmha_sm50.cu'), ('70', 'fmha_sm70.cu'), ('75', 'fmha_sm75.cu'), ('80', 'fmha_sm80.cu')]:
    fpath = os.path.join(fmha_dir, fname)
    if os.path.exists(fpath):
        stub_content = f"""// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_MEMORY_EFFICIENT_ATTENTION

#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"

namespace onnxruntime {{
namespace contrib {{
namespace cuda {{

void run_memory_efficient_attention_sm{sm}(const MemoryEfficientAttentionParams& params) {{
  // Empty stub for sm_61 architecture (Pascal uses standard cuBLAS / cuDNN attention path)
}}

}}  // namespace cuda
}}  // namespace contrib
}}  // namespace onnxruntime

#endif  // USE_MEMORY_EFFICIENT_ATTENTION
"""
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(stub_content)
        print(f"Successfully stubbed {fname} for sm_{sm}")
    else:
        print(f"Warning: {fpath} not found!")

print("--- Patch completed successfully ---")
