# fa-research-cpp

FlashAttention-inspired attention library (C++ only, CPU reference). CUDA support may come later behind a flag.

## Status
- Base commit: ✅ builds and passes baseline tests (Tensor + Utils).
- `attention_forward` is **NYI** in base (intentionally) so PR1’s new tests will be F2P.

## Build & Test (CPU-only)
```bash
rmdir /s /q build
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure


## Files
include/fa/
  attention.hpp      # public API (declared; NYI in base)
  tensor.hpp         # owning Tensor (float32), shape/strides, checked access
  types.hpp          # AttentionOpts (causal, dropout, block sizes)
  mask.hpp           # mask helpers (decl; implemented later)
  math.hpp           # math helpers: row_max, sumexp, etc.

src/
  attention_ref.cpp  # attention_forward (NYI in base; PR1 implements)
  mask.cpp           # mask ops (stub in base; PR2 implements)
  cpu/tensor.cpp     # tensor implementation
  common/checks.cpp  # shared argument/shape checks
  common/math.cpp    # math helpers impl
  common/random.cpp  # RNG utils

tests/
  test_tensor.cpp    # baseline P2P tests for Tensor
  test_utils.cpp     # baseline P2P tests for math helpers

.github/workflows/ci.yml  # CPU-only CI: configure, build, test
CMakeLists.txt
