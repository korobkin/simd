# Architecture support

This library is x86_64-only. `include/simd.hpp` includes `<immintrin.h>` and
uses 32 `__m256`/`__m128` types and ~150 `_mm*` intrinsics, with no alternative
code path; `CMakeLists.txt` passes `-march=native -mavx2` unconditionally.

On AArch64 that fails immediately:

```
include/simd.hpp:3:10: fatal error: immintrin.h: No such file or directory
g++: error: unrecognized command-line option '-mavx2'
```

`immintrin.h` is the Intel intrinsics header, shipped by GCC only for x86
targets. There is no package that provides it on ARM — the dependency is
structural, not a missing library.

## The ARM SIMD landscape

| | width | header | where |
|---|---|---|---|
| NEON (Advanced SIMD) | 128-bit, fixed | `arm_neon.h` | every AArch64 chip, including all Apple Silicon |
| SVE / SVE2 | 128–2048-bit, runtime | `arm_sve.h` | Neoverse V1/V2, NVIDIA Grace, Graviton3+, Fujitsu A64FX, Cortex-X925 |
| SME | streaming / matrix | `arm_sme.h` | Apple M4, newest Arm server cores |

**NEON** is the baseline and is mandatory on AArch64, so it is always
available. Vectors are a fixed 128 bits: `float32x4_t` (4 floats),
`float64x2_t` (2 doubles). That is half the width of AVX2, but width is not
throughput — Apple's M-series run four 128-bit NEON pipelines, giving per-cycle
FLOPs comparable to a 256-bit AVX2 core.

**SVE** is vector-length agnostic: the same binary runs on any implementation
from 128 to 2048 bits, with the length queried at runtime rather than fixed at
compile time. This is the better target on server parts, at the cost of a
programming model that differs more from the x86 one.

**Apple Silicon has no SVE.** M1 through M3 are NEON-only. M4 adds SME. The
older AMX matrix unit is undocumented and reachable only through Accelerate.

## Porting options

Ordered by effort.

1. **SIMDe** (`simd-everywhere`) reimplements the Intel intrinsics on top of
   NEON, so `immintrin.h`-based code often compiles almost unchanged. By far
   the least work. The catch: 256-bit operations are emulated as two 128-bit
   ones, so expect the AVX2 path to be correct but not optimal.

2. **A portable SIMD layer** — Google Highway or xsimd — rewrites the backend
   once and is genuinely native on both x86 and ARM. Highway is the usual
   choice for a single codebase spanning both. `std::experimental::simd` (in
   libstdc++ since GCC 11) is the standards-track equivalent, but its
   math-function coverage is thinner than what this library generates.

3. **A native NEON/SVE backend.** Every `_mm*` call needs an ARM equivalent and
   `simd_f32`/`simd_f64` need new representations. The accuracy tests in
   `src/test.cpp` are the safety net: the generated polynomial approximations
   are architecture-independent, so only the vector primitives change.

4. **A scalar fallback** — `simd_f32` wrapping a plain `float[N]` behind an
   `#ifdef`. No speedup, but it makes dependent projects build and run on ARM.

## Notes for whichever route is taken

- `size()` is not a constant across architectures. On NEON `simd_f32::size()`
  is 4 and `simd_f64::size()` is 2, against 8 and 4 on AVX2. Anything that
  assumes the AVX2 widths needs checking — see `N_bit_shift` and the buffer
  sizing in `src/test.cpp`.
- `-march=native -mavx2` in `CMakeLists.txt` needs to become
  architecture-dependent. On AArch64 `-march=native` alone is usually right;
  SVE needs `-march=armv8-a+sve` or similar.
- The code generator (`src/codegen.cpp`) emits scalar polynomial coefficients
  and splices in `include/code.hpp` / `code64.hpp`. The coefficients are
  architecture-independent; the spliced headers are where the vector types
  appear.
- Verify on the target before trusting results: run `simd_test` and compare the
  ULP columns against a known-good x86 run. Lane-masking behaviour and
  fused-multiply-add contraction differ between the two architectures and will
  show up as small ULP differences.
