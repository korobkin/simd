# Architecture support

> **Status.** The survey below describes the situation before the AArch64 port
> started. As of Phase 1 the library builds and runs on AArch64 through SIMDe,
> reproducing the x86 reference dump bit-for-bit except `rsqrt`. A native NEON
> backend is Phase 2. See [PORT-AARCH64.md](PORT-AARCH64.md); the analysis here
> is kept because it is still the reasoning behind that plan.

Originally this library was x86_64-only. `include/simd.hpp` includes `<immintrin.h>` and
uses 32 `__m256`/`__m128` types and ~150 `_mm*` intrinsics, with no alternative
code path; `CMakeLists.txt` passes `-march=native -mavx2` unconditionally.

On AArch64 that failed immediately:

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

# Capturing an x86 baseline

The last note above says to compare ULP columns against a known-good x86 run.
This section is the concrete procedure. **It has been run** -- the artifacts
are in `baseline/x86/` and are what the AArch64 port is validated against; see
[PORT-AARCH64.md](PORT-AARCH64.md) for how they are used. The procedure is
kept because the baseline has to be re-captured whenever a fix changes
behaviour on x86.

## Why `simd_test` output is not enough

It is worth capturing, but it cannot serve as the reference on its own:

- **It is not reproducible.** `src/test.cpp:1229` calls `srand(time(NULL))`,
  so every run draws different inputs. Two runs on the *same* machine do not
  produce the same numbers, which makes a cross-architecture `diff`
  meaningless.
- **It is aggregate.** It reports average and maximum ULP per function. A port
  that is wrong for one narrow input range but fine elsewhere can easily hide
  inside an average taken over two million samples.
- **It never tests the primitives.** It exercises the transcendental functions
  against libm. It never touches the comparison/mask/shift/permute/gather
  layer underneath — which is exactly the layer a NEON port rewrites, and
  where the unresolved 1-vs-`-1` comparison convention lives.

So: capture `simd_test`, and capture three other things alongside it.

## What to capture

Five artifacts, into `baseline/x86/`:

| file | what it is | why |
|---|---|---|
| `simd_test.txt` | `simd_test` stdout | headline ULP + speedup, for the README table |
| `semantics.txt` | primitive conformance probe | settles the mask convention and the other per-primitive questions empirically |
| `golden.txt` | bit-exact dump on fixed inputs | the oracle the ported backend is diffed against |
| `math.cpp` | the generated `math.cpp` | the 8-lane generated source; needed when re-parameterising the generator for 4 lanes |
| `env.txt` | compiler, flags, CPU | records which ISA the baseline was actually built with |

Two programs for this already exist in the repo:
[tools/baseline/semantics.cpp](tools/baseline/semantics.cpp) and
[tools/baseline/golden.cpp](tools/baseline/golden.cpp).

`golden.cpp` is designed to be diffable across architectures: it uses a fixed
LCG rather than `rand()`, and prints **one line per scalar element** rather
than per vector, so an 8-lane AVX2 run and a 4-lane NEON run yield files with
identical line counts that `diff` compares directly.

## Procedure

```bash
# 0. Unmodified checkout, release build.
git status                       # confirm clean; record the commit
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..
mkdir -p baseline/x86

# 1. Environment. -march=native is what the build actually uses, so record
#    what it expands to. _mm256_cvtpd_epi64 (simd.hpp:1230) needs AVX512DQ
#    + AVX512VL, so note whether those appear.
{
  echo "== commit =="; git rev-parse HEAD; git status --short
  echo "== g++ =="; g++ --version | head -1
  echo "== cmake =="; cmake --version | head -1
  echo "== cpu =="; lscpu | head -25
  echo "== -march=native expansion =="
  gcc -march=native -dM -E - </dev/null | grep -E '__AVX|__FMA|__SSE4' | sort
} > baseline/x86/env.txt 2>&1

# 2. The generated source. Architecture-independent coefficients, 8-lane
#    packing -- both matter for the port.
cp build/generated_code/src/math.cpp baseline/x86/math.cpp

# 3. simd_test. Takes a while (N = 1 << 21 samples per function) and needs
#    a few GB; the numbers are statistical, so one run is enough.
./build/simd_test 2>&1 | tee baseline/x86/simd_test.txt

# 4. The two probes.
g++ -O2 -std=c++20 -DNDEBUG -march=native -mavx2 -Iinclude \
    tools/baseline/semantics.cpp -Lbuild -lsimd -o build/semantics
./build/semantics > baseline/x86/semantics.txt

g++ -O2 -std=c++20 -DNDEBUG -march=native -mavx2 -Iinclude \
    tools/baseline/golden.cpp -Lbuild -lsimd -o build/golden
./build/golden > baseline/x86/golden.txt

# 5. Commit.
git add baseline/x86 && git commit -m "baseline: x86_64 reference run"
```

## Notes for re-running this

- **Do not "fix" anything in `include/` or `src/` in the same pass.** The point
  is to record current behaviour, including the parts that look wrong; record
  them in [TODO.md](TODO.md) instead. If a fix does land, re-capture the
  affected artifact in its own commit and say which files changed, so the
  reference the port is diffed against never drifts silently.
- If `simd_test` is OOM-killed, lower `N_bit_shift` (`src/test.cpp:145`) and
  note the value used in `simd_test.txt`.

## What the first run established

Run on an i5-1135G7 (Tiger Lake), GCC 11.4, at commit `7522374`.

- **Comparisons return 1 or 0, never -1.** All sixteen comparison operators end
  with `return -result;`. `blend` agrees, and the operators built on top of
  them work as written. There is no inconsistency here, contrary to what was
  suspected from reading the intrinsics alone.
- **NaN compares false everywhere, `!=` included**, since the ordered
  predicates are used. Code in the header depends on this.
- **`-march=native` enabled AVX512DQ and AVX512VL** on this machine, so
  `_mm256_cvtpd_epi64` (`simd.hpp:1230`) compiled natively. A plain-AVX2 host
  would not have built it -- the AVX2-only claim in this file's opening is
  optimistic.
- **`simd_i32::operator*` really is an even-lane multiply** (lanes 1, 3, 5, 7
  come back zero), and **`operator>>` is a logical shift on a signed type**
  (`(-8) >> 3` gives 536870911).
- Four findings were set aside into [TODO.md](TODO.md); one of them,
  `nextafter`, has since been fixed and the baseline re-captured.
