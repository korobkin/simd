# Porting plan: AArch64

Companion to [ARCH.md](ARCH.md), which surveys the ARM SIMD landscape. This
document is the concrete plan for this repository on the machine it is being
written on.

## Target machine

```
Cortex-X925 (10 cores) + Cortex-A725 (10 cores), aarch64
NEON: mandatory, 128-bit
SVE2: present -- but the hardware vector length is 128 bits
GCC 13.3, CMake 3.28.3
```

The SVE vector length was measured, not assumed:

```c
#include <arm_sve.h>
printf("%ld\n", (long)(svcntb()*8));   /* -> 128 */
```

**This settles the backend question.** SVE here is the same width as NEON, so
it buys no throughput -- only a nicer programming model (predication, native
gather). It is not worth the added complexity for a first port. Target NEON,
and leave SVE as a later experiment for wider hardware.

## What the survey missed

ARCH.md lists `N_bit_shift` and the test buffers as the width-dependent spots.
There are three more, and one of them is the largest single cost in the port.

### 1. The code generator hard-codes 8 lanes (the big one)

`src/codegen.cpp` does not merely emit architecture-independent coefficients.
For two functions it *packs several coefficient sets into one 8-lane vector*
and addresses them with `permute`:

```c
/* codegen.cpp:363 -- erf(simd_f32): two sets of 4 packed into 8 lanes */
i1 = i0 + simd_i32(4);
y = co[N/2].permute(i0 or i1);

/* codegen.cpp:804-806 -- asin(simd_f32): four sets of 2 packed into 8 lanes */
i1 = i0 + simd_i32(2);
i2 = i0 + simd_i32(4);
i3 = i0 + simd_i32(6);
```

and emits the tables as literal 8-float initializer lists
(`codegen.cpp:337`, four values per iteration over a stride-2 loop).

On NEON `simd_f32` holds 4 floats, so `i0 + 4` indexes past the end of the
vector. Both the packing factor and the emitted table shape have to become
functions of the lane count.

`asin(simd_f64)` is width-dependent a third way, and worse. It converts a lane
mask to an integer and uses it to index a table with **one row per possible
mask**, i.e. `2^lanes` rows (`codegen.cpp:1589`, a literal `16` for 4 f64
lanes, each row `4` wide):

```c
i = -i;
j = _mm256_movemask_pd(((simd_f64&) i).v);   /* 0..15 */
y = co[j][45];
```

Two consequences. The table is `2^lanes x N x lanes`, so it *shrinks* on NEON
(4 rows of 2, not 16 of 4) but the emission loop has to be driven by the lane
count. And this is **the one place a raw intrinsic reaches the generated
code** -- everything else in `math.cpp` is written in terms of the public API.
`simd.hpp:1211` carries a `friend simd_f64 asin(simd_f64 x);` declaration that
exists solely so the generated function can reach the private `v`. The backend
layer has to expose a `movemask` primitive, and the generator has to emit a
call to that rather than to `_mm256_movemask_pd`.

So it is three generated functions, not two -- `erf(simd_f32)`,
`asin(simd_f32)`, `asin(simd_f64)` -- and it is generator work, not header
work.

### 2. `CHECK_ALIGNMENT(this, 32)` appears 128 times

The literal `32` is the AVX2 object size. On NEON the objects are 16 bytes, so
every debug build would `abort()` on the first operation. This needs to become
a per-class constant rather than 128 edited call sites.

### 3. `_mm256_cvtpd_epi64` is not AVX2

`simd.hpp:1230` (`lround(simd_f64)`) uses `_mm256_cvtpd_epi64`, which requires
AVX512DQ + AVX512VL. The build only works today because `-march=native` happens
to enable it on the developer's machine; `-mavx2` alone would not compile it.
Worth knowing before assuming the x86 build is a clean AVX2 baseline.

## Semantics the port must reproduce

The x86 baseline settles this empirically; see
[baseline/x86/semantics.txt](baseline/x86/semantics.txt).

**Comparisons return 1 or 0, never -1.** Every one of the sixteen comparison
operators ends with `return -result;`, negating the all-ones mask the intrinsic
produces. So the convention is uniform, `blend` agrees with it (a mask of `1`
selects `b`), and the operators built on top -- `simd_i32::operator>=` as
`((==) + (>)) > 0`, `fdim` as `(x > y) * (x - y)` -- work as written.

This matters for the port because **it is not automatic on NEON either**.
`vceqq_f32` and friends also yield all-ones, so the negation has to be carried
across deliberately. Getting it wrong flips every branch in the generated code
at once, which is the kind of failure that shows up as uniformly absurd ULP
columns rather than as a subtle regression.

`blend` needs equal care. It resolves its mask as `mask = -mask` followed by a
sign-bit select, so the useful range is `0`/`1` but the *observed* behaviour
for other values is whatever the sign bit says -- the probe records `2` as
true and `-2` as false. A NEON `vbslq_f32` takes a full-width mask, so the
port has to build one (negate, then broadcast the sign bit) rather than pass
the integer straight through.

**NaN comparisons return 0, including `!=`.** The intrinsics use the ordered
predicates, so every comparison against a NaN is false, and negating false
gives 0. IEEE would have `!=` be true. The library depends on the current
behaviour -- `nextafter` detects a NaN with `!(y == y)` precisely because
`y != y` does not work here -- so the port must reproduce it rather than
quietly adopt the IEEE reading.

**Two primitives behave unexpectedly, confirmed by the probe:**

- `simd_i32::operator*` uses `_mm256_mul_epi32`, a 32x32->64 even-lane
  multiply. The probe shows lanes 1, 3, 5 and 7 coming back as zero. It
  appears unused by the math paths. The NEON version should implement the
  intended lane-wise `vmulq_s32` and say so in the commit, rather than
  faithfully reproducing a result nobody can want.
- `operator>>` is a *logical* shift despite `simd_i32` being signed
  (`_mm256_srlv_epi32` / `_mm256_srli_epi32`): `(-8) >> 3` gives 536870911.
  Several bit-manipulation routines in the header rely on this. NEON's
  `vshlq_s32` is arithmetic, so the port must use the unsigned form to match.

[TODO.md](TODO.md) is the register for defects found and deliberately not
fixed. Two are open, and both are **deferred until after the port** by
decision:

- `ilogb` is wrong for negative inputs (`ilogb(-7.25)` returns 258). Isolated:
  nothing in the generated code calls it, so fixing it changes `semantics.txt`
  and nothing else.
- `round` is half-to-even, where C's `round` is half-away-from-zero. **Not
  isolated.** It appears nine times in the generated `math.cpp`, in the
  argument reduction of `sin`, `cos`, `exp` and `tgamma`. Changing it changes
  those four functions at exact half-integers.

Deferring `round` is affordable but not free, and it leaves a live trap in
Phase 2: `vrndaq_f32` is the instruction a function called `round` invites, and
substituting it would change four transcendentals. **The port must therefore
reproduce the current behaviour exactly -- `vrndnq_f32` -- and any change to it
is a separate, deliberate commit, not a side effect of the port.**

What makes this safe to defer is that `semantics.txt` already pins it: the
probe covers +-0.5, +-1.5 and +-2.5, so a ties-behaviour change moves four rows
of that file. `golden.txt` would not catch it -- exact half-integers
essentially never appear in 512 random draws, so the dump can be unchanged
while the behaviour has changed. That is why the Phase 2 validation below
diffs both files and not just the golden dump.

## Strategy

The reference run ARCH.md asks for **has been captured**; it is in
[baseline/x86/](baseline/x86/), from an i5-1135G7 under GCC 11.4. Everything
below is validated against it.

`simd_test` alone does not suffice as that reference: it seeds from
`time(NULL)` ([test.cpp:1229](src/test.cpp#L1229)) so it is not reproducible,
it reports only aggregate ULP, and it never exercises the comparison/mask/
permute/gather layer that a NEON port actually rewrites. Hence the two extra
probes: `semantics.txt` (primitive conformance, which is what settled the
comparison convention above) and `golden.txt` (bit-exact output on fixed
inputs, printed per scalar element so an 8-lane and a 4-lane run diff
directly, with an FNV summary per function to localise a mismatch).

With the baseline in hand, the sequencing question is whether to go straight
to native NEON or via SIMDe first.

**The case for the SIMDe step: it preserves the AVX2 lane counts.** Under
SIMDe `simd_f32::size()` stays 8, so the codegen packing, the `permute`
strides, `N_bit_shift` and the buffer sizing all remain valid and *no
generated code changes at all*. That separates the two independent halves of
the port -- swapping the instruction set, and changing the vector width -- so
that when something breaks it is clear which half did it. It also produces a
working ARM build early, which is worth something on its own if anything
downstream depends on this library.

**The case against: it is a phase you throw away.** With a golden dump from
x86 there is a usable oracle regardless, so SIMDe is no longer load-bearing --
it is a risk-reduction step, not a necessity. If the appetite is for one
focused push rather than two, going straight to Phase 2 is defensible; the
cost is that a failing ULP column could be either an intrinsic translation bug
or a width bug, with no cheap way to tell them apart.

The recommendation is to keep the SIMDe phase, because the width fallout
(section 1 above) is concentrated in the code generator, which is the hardest
part of the port to debug -- and SIMDe lets every other piece be proven
correct before touching it. But this is a judgement call, not a constraint.

```
Phase -1  x86 baseline capture   -> DONE; baseline/x86/
Phase 0   build unblock          -> DONE; cmake configures per architecture
Phase 1   SIMDe backend          -> DONE; baseline/aarch64-simde/
Phase 2   native NEON backend    -> diffed against baseline/x86/golden.txt
Phase 3   SVE (optional)         -> only worthwhile on wider hardware
```

## Phase 0 -- build unblock

Nothing compiles here yet, for two reasons unrelated to SIMD.

**mpfr/gmp headers are absent.** The runtime libraries are present
(`/lib/aarch64-linux-gnu/libmpfr.so.6`, `libgmp.so.10`) but the headers are
not, and there is no root on this machine. They do exist in the conda trees,
e.g. `/scratch2/shared/miniconda/telf/include/`. `CMakeLists.txt` links them by
bare name:

```cmake
target_link_libraries(simd_codegen PRIVATE mpfr gmp pthread)
```

Replace with `find_path`/`find_library` so an out-of-tree prefix can be pointed
at via `CMAKE_PREFIX_PATH`. Note the library target itself does **not** need
mpfr -- only `simd_codegen` and `simd_test` do.

**The compile flags are unconditional.** `target_compile_options(simd PUBLIC
-march=native -mavx2)` fails immediately on ARM. Make it architecture-
dependent, keyed on `CMAKE_SYSTEM_PROCESSOR`, and update the two `-march=native
-mavx2` examples in the README's Usage section to match.

## Phase 1 -- SIMDe backend -- DONE

The library, the generator and `simd_test` all build and run on AArch64. The
include is switched per architecture, so x86 never sees SIMDe:

```c
#if defined(__x86_64__) || ...
#include <immintrin.h>
#else
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#endif
```

CMake finds SIMDe with `find_path`, falling back to fetching v0.8.2, and
`-DSIMDE_INCLUDE_DIR=` overrides both.

**Result: the ARM build reproduces `baseline/x86/golden.txt` bit-for-bit for
47 of 48 functions.** The exception is `f32 rsqrt`, which was predicted:
`_mm256_rsqrt_ps` is a ~12-bit reciprocal estimate whose result is
implementation-defined, so x86 and NEON simply give different approximations.
All 512 samples differ, by at most 3928 ULP -- about 9e-5 relative, within
spec for both. Nothing else in the library uses `rsqrt`, which is why no other
function moved.

`semantics.txt` differs in exactly one place, `alignof` reporting 16 rather
than 32 (below). `simd_test`'s accuracy columns match x86 across the board.
Speed is a different story -- many double-precision cases now run slower than
scalar, which is expected when every 256-bit operation is two 128-bit ones.
Phase 1 makes no speed claim.

### What had to be fixed beyond the include swap

- **SIMDe's 256-bit FMA is not fused on NEON.** Three of the four FMA
  intrinsics used here expand to a separate multiply and add, which destroys
  the double-double arithmetic in `simd_f32_2`/`simd_f64_2` and took `acosh`
  from 5 ULP to 170. Overridden in the SIMDe branch; see TODO item 5. This was
  the whole value of the phase -- it is a subtle, silent, high-consequence
  failure, and it surfaced here for the cost of a header swap rather than in
  the middle of a hand-written NEON backend.
- **`v[i]` subscripting (20 sites).** `simd_f32`/`simd_f64` stored a bare
  `__m256`/`__m256d` and indexed it with the GCC vector-subscript extension,
  which `simde__m256` does not support. Both classes now carry the same
  `union { __m256 v; float w[8]; }` layout `simd_i32` always had.
- **`reduce_sum`** reinterpreted the vector as two `__m128` halves through
  pointer arithmetic. Rewritten over the union, preserving the summation order
  exactly -- a different order rounds differently and would have moved the
  baseline.
- **`simd_f64::permute`** used `__builtin_shuffle`, which needs a GCC vector
  type. Written out as the equivalent loop.
- **`_mm256_cvtpd_epi64`** is the one intrinsic of the 70 that SIMDe does not
  provide, exactly as predicted -- it is AVX512DQ+VL, not AVX2. Emulated for
  its single caller, `lround(simd_f64)`.
- **`CHECK_ALIGNMENT`** hardcoded 32 at all 128 sites, and SIMDe's types are
  16-byte aligned, so every debug build would have aborted immediately. The
  macro now takes the alignment from the pointee and ignores the argument,
  which fixes all 128 sites at once and is correct for Phase 2 as well. This
  was listed below as width fallout; it is done.
- **`src/test.cpp`** had its own vestigial `#include <immintrin.h>`, removed;
  it uses no intrinsics directly.

### Known differences from the x86 baseline

Both are expected and neither indicates a defect:

| where | x86 | AArch64/SIMDe | why |
|---|---|---|---|
| `semantics.txt`, `alignof` | 32 | 16 | SIMDe's 256-bit types are two 128-bit NEON vectors; `sizeof` is still 32 |
| `golden.txt`, `f32 rsqrt` | -- | differs, <= 3928 ULP | reciprocal estimate, implementation-defined on both |

`baseline/x86/golden.txt` remains the canonical oracle for Phase 2; the ARM
dump is not committed because it is identical to it apart from `rsqrt`.

## Phase 2 -- native NEON backend

Only now does the width change. The work:

**Mechanical.** 70 distinct intrinsics across 149 call sites, 32 `__m256`/
`__m128` type mentions, four classes (`simd_i32` 26-270, `simd_f32` 282-455,
`simd_i64` 723-947, `simd_f64` 950-1128) plus `simd_f32_2`/`simd_f64_2`.
Most map one-to-one: `_mm256_add_ps` -> `vaddq_f32`, `_mm256_sqrt_pd` ->
`vsqrtq_f64`, `_mm256_blendv_ps` -> `vbslq_f32`, `_mm256_movemask_pd` -> a
narrow-and-reduce. NEON types keep GCC vector subscripting, so the `v[i]` idiom
works natively.

**Needs care:**

- **`fmsub`.** `_mm256_fmsub_ps(a,b,c) = a*b - c`; NEON `vfmaq_f32(c,a,b) =
  c + a*b`. So `vfmaq_f32(vnegq_f32(c), a, b)`. This sits inside
  `simd_f32_2::two_product`, where the double-double arithmetic depends on the
  FMA being exact -- and on the compiler *not* contracting the neighbouring
  `sub` into another FMA. Build with `-ffp-contract=off` for these paths; GCC
  on ARM defaults to `fast`.
- **Rounding: NEON has both nearest-modes, and the obvious choice is the wrong
  one.** Measured on the target and checked against the baseline:

  | current | NEON | behaviour |
  |---|---|---|
  | `round`, `rint`, `nearbyint` (`_MM_FROUND_TO_NEAREST_INT`) | `vrndnq_f32` | ties to **even** |
  | -- | `vrndaq_f32` | ties **away** -- what C's `round` means |
  | `trunc` (`_MM_FROUND_TO_ZERO`) | `vrndq_f32` | toward zero |
  | `floor` | `vrndmq_f32` | toward -inf |
  | `ceil` | `vrndpq_f32` | toward +inf |

  A function called `round` invites `vrndaq_f32`, which would silently change
  `sin`, `cos`, `exp` and `tgamma` through their argument reduction. Use
  `vrndnq_f32` to match the baseline; see TODO item 3 before deciding
  otherwise.
- **`gather`** has no NEON equivalent; emulate with scalar loads. This is on
  the hot path of the generated `erf`/`tgamma`/`asin` code, so it is the most
  likely place for the speedup column to regress.
- **`permute`** (`_mm256_permutevar8x32_ps`) becomes a 4-lane `vqtbl1q_u8`
  table lookup with byte-index expansion. `simd_f64::permute` already uses
  `__builtin_shuffle`, which ports for free.
- **`_mm256_mul_epi32`**, used for `simd_i32::operator*`, is a 32x32->64
  even-lane multiply, not a lane-wise int32 multiply -- `_mm256_mullo_epi32`
  is the lane-wise one. It appears unused by the math paths, so the NEON
  version should implement the *intended* `vmulq_s32` and the discrepancy
  should be noted in the commit rather than faithfully reproduced.

**Then the width fallout:** the codegen packing (section 1 above),
`N_bit_shift` and the `posix_memalign(..., 32, ...)` calls in `src/test.cpp`.
`CHECK_ALIGNMENT` (section 2) was already dealt with in Phase 1.

Note that Phase 1's FMA workaround goes away here: a native backend calls
`vfmaq_f32`/`vfmaq_f64` directly, so the fusion is explicit rather than
something to be checked for. The semantics probe's `fma exactness` section is
what catches a regression.

**Validation:** rebuild **both** probes and diff against
`baseline/x86/golden.txt` and `baseline/x86/semantics.txt`. Diffing only the
golden dump is not sufficient: it is built from random samples, so behaviour
that differs on special values -- rounding ties, signed zero, NaN, denormals --
can change without moving a single line of it. `semantics.txt` is the file that
pins those, and it is where a wrong rounding-mode choice (see TODO item 3) or
a lost comparison negation would surface.

The per-element output format means the 4-lane ARM run and the 8-lane x86 run
produce directly comparable files. Lane-masking and FMA contraction
differences show up as small ULP deltas -- expected; large ones are bugs. The
FNV summary at the end of the dump tells you which functions to look at
before reading 20k lines of hex.

## Phase 3 -- SVE (optional)

Not worthwhile on this machine (128-bit VL). Revisit on Neoverse V1/V2, Grace,
or Graviton3+, where the vector is 256 bits or wider. Note that vector-length
agnostic code cannot have a `constexpr size()`, which this API exposes and
`src/test.cpp` uses as a compile-time constant -- SVE is an API change, not
just a backend change.

## Effort

| Phase | Scope | Rough cost |
|---|---|---|
| -1 | x86 baseline capture | done |
| 0 | CMake: find mpfr/gmp, arch-conditional flags; README | done |
| 1 | SIMDe: swap header, union layout, `reduce_sum`, FMA fusion | done |
| 2 | Native NEON: 4 classes, 70 intrinsics, codegen re-parameterisation | the bulk |
| 3 | SVE | defer |

Phases -1 to 1 are complete: there is a correct, tested ARM build, making no
speedup claim. Phase 2 is where performance arrives, and it is the only phase
that touches the code generator.
