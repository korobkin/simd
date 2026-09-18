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
It used to carry a raw `_mm256_movemask_pd` and a `friend simd_f64
asin(simd_f64 x);` declaration to reach the private `v`; the generator now
emits a `movemask(simd_i64)` helper instead, so the generated source no longer
reaches into the backend. The remaining work is the table's shape.

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

**The full routine is four things, not one.** Diff both probe files; compare
speed as an A/B of two builds run back to back in one session, never against a
committed file (TODO item 9); and run the probes under AddressSanitizer and
UBSan (TODO item 13). That last one exists because every other check is
output-based and so cannot see a program that computes the right answer by
writing out of bounds -- which is not hypothetical here, and cost some of the
bit-exactness this document originally claimed.

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
Phase 2   native NEON backend    -> DONE; baseline/aarch64-neon/
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

## Phase 2 -- native NEON backend -- DONE

`simd_backend_neon.hpp`, selected with `-DSIMD_NATIVE_NEON=ON`. The lane counts
halve: 4 floats and 2 doubles against 8 and 4.

**42 of 48 functions are bit-exact against `baseline/x86/golden.txt`** -- at
half the vector width, on a different instruction set. The per-scalar-element
dump format is what makes that comparison possible at all: the 4-lane and
8-lane runs diff line for line.

The six that differ are all accounted for, and the baseline is current -- x86
at HEAD reproduces it exactly, so it is a valid oracle and AArch64 is the side
that differs. `f32 rsqrt` is a reciprocal estimate, implementation-defined on
both sides. The other five -- `expm1`, `sinh`, `tanh`, `asinh`, `atanh`, all
f64 -- are the chain in TODO item 10: one bit-reinterpret in `exp2(simd_f64)`
whose surrounding arithmetic is contracted slightly differently on the two
targets, worth 1-2 ULP in about 5% of samples. `simd_test` accuracy is
identical on both, so neither side is the wrong one.

An earlier run of this comparison reported 47 of 48 against the same baseline.
That was real but partly undeserved: two conversion constructors were writing
out of bounds, and removing those writes shifted exactly this f64 chain. Some
of that bit-exactness was being achieved *through* undefined behaviour. See
TODO item 13.

Speed, measured A/B against the SIMDe build in one session as TODO item 9
requires: **geometric mean 1.65x over 42 functions**, with the largest gains
where SIMDe was worst -- f32 `asin` 0.37x to 2.33x, f32 `erf` 0.88x to 4.19x,
f32 `atan` 0.32x to 1.47x. Most single-precision functions now beat scalar
libm by 2-5x. Double precision is mixed and several remain below 1.0x, which
is what two lanes buys. ULP is unchanged throughout.

### Where the obvious NEON instruction is the wrong one

Each of these is commented at its definition, because each would have been a
silent behaviour change rather than a compile error:

- **`vminq_f32`/`vmaxq_f32` implement IEEE minNum/maxNum**, returning the
  non-NaN operand. `_mm256_min_ps` is a plain `a < b ? a : b`, which returns
  `b` when either operand is NaN. Written as the select.
- **`vrndnq_f32` is ties-to-even** and matches `_MM_FROUND_TO_NEAREST_INT`.
  `vrndaq_f32` is ties-away, which is what C's `round` means and what the
  caller's name suggests -- and would have changed `sin`, `cos`, `exp` and
  `tgamma` through their argument reduction. See TODO item 3.
- **`_CMP_NEQ_OS` is ordered**, so a NaN operand compares false. Negating
  `vceqq` alone gives true there, so it is masked with "both operands ordered".
- **`blendv` tests the sign bit**, not whether the lane is nonzero.
- **The right shifts are logical** despite the signed element type.
- **`i32_mul` reproduces `_mm256_mul_epi32`'s even-lane 32x32->64 behaviour**
  rather than correcting it, so the backends agree and the baseline stays a
  clean oracle. Changing it is a separate decision affecting both.

### The generator, parameterised on the lane count

All three predicted functions needed it, and the third was worse than expected:

- `erf(simd_f32)` packs 4 branches per coefficient index, so a vector holds
  `size()/4` of them -- 2 on AVX2, 1 on NEON.
- `asin(simd_f32)` packs 2, so `size()/2` -- 4 on AVX2, 2 on NEON. Fixing it
  also fixed `atan`, which is built on it.
- `asin(simd_f64)` indexes a table with one row per lane mask: `2^lanes` rows
  of `lanes` wide, 4x2 here against 16x4 on x86. **It had been producing
  correct answers by accident.** Only the first 4 of the 16 rows are reachable
  with two lanes and they happen to hold the right values, while each row's
  4-wide initialiser was writing past its array element. Correct output,
  out-of-bounds writes -- the kind of thing no amount of diffing finds.

`simd_i64`'s bitwise operators were calling the `i32` primitives. That is
invisible on x86, where both integer vectors are `__m256i`, and a compile error
on NEON, where they are not -- the naming scheme doing its job.

Two conversion constructors, `simd_f64(const simd_i64&)` and its inverse,
assigned lanes 0 through 3 unconditionally -- an out-of-bounds write on every
call at two lanes. `golden` was producing bit-exact output while overflowing a
stack buffer, which no output comparison can detect; AddressSanitizer found it
immediately. TODO item 13 adds a sanitizer pass to the validation routine.

`src/test.cpp` needed no change: `posix_memalign` at 32 bytes is valid for
16-byte types, and `N_bit_shift` simply means AArch64 samples half as many
points, which is ample.

**The generated `math.cpp` is now backend-specific**, since `codegen.cpp` reads
the lane count from the type it was compiled against. Switching backends
requires a rebuild, which the dependency on `simd_codegen` already forces.

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
| 2 | Native NEON: backend, codegen re-parameterisation | done |
| 3 | SVE | defer |

Phases -1 to 2 are complete. There is a native AArch64 build that matches the x86
reference on 42 of 48 functions, with the six exceptions understood and bounded
at 2 ULP (`rsqrt`, plus one f64 dependency chain), and runs 1.65x faster than
the SIMDe one. What remains is optional: SVE on wider hardware, and the deferred
items in TODO.md.
