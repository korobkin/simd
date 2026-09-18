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
functions of the lane count. The damage is bounded -- exactly two generated
functions, `erf(simd_f32)` and `asin(simd_f32)` -- but it is real work, and it
is generator work, not header work.

### 2. `CHECK_ALIGNMENT(this, 32)` appears 128 times

The literal `32` is the AVX2 object size. On NEON the objects are 16 bytes, so
every debug build would `abort()` on the first operation. This needs to become
a per-class constant rather than 128 edited call sites.

### 3. `_mm256_cvtpd_epi64` is not AVX2

`simd.hpp:1230` (`lround(simd_f64)`) uses `_mm256_cvtpd_epi64`, which requires
AVX512DQ + AVX512VL. The build only works today because `-march=native` happens
to enable it on the developer's machine; `-mavx2` alone would not compile it.
Worth knowing before assuming the x86 build is a clean AVX2 baseline.

## A semantics problem to settle first

The codebase is written throughout as though comparisons return **1**, but the
intrinsics it uses return **all-ones (-1)**. Both conventions appear, sometimes
two lines apart in the same generated function:

```c
/* codegen.cpp:482-483, inside one function */
r   = blend(r, simd_f32(1) - r, r > simd_f32(0.5));              /* -1 */
sgn = blend(simd_f32(-1), simd_f32(1), simd_i32(floor(x0)) & 1); /*  1 */
```

`blend` resolves its mask as `mask = -mask` then `_mm256_blendv_ps`, which
tests the sign bit -- so a mask of `1` selects `b` and a mask of `-1` selects
`a`. The two lines above therefore disagree about what "true" means.

The same assumption shows up in the scalar operator definitions, which only
work with a 1/0 convention:

```c
simd_i32::operator>=  ->  ((*this == other) + (*this > other)) > simd_i32(0)
simd_i32::operator!=  ->  simd_i32(1) - (*this == other)
fdim(x, y)            ->  (x > y) * (x - y)
nextafter             ->  i += (y - x > simd_f32(0))
```

With `==` returning -1, `>=` on equal operands evaluates `-1 > 0` -> false, and
`fdim` returns the negated difference.

This is not an ARM issue -- it is there on x86 -- but it lands in the middle of
the port, because a NEON rewrite has to pick a convention, and picking one
silently changes results. **Decide it before porting, not during.** The
recommendation is to standardise on 1/0 (it matches `blend`, and it is what
every operator above already assumes), and to do it as a separate,
clearly-labelled commit on x86 semantics so the change is not entangled with
the architecture work.

## Strategy

An x86_64 machine is available, so the reference run ARCH.md asks for can be
captured directly. [ARCH.md](ARCH.md#capturing-an-x86-baseline) has the
procedure and [tools/baseline/](tools/baseline/) has the two probe programs.
Do that first -- it is cheap, and everything below is validated against it.

Note that `simd_test` alone does not suffice as that reference: it seeds from
`time(NULL)` ([test.cpp:1229](src/test.cpp#L1229)) so it is not reproducible,
it reports only aggregate ULP, and it never exercises the comparison/mask/
permute/gather layer that a NEON port actually rewrites. Hence the two extra
probes: `semantics.txt` (primitive conformance, settles the 1-vs-`-1`
question empirically) and `golden.txt` (bit-exact output on fixed inputs,
printed per scalar element so an 8-lane and a 4-lane run diff directly).

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
focused push rather than two, going straight to Phase 3 is defensible; the
cost is that a failing ULP column could be either an intrinsic translation bug
or a width bug, with no cheap way to tell them apart.

The recommendation is to keep the SIMDe phase, because the width fallout
(section 1 above) is concentrated in the code generator, which is the hardest
part of the port to debug -- and SIMDe lets every other piece be proven
correct before touching it. But this is a judgement call, not a constraint.

```
Phase -1  x86 baseline capture   -> on the x86 machine; see ARCH.md
Phase 0   build unblock          -> cmake configures and builds on ARM
Phase 1   SIMDe backend          -> correct ARM build, AVX2 lane counts
Phase 2   semantics fix          -> settle the 1/0 convention
Phase 3   native NEON backend    -> diffed against baseline/x86/golden.txt
Phase 4   SVE (optional)         -> only worthwhile on wider hardware
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

## Phase 1 -- SIMDe backend

Vendor SIMDe (header-only) and swap the include:

```c
#include <simde/x86/avx2.h>
#define SIMDE_ENABLE_NATIVE_ALIASES
```

Expect a small, contained set of fixes beyond the swap:

- **`v[i]` subscripting (18 sites).** `simd_f32`/`simd_f64` store a bare
  `__m256`/`__m256d` and index it with the GCC vector-subscript extension.
  SIMDe's `simde__m256` is a union type and is not subscriptable. Fix by giving
  those two classes the same `union { ...; float w[8]; }` layout `simd_i32`
  already has -- which is tidier than the status quo anyway.
- **`reduce_sum`** (`simd.hpp:587`, `:1684`) reinterprets the vector as two
  `__m128` halves via pointer arithmetic. Rewrite over the union.
- **`_mm256_cvtpd_epi64`** -- verify SIMDe's AVX512DQ/VL coverage includes it;
  if not, emulate (2 lanes, trivial).
- **Verify `_mm256_i32gather_ps`, `_mm256_i64gather_pd`,
  `_mm256_permutevar8x32_ps/_epi32`** are covered. These are the ones the
  generated code leans on hardest.

The 32-byte type punning (`(simd_i32&) x`, used pervasively) stays valid
because SIMDe keeps the 32-byte object size.

**Exit criterion:** `simd_test` runs, and `golden` and `semantics` rebuilt on
ARM reproduce `baseline/x86/golden.txt` and `baseline/x86/semantics.txt`.
Since SIMDe keeps 8 lanes and emulates AVX2 semantics, these should match
very closely -- `rsqrt` is the expected exception (reciprocal estimate
instructions are implementation-defined). Any other mismatch is a SIMDe
coverage gap worth understanding before moving on.

## Phase 2 -- semantics fix

With a runnable build in hand, settle the comparison convention described
above. Add a small conformance test (not the ULP sweep -- a direct assertion on
`==`, `!=`, `<`, `>=`, `blend`, `fdim`, `nextafter` for known inputs) so the
convention is pinned by a test rather than by reading. Expect some ULP columns
to *improve*; that is the signal the fix is right.

## Phase 3 -- native NEON backend

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
`CHECK_ALIGNMENT` (section 2), `N_bit_shift` and the `posix_memalign(..., 32,
...)` calls in `src/test.cpp`.

**Validation:** rebuild `golden` and diff against `baseline/x86/golden.txt`.
The per-element output format means the 4-lane ARM run and the 8-lane x86 run
produce directly comparable files. Lane-masking and FMA contraction
differences show up as small ULP deltas -- expected; large ones are bugs. The
FNV summary at the end of the dump tells you which functions to look at
before reading 20k lines of hex.

## Phase 4 -- SVE (optional)

Not worthwhile on this machine (128-bit VL). Revisit on Neoverse V1/V2, Grace,
or Graviton3+, where the vector is 256 bits or wider. Note that vector-length
agnostic code cannot have a `constexpr size()`, which this API exposes and
`src/test.cpp` uses as a compile-time constant -- SVE is an API change, not
just a backend change.

## Effort

| Phase | Scope | Rough cost |
|---|---|---|
| -1 | x86 baseline capture (on the x86 box) | small |
| 0 | CMake: find mpfr/gmp, arch-conditional flags; README | small |
| 1 | SIMDe: swap header, union layout, `reduce_sum`, verify coverage | moderate |
| 2 | Comparison convention + conformance test | small, but needs a decision |
| 3 | Native NEON: 4 classes, 70 intrinsics, codegen re-parameterisation | the bulk |
| 4 | SVE | defer |

Phases -1 to 2 give a correct, tested ARM build with no speedup claim. Phase 3 is
where performance arrives, and it is the only phase that touches the code
generator.
