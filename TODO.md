# TODO

Findings deliberately set aside. Each was observed and verified, but fixing it
was out of scope at the time. Items marked FIXED are kept rather than deleted,
because the cause is usually worth remembering.

When a fix lands, mark it here and re-capture the affected baseline artifact —
several of these are visible in `baseline/x86/semantics.txt`, so a fix changes
the reference the AArch64 port is diffed against.

---

## 1. `nextafter` cannot step downwards — FIXED

`include/simd.hpp:476` (f32), `:1671` (f64). Fixed 2026-09-17;
`baseline/x86/semantics.txt` re-captured (one line changed).

Kept here because the cause is worth remembering for the port. The old code was

```c
simd_i32 dir = y - x > simd_f32(0);
i += dir;
```

and comparisons in this library return `1` or `0`, never `-1`, so `dir` was `0`
whenever `y < x` — the bit pattern could only ever step up. Three further
defects were hiding behind that one: floats are sign-magnitude, so the
bit-space direction has to be reversed for negative `x`; `+-0` do not increment
into their correct neighbours and need special-casing; and a NaN `y` cannot be
detected with `y != y`, because `!=` returns `0` for NaN operands here —
`!(y == y)` works.

The replacement is verified against libm: 16/16 curated edge cases and 0
mismatches in 300,000 random comparisons, for both overloads. Note it uses
`sign(y) | 1` rather than `copysign`, because `copysign` is declared further
down the header than `nextafter`.

## 2. `ilogb` is wrong for negative inputs

`include/simd.hpp:651` (f32), `:1325` (f64).

`ilogb(-7.25f)` returns `258`; the correct answer is `2`. Every positive input
tested is correct, and `frexp` handles the same value correctly.

```c
simd_i32 i = (simd_i32&) x;
i >>= 23;
i -= 127;
```

The sign bit is never masked off, so for a negative argument it shifts down
into the exponent field. Masking the magnitude before the shift
(`i &= 0x7fffffff`) is the obvious fix. Worth checking the f64 path at the same
time, and worth confirming against a denormal, which this sequence also does
not handle.

## 3. `round` rounds half to even, unlike libm — DEFERRED

`include/simd.hpp:559` (f32), `:1221` (f64).

`round(2.5)` gives `2` and `round(-2.5)` gives `-2`. C's `round` is half away
from zero and gives `3` and `-3`. Across the eight inputs in the probe, this
function's output is byte-identical to `rint`, and `lround` agrees with it.

The cause is `_MM_FROUND_TO_NEAREST_INT`, which is round-half-to-even. There is
no single AVX intrinsic for half-away-from-zero; the usual construction is
`trunc(x + copysign(0.5, x))`.

Decide deliberately which behaviour is wanted. If half-to-even is intended,
that is defensible for numerical work, but the name should not be `round` — or
it should be documented, since callers will reasonably assume libm semantics.

Deferred until after the AArch64 port (decided 2026-09-17). Until then the
current behaviour is the specification: the NEON backend must use `vrndnq_f32`
(ties to even), **not** `vrndaq_f32` (ties away), even though the latter is
what the name `round` suggests. Substituting it would silently change `sin`,
`cos`, `exp` and `tgamma` through their argument reduction.

`semantics.txt` guards this — it covers `±0.5`, `±1.5` and `±2.5`, so a
ties-behaviour change moves four of its rows. `golden.txt` does not: its random
samples essentially never land on an exact half-integer. Diff both when
validating a port.

When this is eventually settled, the cost is a re-capture of `golden.txt` on
x86 *and* a re-verification on the ARM build, rather than the single re-capture
it would have taken before the port.

## 4. The f32 FMA check in the semantics probe is a bad test — FIXED

`tools/baseline/semantics.cpp:260`.

The comment asserted the residual "must be nonzero here". It is zero, and that
is correct: `(1 + 2^-12)*(1 - 2^-12) = 1 - 2^-24`, which is exactly
representable as a float, so there is no rounding error for the FMA to recover.

Not a library defect — the library is fine, and the f64 case immediately below
does exercise the path properly, returning exactly `-2^-54`.

Fixed 2026-09-17. Operands are now `1 + 2^-13` and `1 - 2^-13`, whose product
`1 - 2^-26` needs 26 bits below the leading one and so does not fit a float's
24-bit significand; it rounds to `1.0` and leaves a residual of `-2^-26`.

A first attempt used `1 + 2^-13` and `1 - 2^-11` and was wrong in the same way
as the original — that product is `1 - 2^-11 + 2^-13 - 2^-24`, which is again
exactly representable, and the residual was again zero. Worth remembering: it
is not enough for the operands to have differing exponents, or for the product
to *look* awkward. The test is whether the exact product needs more than 24
significant bits. Check a candidate with scalar `fmaf()` before trusting it.

Baseline re-captured on x86 (i5-1135G7, GCC 11.4) on 2026-09-17. The diff was
exactly the two predicted lines, and the predicted values were confirmed
independently with scalar `fmaf()` on x86 before the run:

```diff
-a*b       = 3f7fffff
-fma(a,b,-p)= 00000000  (exact residual; must be nonzero here)
+a*b       = 3f800000
+fma(a,b,-p)= b2800000  (exact residual; nonzero for these operands)
```

Nothing else in the file moved, so no x86 behaviour changed alongside it. The
`f64` lines below are unaffected, as is `golden.txt`, which does not exercise
`fma()` directly. The general re-capture procedure lives in
[ARCH.md](ARCH.md).

## 5. SIMDe's 256-bit FMA is not fused on NEON — WORKED AROUND

`include/simd.hpp`, the SIMDe branch at the top. Found during Phase 1 of the
AArch64 port; worth reporting upstream.

Of the four 256-bit FMA intrinsics this library uses, only
`simde_mm256_fmadd_ps` delegates to the fused 128-bit path
(`simde_mm_fmadd_ps`, which uses `vfmaq_f32`). The other three expand to a
separate multiply and add:

```c
simde_mm256_fmadd_pd(a,b,c)  ->  simde_mm256_add_pd(simde_mm256_mul_pd(a, b), c)
simde_mm256_fmsub_ps(a,b,c)  ->  simde_mm256_sub_ps(simde_mm256_mul_ps(a, b), c)
simde_mm256_fmsub_pd(a,b,c)  ->  simde_mm256_sub_pd(simde_mm256_mul_pd(a, b), c)
```

That rounds twice. `simde_mm_fmsub_ps` and `simde_mm_fmsub_pd` have no NEON
path either, so `fmadd` is the only usable fused building block.

This library cannot tolerate it. `simd_f32_2`/`simd_f64_2` implement
double-double arithmetic, and `two_product` depends on `fma(a, b, -a*b)`
recovering the exact rounding error of the product. Without a true FMA that
residual is zero and the extra precision silently disappears. The symptom was
`f64 resid = 0000000000000000` in the semantics probe where x86 gives
`bc90000000000000`, and, downstream, `acosh(simd_f32)` at 170 ULP against 5 on
x86 and `tgamma` at 78 against 9.

Worked around by overriding the three broken intrinsics in the SIMDe branch of
`simd.hpp`: `fmadd_pd` is built from two `simde_mm_fmadd_pd` halves, and both
`fmsub` forms as `fmadd` against a sign-flipped addend. Negation flips the sign
bit rather than subtracting from zero, so `-0` and NaN payloads survive.

With that in place the ARM build reproduces the x86 golden dump bit-for-bit
except `rsqrt`. Revisit if SIMDe fixes this upstream — the override is guarded
by nothing and will simply shadow a corrected implementation.

## 6. Pre-existing warnings in simd.hpp

`-Wall` on `include/simd.hpp` reports 11 warnings, down from 83 once item 7
removed the 72 strict-aliasing ones. None were introduced by the port; they are
on the x86 path too and are recorded here rather than fixed.
- **8 × sign-compare.** `for (int i = 0; i < size(); i++)` against a `size_t`
  `size()`.
- **2 × unused variable.** A dead `simd_f32 y;` in `scalbn`, and its f64 twin.
- **1 × control reaches end of non-void function.** `rint` switches on
  `fegetround()` with no `default:`, so an unexpected rounding mode returns
  nothing. Worth a `default: return round(x);`.

## 7. The type punning is undefined behaviour and has started to bite — FIXED

`include/simd.hpp`, ~70 sites: `ldexp`, `frexp`, `scalbn`, `hypot`, `copysign`,
`fabs`, `nextafter`, `ilogb` and others.

The library reinterprets a `simd_i32` as a `simd_f32` and back with casts of
the form

```c
simd_i32 i = (simd_i32&) x;      /* read the float's bits as integers  */
...
return (simd_f32&) i;            /* and back                           */
```

This is undefined behaviour. The two classes are layout-compatible, so it does
what is intended as long as the compiler does not act on the assumption that an
`int` and a `float` object cannot overlap — and it had been doing what was
intended only by luck.

Introducing the backend layer in Phase 2 changed the inlining enough for GCC to
start acting on that assumption. It reordered the punned read before the write
that produced it, and `ldexp(1.0f, 3)` returned `0x00000082` — a denormal,
being the exponent field left unshifted — where `8.0` (`0x41000000`) was
expected. `ldexp`'s own source was untouched by the refactor.

Worth noting how it was caught. `golden.txt` does not exercise `ldexp` and
showed nothing at all; every one of its 48 functions still matched. The
`semantics.txt` diff is what surfaced it. This is the concrete case for the
rule that a port diffs both files.

**Fixed 2026-09-17**, and `-fno-strict-aliasing` is gone with it.

The casts are replaced by `to_bits()` and `from_bits()`, which go through
backend primitives (`f32_as_i32` and friends, i.e. `_mm256_castps_si256` on
x86, `vreinterpretq_*` on NEON). That is better than `memcpy` or
`std::bit_cast`: it is a register-level reinterpret with no memory traffic, so
it expresses exactly what the original casts meant without the round trip a
byte-copy would imply.

Also applies to the generated code. `src/codegen.cpp` emitted 22 of these casts
into `math.cpp`, and now emits `to_bits`/`from_bits` instead. The one raw
intrinsic it emitted, `_mm256_movemask_pd`, becomes a `movemask(simd_i64)`
helper, so the generated source no longer reaches into the backend or into
private members -- the `friend simd_f64 asin(simd_f64 x);` declaration that
existed only for that purpose is no longer needed.

Confirmation: the 72 strict-aliasing warnings in item 6 are gone, leaving 11.
On AArch64 `golden.txt` and `semantics.txt` are byte-identical across the
change, and the regenerated `math.cpp` differs in exactly 22 lines, every one
of them a punning site.

A third motivation had accumulated by the time this was fixed, beyond the UB
itself and the workaround: `-fno-strict-aliasing` perturbed GCC's alias
analysis on x86, which changed its FMA contraction decisions (item 8), which
made x86 and AArch64 disagree on about ten functions at the same commit. That
broke the cross-architecture bit-exactness the whole validation strategy rests
on. Removing the flag is what restores it.

## 8. Results depend on FMA contraction, so `golden.txt` is not stable across recompiles

Measured on AArch64, toggling `-ffp-contract` changes the output of **33 of the
48** functions in the golden dump. Contraction is not optional here — it is
load-bearing for accuracy:

| | `fast` | `off` |
|---|---|---|
| f32 `exp` | 4 ULP | **68 ULP** |
| f32 `erfc` | 9 ULP | **71 ULP** |
| f32 `cosh` | 2 ULP | 10 ULP |
| f32 `sinh` | 3 ULP | 9 ULP |

`-ffp-contract=fast` is now pinned explicitly on the `simd` target rather than
inherited from GCC's default, since Clang and some distributions default
differently and would silently lose that accuracy.

The consequence matters for how the baseline is used. Whether a given `a*b + c`
gets contracted is an optimiser decision, and anything that perturbs the
optimiser can change it — which is how adding `-fno-strict-aliasing` (item 7)
shifted about ten x86 functions by 1-2 ULP without any source change to them.

So `baseline/x86/golden.txt` is a bit-exact oracle **across architectures at a
fixed commit and fixed flags**, which is what the port needs it for. It is
*not* a regression test across compiler-flag or structural changes; those
require re-capturing it, with `simd_test` as the arbiter of whether accuracy
actually moved. Treat a golden diff after a flag change as "re-baseline and
check ULP", and a golden diff after an architecture change as "a bug".

## 9. The validation routine had no performance check

Phase 1 gave `simd_f32`/`simd_f64` a `union { __m256 v; float w[8]; }` so that
lanes could be reached under SIMDe, whose vector type cannot be subscripted.
That union costs **2.5x on x86**: the array member stops GCC promoting the
class to a vector register, and the generated Horner chains have dozens of
live temporaries, so they spill. AArch64 has 32 vector registers and absorbed
it, which is why it went unnoticed there.

Measured on the x86 machine, f32 `asin` speedup: 4.27x at `7522374`
(pre-port), 4.20x at `6f67320` (Phase 0), 1.71x at `92d16ca` (Phase 1). Fixed
by making the union conditional on `SIMD_BACKEND_LANE_UNION`, which only the
SIMDe backend defines; the native backend keeps the bare member and reaches
lanes through `lane()`, exactly as before the port.

**The real defect is the validation routine, not the union.** The standard
check after every change had been three diffs -- `math.cpp`, `golden.txt`,
`semantics.txt` -- all of which are correctness-only. A 2.5x regression sat
undetected across two phases and was found only because a golden diff prompted
a `simd_test` run for an unrelated reason.

Add a speed check, with one caveat that makes it easy to get wrong: **the
speed column is not comparable against a committed file.** It is a ratio of two
timings taken sequentially in one process on random inputs, and it varies with
machine state -- the same machine produced 1.7x and 4.2x on the same commit at
different moments. Compare A against B by building both commits and running
them back to back in the same session, never against
`baseline/x86/simd_test.txt`. The accuracy columns of that file are stable and
comparable; the speed column is not, and is kept only as a rough record.
