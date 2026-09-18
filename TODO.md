# TODO

Findings deliberately set aside. Each was observed and verified, but fixing it
was out of scope at the time. Items marked FIXED are kept rather than deleted,
because the cause is usually worth remembering.

When a fix lands, mark it here and re-capture the affected baseline artifact —
several of these are visible in `baseline/x86/semantics.txt`, so a fix changes
the reference the AArch64 port is diffed against.

---
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

**Worth reporting upstream, and not yet reported.** The bug is small, clearly
bounded and has an obvious fix: `simde_mm256_fmadd_ps` already delegates to the
fused 128-bit path under `SIMDE_NATURAL_VECTOR_SIZE_LE(128)`, and the other
three want the same treatment. `simde_mm_fmsub_ps`/`_pd` need a NEON path too,
since they currently have none. A reproducer is a single `fma(a, b, -a*b)` on
values whose product is inexact -- the residual comes back zero where a true
FMA recovers it, which is `tools/baseline/semantics.cpp`'s `fma exactness`
section. The user-visible consequence here was `acosh(simd_f32)` at 170 ULP
against 5, because double-double arithmetic silently loses its extra precision.

With that in place the ARM build reproduces the x86 golden dump bit-for-bit
except `rsqrt`. Revisit if SIMDe fixes this upstream — the override is guarded
by nothing and will simply shadow a corrected implementation.

## 6. Pre-existing warnings in simd.hpp — `rint` FIXED, rest left

`-Wall` on `include/simd.hpp` reports 11 warnings, down from 83 once item 7
removed the 72 strict-aliasing ones. None were introduced by the port; they are
on the x86 path too and are recorded here rather than fixed.
- **8 × sign-compare.** `for (int i = 0; i < size(); i++)` against a `size_t`
  `size()`.
- **2 × unused variable.** A dead `simd_f32 y;` in `scalbn`, and its f64 twin.
- **1 × control reaches end of non-void function — FIXED.** `rint` switched on
  `fegetround()` with no `default:`, returning nothing for any rounding mode
  outside the four standard ones. Both overloads now fold `FE_TONEAREST` into a
  `default:`. No behaviour change for the four valid modes. The remaining 10
  warnings are cosmetic and are left alone.

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

## 10. x86 and AArch64 differ in five f64 functions by <= 2 ULP

`expm1`, `sinh`, `tanh`, `asinh` and `atanh`, all double precision, differ by
1-2 ULP in roughly 5% of samples. Everything else matches bit for bit except
`f32 rsqrt`, which is a reciprocal estimate and implementation-defined on both
sides.

All five are one root: `exp2(simd_f64)` contains a single bit-reinterpret, and
whether the surrounding arithmetic is contracted into FMAs differs slightly
between the two targets.

```
exp2(simd_f64)
  `- exp(simd_f64)      calls exp2
       `- expm1(simd_f64)
            |- sinh  |- tanh  |- asinh  `- atanh
```

`exp2` and `exp` do not show a difference themselves: their own 512 samples do
not land on a case where it matters. `expm1`'s do, and the four functions built
on it inherit it. `simd_test` accuracy is identical on both architectures, so
neither side is the wrong one -- contraction is an optimiser choice and both
results are legitimate.

Not worth chasing further. The divergence is bounded at 2 ULP, confined to one
dependency chain, and understood. Pinning it would mean rewriting the
contraction-sensitive expression in the generator with explicit `fma()` calls,
which is a change to the emitted mathematics for a cosmetic gain.

**`baseline/x86/golden.txt` is current.** x86 at HEAD reproduces it exactly, so
it is a valid oracle; the AArch64 NEON build is the side that differs, in the
six functions above.

Two earlier readings of this were wrong and are worth recording, because both
came from reasoning about which build had moved rather than measuring it. When
this was first noticed, after item 7 removed the punning, it was x86 that
differed from the baseline and AArch64 that matched -- the opposite of now.
Fixing the out-of-bounds lane conversions (item 13) moved both builds, x86 back
onto the baseline and NEON off it. The prediction that they had "converged" was
made without re-measuring either and was simply wrong.

## 13. Add a sanitizer run to the validation routine

Every check the project had was output-based -- `math.cpp`, `golden.txt`,
`semantics.txt` and `simd_test` all compare numbers. None of them can see a
program that computes the right answer by the wrong means.

Two defects were found the first time AddressSanitizer and UBSan were pointed
at it, both invisible to the existing checks:

- `simd_f64(const simd_i64&)` and `simd_i64(const simd_f64&)` assigned lanes 0
  through 3 unconditionally. With two lanes that is an out-of-bounds write on
  every call, and `golden` was producing bit-exact output while overflowing a
  stack buffer. Fixed; note that fixing it shifted five f64 functions by 1-2
  ULP, so some of the earlier bit-exactness was being achieved *through* the
  undefined behaviour.
- The factorial overflow above, and confirmation that item 11's `blend`
  divergence is UB rather than a difference of opinion between backends.

So the routine should be: the correctness diffs, the speed A/B, **and**

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
make && ./golden >/dev/null && ./semantics >/dev/null
```

Both should be **completely silent**; they are as of item 11, which removed the
last report. `simd_test` under ASan is slow enough to want a reduced
`N_bit_shift`, and the probes cover the same code paths.
## Closed

Fixed or resolved, and removed from the list above. Numbers are kept because
comments in the source cite them; `git log` has the detail.

- **1.** `nextafter` could only step upwards; sign-magnitude, +-0 and NaN were wrong with it
- **2.** `ilogb` shifted a negative argument's sign bit into the exponent field
- **3.** `round` ties to even, not libm's ties-away -- kept deliberately, documented in the README
- **4.** the probe's f32 FMA check used operands whose product was exactly representable
- **7.** the `(simd_i32&)` type punning was undefined behaviour and had begun to miscompile
- **11.** `blend` negated its mask, which is undefined for INT_MIN -- now an explicit `mask > 0`
- **12.** the generator built factorials in an `int` and overflowed at 13!

All seven were pre-existing defects on the x86 path. None was introduced by
the AArch64 port; the port is what made them visible.
