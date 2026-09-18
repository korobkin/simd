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

## 3. `round` rounds half to even, unlike libm

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
