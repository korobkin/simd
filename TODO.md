# TODO

Findings deliberately set aside. Each was observed and verified, but fixing it
was out of scope at the time. Nothing here is fixed yet.

When a fix lands, note it and re-capture the affected baseline artifact —
several of these are visible in `baseline/x86/semantics.txt`, so a fix changes
the reference the AArch64 port is diffed against.

---

## 1. `nextafter` cannot step downwards

`include/simd.hpp:476` (f32), `:1671` (f64).

`nextafter(1.0f, 0.0f)` returns `0x3f800000` — the input unchanged. libm gives
`0x3f7fffff`. Stepping upwards is correct.

```c
simd_i32 dir = y - x > simd_f32(0);
simd_i32 i = (simd_i32&) x;
i += dir;
```

Comparisons in this library return `1` for true and `0` for false, never `-1`
(see `baseline/x86/semantics.txt`). So `dir` is `0` whenever `y < x`, and the
integer representation is incremented or left alone — never decremented.

A fix needs `dir` in `{-1, +1}`, e.g. `2*(y > x) - 1`, with the `x == y` case
decided separately. Note this is one of the places that would silently change
meaning if a port switched comparisons to the all-ones convention.

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

## 4. The f32 FMA check in the semantics probe is a bad test

`tools/baseline/semantics.cpp:260`.

The comment asserts the residual "must be nonzero here". It is zero, and that
is correct: `(1 + 2^-12)*(1 - 2^-12) = 1 - 2^-24`, which is exactly
representable as a float, so there is no rounding error for the FMA to recover.

Not a library defect — the library is fine, and the f64 case immediately below
does exercise the path properly, returning exactly `-2^-54`. Only the test
input and its comment need changing: pick operands whose product is not
representable, e.g. `1 + 2^-13` and `1 - 2^-13`.

Flagged so nobody debugs the FMA implementation chasing this.
