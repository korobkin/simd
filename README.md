# simd

This is a library that provides fast mathematical functions, 
such as `sqrt, sin, cos, erf` etc., for the SIMD (Single Instruction,
Multiple Data) processor capabilities and SIMD data types.
SIMD instructions were introduced for `x86_64` architecture (AVX,
AVX2, AVX512, some AMD versions).

These library introduces the capability to use the math functions
similar to what is provided in `<math.h>` for the `double` and `float`
types in C.


## Installation

Architectures: **x86_64** natively, requiring AVX2 and FMA, and **AArch64**
through [SIMDe](https://github.com/simd-everywhere/simde), which reimplements
the Intel intrinsics over NEON. The ARM build is correct -- it reproduces the
x86 results bit-for-bit apart from `rsqrt`, an implementation-defined
approximation -- but makes no speed claim, since every 256-bit operation
becomes two 128-bit ones. A native NEON backend is planned; see
[PORT-AARCH64.md](PORT-AARCH64.md).

On AArch64, CMake looks for SIMDe and fetches it if absent. To use an existing
checkout instead:

```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DSIMDE_INCLUDE_DIR=/path/to/simde
```

Prerequisites:
 - gmp, gmp-devel library;
 - mpfr, mpfr-devel library.

Both are needed by the code generator and the test; the library itself does
not link them.

To install, use the standard cmake procedure:

```bash
   mkdir build; cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make
```

If gmp and mpfr live outside the default search path -- in a conda environment,
say -- point CMake at the prefix:

```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/prefix
```

This will create an executable `simd_test`. If you run it, it should
display the results like this:
```
$ ./simd_test
5.000000e+00 nan
Testing SIMD Functions

Single Precision
name   speed        avg err      max err
  asin 6.382254e+00 4.199801e-01 3.000000e+00
  acos 8.652846e+00 2.972159e-01 2.000000e+00
  atan 3.004277e+00 4.158778e-01 6.000000e+00
 acosh 1.524083e+00 2.971838e-01 5.000000e+00

...
```

Here, the first column is the name of the transcendental function tested,
the second one is the speedup compared to the standard scalar types, and
the third and fourth columns are the average and max errors with respect to
the scalar type, in the units of ULP (=Unit in the Last Place for the FP
representation).

## Usage

Everything lives in namespace `simd`. Include `simd.hpp`, link `libsimd`, and
compile with the SIMD instruction set enabled. On x86_64:

```bash
g++ -std=c++20 -march=native -mavx2 -I../include main.cpp -L. -lsimd
```

On AArch64, drop `-mavx2` and add the SIMDe include path:

```bash
g++ -std=c++20 -march=native -I../include -I/path/to/simde main.cpp -L. -lsimd
```

From CMake, link the `simd::simd` target, which already carries the include
directory and the instruction-set flags chosen for the target architecture
(`-march=native -mavx2` on x86_64, `-march=native` plus the SIMDe include path
on AArch64):

```cmake
add_subdirectory(external/simd)
target_link_libraries(my_program PRIVATE simd::simd)
```

The library defines the following vector types: `simd_f32` and `simd_f64` for
floating point, plus the integer types `simd_i32` and `simd_i64`. Their width
depends on the instruction set and is available as the compile-time constant
`size()`; on AVX2 a `simd_f32` holds 8 floats and a `simd_f64` holds 4 doubles.
They behave like a scalar: arithmetic operators and math functions apply to
every lane at once, a scalar broadcasts to all lanes, and `operator[]` reads or
writes an individual lane.

```cpp
#include <simd.hpp>
#include <cstdio>

int main() {
    using namespace simd;

    simd_f64 x;
    for (size_t i = 0; i < simd_f64::size(); i++) {
        x[i] = 0.5*(double)(i + 1);
    }

    const simd_f64 y = erf(x) + exp(-x*x);   /* whole vector at once */
    const simd_f64 z = simd_f64(2.0)*y;      /* a scalar broadcasts  */

    for (size_t i = 0; i < simd_f64::size(); i++) {
        printf("x = %4.2f   y = %.8f   z = %.8f\n", x[i], y[i], z[i]);
    }
    return 0;
}
```

```
x = 0.50   y = 1.29930066   z = 2.59860132
x = 1.00   y = 1.21058023   z = 2.42116047
x = 1.50   y = 1.07150437   z = 2.14300874
x = 2.00   y = 1.01363790   z = 2.02727581
```

Available for both `simd_f32` and `simd_f64`:

| | |
|---|---|
| exponential, logarithmic | `exp` `exp2` `expm1` `log` `log2` `log10` `log1p` `pow` |
| trigonometric | `sin` `cos` `tan` `asin` `acos` `atan` `atan2` |
| hyperbolic | `sinh` `cosh` `tanh` `asinh` `acosh` `atanh` |
| special | `erf` `erfc` `tgamma` `lgamma` |
| algebraic | `sqrt` `rsqrt` `cbrt` `hypot` `fma` |
| rounding, sign, misc | `floor` `ceil` `round` `trunc` `rint` `nearbyint` `abs` `fabs` `copysign` `fmin` `fmax` `fmod` `fdim` `remainder` `frexp` `ldexp` `logb` `modf` `scalbn` `nextafter` `blend` |

Across the current test suite the average error is a fraction of an ULP and the
worst case is 10 ULP; run `simd_test` for the figures on your own hardware.
