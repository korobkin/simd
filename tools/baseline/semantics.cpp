/* semantics.cpp -- primitive-level conformance probe.
 *
 * Purpose: pin down the behaviour of the vector primitives that a NEON port
 * has to reproduce, on fixed inputs, with exact bit patterns. This is the
 * part `simd_test` does not cover at all: it exercises the transcendental
 * functions against libm, but never the comparison/mask/bit-twiddling layer
 * underneath them.
 *
 * Output is meant to be READ (and committed), not diffed line-for-line --
 * per-lane sections legitimately differ in length between an 8-lane x86 build
 * and a 4-lane NEON one. For the diffable oracle see golden.cpp.
 *
 * Build (from the repo root, after a normal build):
 *   g++ -O2 -std=c++20 -DNDEBUG -march=native -mavx2 -Iinclude \
 *       tools/baseline/semantics.cpp -Lbuild -lsimd -o build/semantics
 */

#include <simd.hpp>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <limits>

using namespace simd;

static uint32_t bits32(float f)  { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint64_t bits64(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

static void hdr(const char* s) { printf("\n===== %s =====\n", s); }

int main() {
	printf("# simd primitive semantics probe\n");

	hdr("layout");
	printf("simd_i32: size=%zu sizeof=%zu alignof=%zu\n",
	       simd_i32::size(), sizeof(simd_i32), alignof(simd_i32));
	printf("simd_f32: size=%zu sizeof=%zu alignof=%zu\n",
	       simd_f32::size(), sizeof(simd_f32), alignof(simd_f32));
	printf("simd_i64: size=%zu sizeof=%zu alignof=%zu\n",
	       simd_i64::size(), sizeof(simd_i64), alignof(simd_i64));
	printf("simd_f64: size=%zu sizeof=%zu alignof=%zu\n",
	       simd_f64::size(), sizeof(simd_f64), alignof(simd_f64));

	/* THE decisive question for the port: do comparisons yield 1 or -1?
	   Every operator built on top of them (!=, <, >=, fdim, nextafter) and
	   every blend() call site depends on the answer. */
	hdr("f32 comparison result values");
	printf("%-14s %-14s %6s %6s %6s %6s %6s %6s\n",
	       "a", "b", "==", "!=", "<", "<=", ">", ">=");
	{
		const float qnan = std::numeric_limits<float>::quiet_NaN();
		const float inf  = std::numeric_limits<float>::infinity();
		const float pairs[][2] = {
			{1.0f, 2.0f}, {2.0f, 2.0f}, {2.0f, 1.0f}, {-1.0f, 1.0f},
			{0.0f, -0.0f}, {qnan, 1.0f}, {1.0f, qnan}, {inf, inf},
		};
		for (const auto& p : pairs) {
			simd_f32 a(p[0]), b(p[1]);
			printf("%-14.6g %-14.6g %6d %6d %6d %6d %6d %6d\n",
			       p[0], p[1],
			       (a == b)[0], (a != b)[0], (a <  b)[0],
			       (a <= b)[0], (a >  b)[0], (a >= b)[0]);
		}
	}

	hdr("f64 comparison result values");
	printf("%-14s %-14s %6s %6s %6s %6s %6s %6s\n",
	       "a", "b", "==", "!=", "<", "<=", ">", ">=");
	{
		const double qnan = std::numeric_limits<double>::quiet_NaN();
		const double pairs[][2] = {
			{1.0, 2.0}, {2.0, 2.0}, {2.0, 1.0}, {-1.0, 1.0}, {qnan, 1.0},
		};
		for (const auto& p : pairs) {
			simd_f64 a(p[0]), b(p[1]);
			printf("%-14.6g %-14.6g %6lld %6lld %6lld %6lld %6lld %6lld\n",
			       p[0], p[1],
			       (a == b)[0], (a != b)[0], (a <  b)[0],
			       (a <= b)[0], (a >  b)[0], (a >= b)[0]);
		}
	}

	hdr("i32 comparison result values");
	printf("%-8s %-8s %6s %6s %6s %6s %6s %6s\n",
	       "a", "b", "==", "!=", "<", "<=", ">", ">=");
	{
		const int pairs[][2] = { {1,2}, {2,2}, {2,1}, {-1,1}, {0,0}, {-5,-5} };
		for (const auto& p : pairs) {
			simd_i32 a(p[0]), b(p[1]);
			printf("%-8d %-8d %6d %6d %6d %6d %6d %6d\n",
			       p[0], p[1],
			       (a == b)[0], (a != b)[0], (a <  b)[0],
			       (a <= b)[0], (a >  b)[0], (a >= b)[0]);
		}
	}

	/* blend() resolves its mask as `mask = -mask` then a sign-bit select,
	   so which integer values count as "true" is not obvious. */
	hdr("blend mask interpretation  blend(10, 20, m)");
	{
		const int masks[] = { 0, 1, -1, 2, -2, 0x7FFFFFFF, (int)0x80000000 };
		for (int m : masks) {
			simd_f32 r = blend(simd_f32(10.0f), simd_f32(20.0f), simd_i32(m));
			simd_f64 r64 = blend(simd_f64(10.0), simd_f64(20.0), simd_i64(m));
			printf("mask=%-12d f32 -> %5.1f    f64 -> %5.1f\n", m, r[0], r64[0]);
		}
	}

	hdr("derived-from-comparison helpers");
	{
		printf("fdim(5,3)      = %.9g   (expect 2)\n", fdim(simd_f32(5.0f), simd_f32(3.0f))[0]);
		printf("fdim(3,5)      = %.9g   (expect 0)\n", fdim(simd_f32(3.0f), simd_f32(5.0f))[0]);
		simd_f32 na = nextafter(simd_f32(1.0f), simd_f32(2.0f));
		simd_f32 nb = nextafter(simd_f32(1.0f), simd_f32(0.0f));
		printf("nextafter(1,2) = %08x  (libm %08x)\n", bits32(na[0]), bits32(::nextafterf(1.0f, 2.0f)));
		printf("nextafter(1,0) = %08x  (libm %08x)\n", bits32(nb[0]), bits32(::nextafterf(1.0f, 0.0f)));
	}

	/* simd_i32::operator* uses _mm256_mul_epi32, a 32x32->64 even-lane
	   multiply, not a lane-wise int32 multiply. Distinct values per lane so
	   the actual behaviour is visible. */
	hdr("i32 arithmetic, per lane");
	{
		simd_i32 a, b;
		for (size_t i = 0; i < simd_i32::size(); i++) { a[i] = (int)i + 1; b[i] = 10; }
		simd_i32 m = a * b, s = a + b, d = a - b;
		simd_i32 nt = ~a, no = !a;
		for (size_t i = 0; i < simd_i32::size(); i++) {
			printf("lane %zu: a=%d b=%d  a*b=%-12d a+b=%-6d a-b=%-6d ~a=%-12d !a=%d\n",
			       i, a[i], b[i], m[i], s[i], d[i], nt[i], no[i]);
		}
	}

	hdr("i32 shifts");
	{
		simd_i32 a(-8), b(3);
		printf("(-8) >> 3 (vector count) = %d\n", (a >> b)[0]);
		printf("(-8) >> 3 (scalar count) = %d\n", (a >> 3)[0]);
		printf("(-8) << 1 (scalar count) = %d\n", (a << 1)[0]);
		printf("note: _mm256_srlv/srli_epi32 are LOGICAL shifts on a signed type\n");
	}

	hdr("rounding, fixed inputs (hex bits)");
	{
		const float xs[] = { -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 3.7f, -3.7f };
		printf("%-8s %-10s %-10s %-10s %-10s %-10s %-8s\n",
		       "x", "round", "rint", "trunc", "floor", "ceil", "lround");
		for (float x : xs) {
			simd_f32 v(x);
			printf("%-8.2f %08x   %08x   %08x   %08x   %08x   %-8d\n",
			       x, bits32(round(v)[0]), bits32(rint(v)[0]), bits32(trunc(v)[0]),
			       bits32(floor(v)[0]), bits32(ceil(v)[0]), lround(v)[0]);
		}
	}

	hdr("exponent manipulation (hex bits)");
	{
		const float xs[] = { 1.0f, 0.5f, 2.0f, 3.0f, 0.1f, 1e-30f, 1e30f, -7.25f };
		printf("%-12s %-10s %-6s %-10s %-10s %-6s\n",
		       "x", "frexp.m", "frexp.e", "ldexp(x,3)", "scalbn(x,3)", "ilogb");
		for (float x : xs) {
			simd_f32 v(x);
			simd_i32 e;
			simd_f32 m = frexp(v, &e);
			printf("%-12.6g %08x   %-6d %08x   %08x   %-6d\n",
			       x, bits32(m[0]), e[0],
			       bits32(ldexp(v, simd_i32(3))[0]),
			       bits32(scalbn(v, simd_i32(3))[0]),
			       ilogb(v)[0]);
		}
	}

	hdr("copysign / fmin / fmax / abs (hex bits)");
	{
		const float pairs[][2] = { {3.0f,-1.0f}, {-3.0f,1.0f}, {0.0f,-1.0f}, {-0.0f,1.0f} };
		for (const auto& p : pairs) {
			simd_f32 a(p[0]), b(p[1]);
			printf("a=%-8.2f b=%-8.2f copysign=%08x fmin=%08x fmax=%08x fabs=%08x\n",
			       p[0], p[1], bits32(copysign(a, b)[0]), bits32(fmin(a, b)[0]),
			       bits32(fmax(a, b)[0]), bits32(fabs(a)[0]));
		}
	}

	/* permute and gather are the two primitives with no direct NEON
	   equivalent, and the generated erf/asin/tgamma lean on them hard. */
	hdr("permute");
	{
		simd_f32 v;
		simd_i32 idx;
		for (size_t i = 0; i < simd_f32::size(); i++) {
			v[i] = (float)(100 + i);
			idx[i] = (int)(simd_f32::size() - 1 - i);   /* reverse */
		}
		simd_f32 r = v.permute(idx);
		printf("reverse permute:");
		for (size_t i = 0; i < simd_f32::size(); i++) printf(" %.0f", r[i]);
		printf("\n");
		for (size_t i = 0; i < simd_f32::size(); i++) idx[i] = 0;   /* broadcast lane 0 */
		r = v.permute(idx);
		printf("broadcast lane0:");
		for (size_t i = 0; i < simd_f32::size(); i++) printf(" %.0f", r[i]);
		printf("\n");
	}

	hdr("gather");
	{
		float table[16];
		for (int i = 0; i < 16; i++) table[i] = (float)(i * i);
		simd_i32 idx;
		for (size_t i = 0; i < simd_f32::size(); i++) idx[i] = (int)(2 * i);
		simd_f32 g;
		g.gather(table, idx);
		printf("gather stride2:");
		for (size_t i = 0; i < simd_f32::size(); i++) printf(" %.0f", g[i]);
		printf("\n");
	}

	hdr("reduce_sum");
	{
		simd_f32 a;
		simd_f64 b;
		for (size_t i = 0; i < simd_f32::size(); i++) a[i] = (float)(i + 1);
		for (size_t i = 0; i < simd_f64::size(); i++) b[i] = (double)(i + 1);
		printf("reduce_sum(f32 1..%zu) = %.9g\n", simd_f32::size(), reduce_sum(a));
		printf("reduce_sum(f64 1..%zu) = %.17g\n", simd_f64::size(), reduce_sum(b));
	}

	hdr("conversions");
	{
		const float xs[] = { 1.9f, -1.9f, 2.5f, -2.5f, 0.4f };
		for (float x : xs) {
			simd_i32 i(simd_f32(x));           /* float -> int  (truncating?) */
			simd_f32 f(simd_i32((int)x));      /* int   -> float */
			printf("x=%-6.2f  simd_i32(simd_f32(x))=%-4d  simd_f32(simd_i32(%d))=%.1f\n",
			       x, i[0], (int)x, f[0]);
		}
	}

	hdr("mask / pad");
	{
		simd_f32 m = simd_f32::mask(2);
		printf("simd_f32::mask(2):");
		for (size_t i = 0; i < simd_f32::size(); i++) printf(" %08x", bits32(m[i]));
		printf("\n");
		simd_i32 mi = simd_i32::mask(2);
		printf("simd_i32::mask(2):");
		for (size_t i = 0; i < simd_i32::size(); i++) printf(" %d", mi[i]);
		printf("\n");
	}

	/* two_product's correctness depends on an exact FMA and on the compiler
	   NOT contracting the neighbouring subtract. Worth a direct check. */
	hdr("fma exactness (double-double building block)");
	{
		simd_f32 a(1.0f + 0x1p-12f), b(1.0f - 0x1p-12f);
		simd_f32 p = a * b;
		simd_f32 err = fma(a, b, -p);
		printf("a*b       = %08x\n", bits32(p[0]));
		printf("fma(a,b,-p)= %08x  (exact residual; must be nonzero here)\n", bits32(err[0]));
		double ad = 1.0 + 0x1p-27, bd = 1.0 - 0x1p-27;
		simd_f64 a2(ad), b2(bd);
		simd_f64 p2 = a2 * b2;
		simd_f64 e2 = fma(a2, b2, -p2);
		printf("f64 a*b   = %016llx\n", (unsigned long long)bits64(p2[0]));
		printf("f64 resid = %016llx\n", (unsigned long long)bits64(e2[0]));
	}

	printf("\n# end\n");
	return 0;
}
