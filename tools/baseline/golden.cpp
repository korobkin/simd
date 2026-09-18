/* golden.cpp -- deterministic bit-exact reference dump.
 *
 * Purpose: produce the oracle that a ported backend is diffed against.
 * `simd_test` cannot serve this role: it calls srand(time(NULL)) (test.cpp:1229),
 * so its inputs differ every run and it reports only aggregate ULP statistics.
 *
 * This program uses a fixed LCG, re-seeded per function, and prints ONE LINE
 * PER SCALAR ELEMENT rather than per vector. That is deliberate: an 8-lane
 * AVX2 build and a 4-lane NEON build therefore produce files with identical
 * line counts and identical content wherever the math agrees, so a plain
 * `diff` is meaningful across architectures.
 *
 * Build (from the repo root, after a normal build):
 *   g++ -O2 -std=c++20 -DNDEBUG -march=native -mavx2 -Iinclude \
 *       tools/baseline/golden.cpp -Lbuild -lsimd -o build/golden
 *
 * Expected NOT to match bit-exactly after a port: rsqrt (hardware reciprocal
 * estimate; AVX2 _mm256_rsqrt_ps and NEON vrsqrteq_f32 have different,
 * implementation-defined accuracy). It is dumped anyway so the size of the
 * difference is known rather than guessed.
 */

#include <simd.hpp>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>

using namespace simd;

/* Samples per function. Divisible by 8, 4 and 2 so every lane width tiles it
   exactly, which keeps the per-element output identical across widths. */
static const int NSAMP = 512;

struct lcg {
	uint64_t s;
	explicit lcg(uint64_t seed) : s(seed) {}
	double operator()() {              /* uniform in [0,1) */
		s = s * 6364136223846793005ULL + 1442695040888963407ULL;
		return (double)(s >> 11) * (1.0 / 9007199254740992.0);
	}
};

static uint64_t fnv = 0;
static void fnv_reset() { fnv = 1469598103934665603ULL; }
static void fnv_add(const void* p, size_t n) {
	const unsigned char* b = (const unsigned char*)p;
	for (size_t i = 0; i < n; i++) { fnv ^= b[i]; fnv *= 1099511628211ULL; }
}

struct entry { std::string name; uint64_t hash; };
static std::vector<entry> summary;

static uint32_t bits32(float f)  { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint64_t bits64(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

/* ---- single precision, one argument ---- */
template<class F>
static void dump_f32(const char* name, F fn, double a, double b) {
	const int W = (int)simd_f32::size();
	std::vector<float> in(NSAMP);
	std::vector<float> out(NSAMP);
	lcg r(0x9E3779B97F4A7C15ULL);
	for (int i = 0; i < NSAMP; i++) in[i] = (float)(r() * (b - a) + a);

	for (int i = 0; i < NSAMP; i += W) {
		simd_f32 x;
		for (int j = 0; j < W; j++) x[j] = in[i + j];
		simd_f32 y = fn(x);
		for (int j = 0; j < W; j++) out[i + j] = y[j];
	}
	printf("\n## f32 %s  [%g, %g]  n=%d\n", name, a, b, NSAMP);
	fnv_reset();
	for (int i = 0; i < NSAMP; i++) {
		uint32_t xb = bits32(in[i]), yb = bits32(out[i]);
		printf("%4d %08x %08x\n", i, xb, yb);
		fnv_add(&yb, 4);
	}
	summary.push_back({ std::string("f32 ") + name, fnv });
}

/* ---- double precision, one argument ---- */
template<class F>
static void dump_f64(const char* name, F fn, double a, double b) {
	const int W = (int)simd_f64::size();
	std::vector<double> in(NSAMP);
	std::vector<double> out(NSAMP);
	lcg r(0x9E3779B97F4A7C15ULL);
	for (int i = 0; i < NSAMP; i++) in[i] = r() * (b - a) + a;

	for (int i = 0; i < NSAMP; i += W) {
		simd_f64 x;
		for (int j = 0; j < W; j++) x[j] = in[i + j];
		simd_f64 y = fn(x);
		for (int j = 0; j < W; j++) out[i + j] = y[j];
	}
	printf("\n## f64 %s  [%g, %g]  n=%d\n", name, a, b, NSAMP);
	fnv_reset();
	for (int i = 0; i < NSAMP; i++) {
		uint64_t xb = bits64(in[i]), yb = bits64(out[i]);
		printf("%4d %016llx %016llx\n", i,
		       (unsigned long long)xb, (unsigned long long)yb);
		fnv_add(&yb, 8);
	}
	summary.push_back({ std::string("f64 ") + name, fnv });
}

/* ---- double precision, two arguments ---- */
template<class F>
static void dump_f64_2(const char* name, F fn,
                       double a0, double b0, double a1, double b1) {
	const int W = (int)simd_f64::size();
	std::vector<double> in0(NSAMP), in1(NSAMP), out(NSAMP);
	lcg r(0x9E3779B97F4A7C15ULL);
	for (int i = 0; i < NSAMP; i++) {
		in0[i] = r() * (b0 - a0) + a0;
		in1[i] = r() * (b1 - a1) + a1;
	}
	for (int i = 0; i < NSAMP; i += W) {
		simd_f64 x, y;
		for (int j = 0; j < W; j++) { x[j] = in0[i + j]; y[j] = in1[i + j]; }
		simd_f64 z = fn(x, y);
		for (int j = 0; j < W; j++) out[i + j] = z[j];
	}
	printf("\n## f64 %s  [%g,%g]x[%g,%g]  n=%d\n", name, a0, b0, a1, b1, NSAMP);
	fnv_reset();
	for (int i = 0; i < NSAMP; i++) {
		uint64_t yb = bits64(out[i]);
		printf("%4d %016llx %016llx %016llx\n", i,
		       (unsigned long long)bits64(in0[i]),
		       (unsigned long long)bits64(in1[i]),
		       (unsigned long long)yb);
		fnv_add(&yb, 8);
	}
	summary.push_back({ std::string("f64 ") + name, fnv });
}

#define F32(name, lo, hi) dump_f32(#name, [](simd_f32 v) { return name(v); }, lo, hi)
#define F64(name, lo, hi) dump_f64(#name, [](simd_f64 v) { return name(v); }, lo, hi)

int main() {
	printf("# golden reference dump\n");
	printf("# lanes: simd_f32=%zu simd_f64=%zu\n", simd_f32::size(), simd_f64::size());
	printf("# format: index  input(hex)  output(hex)\n");

	/* Domains mirror the TEST1 calls in src/test.cpp so the two agree. */
	F32(asin,  -1.0, 1.0);
	F32(acos,  -1.0, 1.0);
	F32(atan,  -10.0, 10.0);
	F32(acosh, 1.001, 10.0);
	F32(asinh, 0.001, 10.0);
	F32(atanh, 0.001, 0.999);
	F32(exp,   -86.0, 86.0);
	F32(exp2,  -125.0, 125.0);
	F32(expm1, -2.0, 2.0);
	F32(log,   0.36787944, 2.3538527e17);
	F32(log2,  0.00001, 100000.0);
	F32(log1p, 0.049787068, 20.085537);
	F32(erf,   -7.0, 7.0);
	F32(erfc,  -8.9, 8.9);
	F32(tgamma,-33.0, 33.0);
	F32(cosh,  -10.0, 10.0);
	F32(sinh,  -10.0, 10.0);
	F32(tanh,  -10.0, 10.0);
	F32(sin,   -6.2831853, 6.2831853);
	F32(cos,   -6.2831853, 6.2831853);
	F32(tan,   -6.2831853, 6.2831853);
	F32(sqrt,  0.0, 1.0e18);
	F32(cbrt,  0.00025, 4000.0);
	F32(rsqrt, 0.00025, 4000.0);   /* approximation: expect a mismatch */

	F64(asin,  -1.0, 1.0);
	F64(acos,  -0.999999, 0.999999);
	F64(atan,  -10.0, 10.0);
	F64(acosh, 1.001, 10.0);
	F64(asinh, 0.001, 10.0);
	F64(atanh, 0.001, 0.999);
	F64(exp,   -600.0, 600.0);
	F64(exp2,  -1000.0, 1000.0);
	F64(expm1, -2.0, 2.0);
	F64(log,   0.36787944117144233, 2.3538526683702e17);
	F64(log2,  0.0001, 100000.0);
	F64(log1p, 0.049787068367864, 20.085536923187668);
	F64(erf,   -9.0, 9.0);
	F64(erfc,  -25.0, 25.0);
	F64(cosh,  -10.0, 10.0);
	F64(sinh,  -10.0, 10.0);
	F64(tanh,  -10.0, 10.0);
	F64(sin,   -6.283185307179586, 6.283185307179586);
	F64(cos,   -6.283185307179586, 6.283185307179586);
	F64(tan,   -6.283185307179586, 6.283185307179586);
	F64(sqrt,  0.0, 1.0e18);
	F64(cbrt,  0.00025, 4000.0);

	dump_f64_2("pow", [](simd_f64 a, simd_f64 b) { return pow(a, b); },
	           0.1, 10.0, -300.0, 300.0);

	printf("\n\n## SUMMARY (FNV-1a over output bits)\n");
	for (const auto& e : summary) {
		printf("%-16s %016llx\n", e.name.c_str(), (unsigned long long)e.hash);
	}
	return 0;
}
