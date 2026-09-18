#pragma once

/* x86_64 backend: the vector primitives simd.hpp is written against.
 *
 * Serves two configurations. On x86_64 these are AVX2 intrinsics from
 * <immintrin.h>. Everywhere else they are SIMDe's reimplementation of the same
 * intrinsics over the host's native vectors -- on AArch64, NEON. Both keep
 * 8 float / 4 double lanes, which is what lets src/codegen.cpp, the generated
 * math.cpp and the buffer sizing in src/test.cpp stay as they are.
 *
 * A native NEON backend, where the lane counts actually change, is
 * simd_backend_neon.hpp. See PORT-AARCH64.md.
 *
 * Naming is <type>_<operation>, not overloading, because on x86 the 32- and
 * 64-bit integer vectors are the same type (__m256i) and overloads on them
 * would be ambiguous.
 */

#include <cstdint>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define SIMD_BACKEND_NAME "AVX2"
#else
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#define SIMD_BACKEND_NAME "SIMDe/AVX2"
#define SIMD_BACKEND_SIMDE 1
/* simde__m256 is a union type and cannot be subscripted, so the float classes
   must reach their lanes through a union of their own. The native backend must
   NOT do that: adding an array member alongside the vector stops GCC keeping
   the object in a register, and measured 2.5x slower on x86. */
#define SIMD_BACKEND_LANE_UNION 1
#endif

namespace simd {
namespace backend {

using f32v = __m256;
using f64v = __m256d;
using i32v = __m256i;
using i64v = __m256i;

static constexpr int f32_lanes = 8;
static constexpr int f64_lanes = 4;
static constexpr int i32_lanes = 8;
static constexpr int i64_lanes = 4;

/* Object alignment the classes are expected to have. */
static constexpr int vec_align = alignof(__m256);

/* ---------------------------------------------------------------- float32 */

static inline f32v f32_set1(float a)             { return _mm256_set1_ps(a); }
static inline f32v f32_add(f32v a, f32v b)       { return _mm256_add_ps(a, b); }
static inline f32v f32_sub(f32v a, f32v b)       { return _mm256_sub_ps(a, b); }
static inline f32v f32_mul(f32v a, f32v b)       { return _mm256_mul_ps(a, b); }
static inline f32v f32_div(f32v a, f32v b)       { return _mm256_div_ps(a, b); }
static inline f32v f32_sqrt(f32v a)              { return _mm256_sqrt_ps(a); }
static inline f32v f32_min(f32v a, f32v b)       { return _mm256_min_ps(a, b); }
static inline f32v f32_max(f32v a, f32v b)       { return _mm256_max_ps(a, b); }

/* Reciprocal square-root ESTIMATE. Accurate to about 12 bits, and the exact
   result is implementation-defined -- it is the one operation that does not
   agree bit-for-bit between backends. */
static inline f32v f32_rsqrt(f32v a)             { return _mm256_rsqrt_ps(a); }

static inline f32v f32_floor(f32v a)             { return _mm256_floor_ps(a); }
static inline f32v f32_ceil(f32v a)              { return _mm256_ceil_ps(a); }

/* Ties to EVEN, matching _MM_FROUND_TO_NEAREST_INT. Not C's round(), which is
   ties-away-from-zero; see TODO item 3 before changing this. */
static inline f32v f32_round_nearest(f32v a) {
	return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT);
}
static inline f32v f32_round_zero(f32v a) {
	return _mm256_round_ps(a, _MM_FROUND_TO_ZERO);
}

/* Fused: one rounding. The double-double arithmetic in simd_f32_2 depends on
   it -- two_product recovers the product's exact error with fma(a,b,-a*b). */
static inline f32v f32_fmadd(f32v a, f32v b, f32v c) {
#if defined(SIMD_BACKEND_SIMDE)
	/* simde_mm256_fmadd_ps already delegates to the fused 128-bit path. */
#endif
	return _mm256_fmadd_ps(a, b, c);
}
static inline f32v f32_fmsub(f32v a, f32v b, f32v c) {
#if defined(SIMD_BACKEND_SIMDE)
	/* simde_mm256_fmsub_ps is NOT fused (TODO item 5): it expands to a
	   separate multiply and subtract. Build it from the fused add against a
	   sign-flipped addend; flipping the sign bit rather than subtracting from
	   zero keeps -0 and NaN payloads exact. */
	return _mm256_fmadd_ps(a, b, _mm256_xor_ps(c, _mm256_set1_ps(-0.0f)));
#else
	return _mm256_fmsub_ps(a, b, c);
#endif
}

/* Comparisons yield all-ones per lane. simd.hpp negates that to 1, which is
   the convention the whole library and all generated code assume. */
static inline i32v f32_cmp_eq(f32v a, f32v b)  { return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_EQ_OS)); }
static inline i32v f32_cmp_neq(f32v a, f32v b) { return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_NEQ_OS)); }
static inline i32v f32_cmp_gt(f32v a, f32v b)  { return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_GT_OS)); }
static inline i32v f32_cmp_ge(f32v a, f32v b)  { return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_GE_OS)); }
static inline i32v f32_cmp_lt(f32v a, f32v b)  { return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LT_OS)); }
static inline i32v f32_cmp_le(f32v a, f32v b)  { return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LE_OS)); }

/* Selects b where the SIGN BIT of the mask lane is set. simd.hpp negates a
   0/1 mask before calling this, so 1 selects b. */
static inline f32v f32_blendv(f32v a, f32v b, f32v mask) { return _mm256_blendv_ps(a, b, mask); }

static inline f32v f32_gather(const float* p, i32v idx)  { return _mm256_i32gather_ps(p, idx, sizeof(float)); }
static inline f32v f32_permute(f32v a, i32v idx)         { return _mm256_permutevar8x32_ps(a, idx); }

static inline f32v f32_from_i32(i32v a)          { return _mm256_cvtepi32_ps(a); }
static inline i32v f32_to_i32_nearest(f32v a)    { return _mm256_cvtps_epi32(f32_round_nearest(a)); }
static inline i32v f32_to_i32_zero(f32v a)       { return _mm256_cvtps_epi32(f32_round_zero(a)); }

/* ------------------------------------------------------------------ int32 */

static inline i32v i32_set1(int a)               { return _mm256_set1_epi32(a); }
static inline i32v i32_add(i32v a, i32v b)       { return _mm256_add_epi32(a, b); }
static inline i32v i32_sub(i32v a, i32v b)       { return _mm256_sub_epi32(a, b); }

/* NOTE: _mm256_mul_epi32 is a 32x32->64 multiply of the EVEN lanes only, not a
   lane-wise int32 multiply -- odd lanes come back zero (see
   baseline/x86/semantics.txt). Preserved here so the backends agree; nothing
   in the math paths uses it. */
static inline i32v i32_mul(i32v a, i32v b)       { return _mm256_mul_epi32(a, b); }

static inline i32v i32_and(i32v a, i32v b)       { return _mm256_and_si256(a, b); }
static inline i32v i32_or(i32v a, i32v b)        { return _mm256_or_si256(a, b); }
static inline i32v i32_xor(i32v a, i32v b)       { return _mm256_xor_si256(a, b); }
static inline i32v i32_andnot(i32v a, i32v b)    { return _mm256_andnot_si256(a, b); }

/* LOGICAL shifts, despite the signed element type: (-8) >> 3 is 536870911.
   Several bit-manipulation routines in simd.hpp rely on that. */
static inline i32v i32_srlv(i32v a, i32v n)      { return _mm256_srlv_epi32(a, n); }
static inline i32v i32_sllv(i32v a, i32v n)      { return _mm256_sllv_epi32(a, n); }
static inline i32v i32_srli(i32v a, int n)       { return _mm256_srli_epi32(a, n); }
static inline i32v i32_slli(i32v a, int n)       { return _mm256_slli_epi32(a, n); }

static inline i32v i32_cmpeq(i32v a, i32v b)     { return _mm256_cmpeq_epi32(a, b); }
static inline i32v i32_cmpgt(i32v a, i32v b)     { return _mm256_cmpgt_epi32(a, b); }
static inline i32v i32_min(i32v a, i32v b)       { return _mm256_min_epi32(a, b); }
static inline i32v i32_max(i32v a, i32v b)       { return _mm256_max_epi32(a, b); }

static inline i32v i32_gather(const int* p, i32v idx) { return _mm256_i32gather_epi32(p, idx, sizeof(int)); }
static inline i32v i32_permute(i32v a, i32v idx)      { return _mm256_permutevar8x32_epi32(a, idx); }

/* ---------------------------------------------------------------- float64 */

static inline f64v f64_set1(double a)            { return _mm256_set1_pd(a); }
static inline f64v f64_add(f64v a, f64v b)       { return _mm256_add_pd(a, b); }
static inline f64v f64_sub(f64v a, f64v b)       { return _mm256_sub_pd(a, b); }
static inline f64v f64_mul(f64v a, f64v b)       { return _mm256_mul_pd(a, b); }
static inline f64v f64_div(f64v a, f64v b)       { return _mm256_div_pd(a, b); }
static inline f64v f64_sqrt(f64v a)              { return _mm256_sqrt_pd(a); }
static inline f64v f64_min(f64v a, f64v b)       { return _mm256_min_pd(a, b); }
static inline f64v f64_max(f64v a, f64v b)       { return _mm256_max_pd(a, b); }
static inline f64v f64_floor(f64v a)             { return _mm256_floor_pd(a); }
static inline f64v f64_ceil(f64v a)              { return _mm256_ceil_pd(a); }

static inline f64v f64_round_nearest(f64v a) {
	return _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT);
}
static inline f64v f64_round_zero(f64v a) {
	return _mm256_round_pd(a, _MM_FROUND_TO_ZERO);
}

static inline f64v f64_fmadd(f64v a, f64v b, f64v c) {
#if defined(SIMD_BACKEND_SIMDE)
	/* simde_mm256_fmadd_pd is NOT fused (TODO item 5) and, unlike the ps
	   form, has no 128-bit delegation. Build it from two fused halves. */
	return _mm256_set_m128d(
		_mm_fmadd_pd(_mm256_extractf128_pd(a, 1), _mm256_extractf128_pd(b, 1),
		             _mm256_extractf128_pd(c, 1)),
		_mm_fmadd_pd(_mm256_extractf128_pd(a, 0), _mm256_extractf128_pd(b, 0),
		             _mm256_extractf128_pd(c, 0)));
#else
	return _mm256_fmadd_pd(a, b, c);
#endif
}
static inline f64v f64_fmsub(f64v a, f64v b, f64v c) {
#if defined(SIMD_BACKEND_SIMDE)
	return f64_fmadd(a, b, _mm256_xor_pd(c, _mm256_set1_pd(-0.0)));
#else
	return _mm256_fmsub_pd(a, b, c);
#endif
}

static inline i64v f64_cmp_eq(f64v a, f64v b)  { return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_EQ_OS)); }
static inline i64v f64_cmp_neq(f64v a, f64v b) { return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_NEQ_OS)); }
static inline i64v f64_cmp_gt(f64v a, f64v b)  { return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GT_OS)); }
static inline i64v f64_cmp_ge(f64v a, f64v b)  { return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_GE_OS)); }
static inline i64v f64_cmp_lt(f64v a, f64v b)  { return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LT_OS)); }
static inline i64v f64_cmp_le(f64v a, f64v b)  { return _mm256_castpd_si256(_mm256_cmp_pd(a, b, _CMP_LE_OS)); }

static inline f64v f64_blendv(f64v a, f64v b, f64v mask) { return _mm256_blendv_pd(a, b, mask); }
static inline f64v f64_gather(const double* p, i64v idx) { return _mm256_i64gather_pd(p, idx, sizeof(double)); }

/* One bit per lane, taken from each lane's sign bit. The generated
   asin(simd_f64) uses this to index a table with one row per mask value, so
   this primitive is part of the contract with src/codegen.cpp. */
static inline int f64_movemask(f64v a) { return _mm256_movemask_pd(a); }

/* Rounds to nearest. _mm256_cvtpd_epi64 is AVX512DQ+VL rather than AVX2, and
   SIMDe does not provide it; its only caller passes an argument already
   rounded to an exact integer, so a truncating conversion agrees. */
static inline i64v f64_to_i64_nearest(f64v a) {
#if defined(SIMD_BACKEND_SIMDE) || !defined(__AVX512DQ__) || !defined(__AVX512VL__)
	double t[f64_lanes];
	int64_t u[f64_lanes];
	_mm256_storeu_pd(t, f64_round_nearest(a));
	for (int i = 0; i < f64_lanes; i++) {
		u[i] = (int64_t) t[i];
	}
	return _mm256_loadu_si256((const __m256i*) u);
#else
	return _mm256_cvtpd_epi64(f64_round_nearest(a));
#endif
}

/* ------------------------------------------------------------------ int64 */

static inline i64v i64_set1(long long a)         { return _mm256_set_epi64x(a, a, a, a); }
static inline i64v i64_add(i64v a, i64v b)       { return _mm256_add_epi64(a, b); }
static inline i64v i64_sub(i64v a, i64v b)       { return _mm256_sub_epi64(a, b); }
static inline i64v i64_and(i64v a, i64v b)       { return _mm256_and_si256(a, b); }
static inline i64v i64_or(i64v a, i64v b)        { return _mm256_or_si256(a, b); }
static inline i64v i64_xor(i64v a, i64v b)       { return _mm256_xor_si256(a, b); }
static inline i64v i64_andnot(i64v a, i64v b)    { return _mm256_andnot_si256(a, b); }
static inline i64v i64_srlv(i64v a, i64v n)      { return _mm256_srlv_epi64(a, n); }
static inline i64v i64_sllv(i64v a, i64v n)      { return _mm256_sllv_epi64(a, n); }
static inline i64v i64_srli(i64v a, int n)       { return _mm256_srli_epi64(a, n); }
static inline i64v i64_slli(i64v a, int n)       { return _mm256_slli_epi64(a, n); }
static inline i64v i64_cmpeq(i64v a, i64v b)     { return _mm256_cmpeq_epi64(a, b); }
static inline i64v i64_cmpgt(i64v a, i64v b)     { return _mm256_cmpgt_epi64(a, b); }
static inline i64v i64_gather(const long long* p, i64v idx) {
	return _mm256_i64gather_epi64((const long long int*) p, idx, sizeof(long long));
}

}
}
