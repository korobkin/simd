#pragma once

/* AArch64 NEON backend: the same primitive interface as simd_backend_x86.hpp,
 * implemented on 128-bit NEON. See PORT-AARCH64.md.
 *
 * The lane counts are HALF the x86 ones -- 4 floats and 2 doubles, against 8
 * and 4. That is the whole difference between this file and the SIMDe backend,
 * which emulates the AVX2 widths and so leaves src/codegen.cpp and
 * src/test.cpp untouched. Everything that assumed 8 and 4 has to follow.
 *
 * Semantics are matched to the x86 backend deliberately, not approximately:
 * baseline/x86/golden.txt and baseline/x86/semantics.txt are diffed against
 * this, so anything that differs must differ for a reason that is written
 * down. The places where the obvious NEON instruction is NOT the right one are
 * commented individually.
 */

#include <arm_neon.h>
#include <cstdint>

#define SIMD_BACKEND_NAME "NEON"
#define SIMD_BACKEND_NEON 1
/* NEON vector types are GCC vector types and support v[i], so the float
   classes keep their bare member. Defining SIMD_BACKEND_LANE_UNION here would
   cost what it cost on x86 -- see TODO item 9. */

namespace simd {
namespace backend {

using f32v = float32x4_t;
using f64v = float64x2_t;
using i32v = int32x4_t;
using i64v = int64x2_t;

static constexpr int f32_lanes = 4;
static constexpr int f64_lanes = 2;
static constexpr int i32_lanes = 4;
static constexpr int i64_lanes = 2;

static constexpr int vec_align = alignof(float32x4_t);

/* ---------------------------------------------------------------- float32 */

static inline f32v f32_set1(float a)             { return vdupq_n_f32(a); }
static inline f32v f32_add(f32v a, f32v b)       { return vaddq_f32(a, b); }
static inline f32v f32_sub(f32v a, f32v b)       { return vsubq_f32(a, b); }
static inline f32v f32_mul(f32v a, f32v b)       { return vmulq_f32(a, b); }
static inline f32v f32_div(f32v a, f32v b)       { return vdivq_f32(a, b); }
static inline f32v f32_sqrt(f32v a)              { return vsqrtq_f32(a); }

/* NOT vminq_f32/vmaxq_f32. Those implement IEEE minNum/maxNum, which return
   the non-NaN operand and treat -0 and +0 as equal. _mm256_min_ps is a plain
   "a < b ? a : b", so it returns b when either operand is NaN and when they
   compare equal. Written as the select to match exactly. */
static inline f32v f32_min(f32v a, f32v b) {
	return vbslq_f32(vcltq_f32(a, b), a, b);
}
static inline f32v f32_max(f32v a, f32v b) {
	return vbslq_f32(vcgtq_f32(a, b), a, b);
}

/* Reciprocal square-root estimate. Neither this nor _mm256_rsqrt_ps is
   specified exactly, so this is the one primitive whose result legitimately
   differs between backends. */
static inline f32v f32_rsqrt(f32v a)             { return vrsqrteq_f32(a); }

static inline f32v f32_floor(f32v a)             { return vrndmq_f32(a); }
static inline f32v f32_ceil(f32v a)              { return vrndpq_f32(a); }

/* vrndnq is ties-to-EVEN and matches _MM_FROUND_TO_NEAREST_INT. vrndaq is
   ties-away-from-zero, which is what C's round() means and what the name of
   the caller suggests -- using it would silently change sin, cos, exp and
   tgamma through their argument reduction. See TODO item 3. */
static inline f32v f32_round_nearest(f32v a)     { return vrndnq_f32(a); }
static inline f32v f32_round_zero(f32v a)        { return vrndq_f32(a); }

/* vfmaq_f32(c, a, b) computes c + a*b, so the addend comes first. */
static inline f32v f32_fmadd(f32v a, f32v b, f32v c) { return vfmaq_f32(c, a, b); }
static inline f32v f32_fmsub(f32v a, f32v b, f32v c) {
	return vfmaq_f32(vnegq_f32(c), a, b);
}

/* All ordered: false whenever either operand is NaN, matching _CMP_*_OS. */
static inline i32v f32_cmp_eq(f32v a, f32v b) { return vreinterpretq_s32_u32(vceqq_f32(a, b)); }
static inline i32v f32_cmp_gt(f32v a, f32v b) { return vreinterpretq_s32_u32(vcgtq_f32(a, b)); }
static inline i32v f32_cmp_ge(f32v a, f32v b) { return vreinterpretq_s32_u32(vcgeq_f32(a, b)); }
static inline i32v f32_cmp_lt(f32v a, f32v b) { return vreinterpretq_s32_u32(vcltq_f32(a, b)); }
static inline i32v f32_cmp_le(f32v a, f32v b) { return vreinterpretq_s32_u32(vcleq_f32(a, b)); }

/* _CMP_NEQ_OS is ordered, so a NaN operand gives FALSE. The negation of vceqq
   alone would give true there, so it is masked with "both operands ordered". */
static inline i32v f32_cmp_neq(f32v a, f32v b) {
	uint32x4_t ord = vandq_u32(vceqq_f32(a, a), vceqq_f32(b, b));
	return vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(vceqq_f32(a, b)), ord));
}

/* Selects b where the mask lane's SIGN BIT is set, as _mm256_blendv_ps does --
   not where the lane is nonzero. Broadcast the sign bit with an arithmetic
   shift so the rest of the lane is ignored. */
static inline f32v f32_blendv(f32v a, f32v b, f32v mask) {
	uint32x4_t m = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(mask), 31));
	return vbslq_f32(m, b, a);
}

/* NEON has no gather; the generated erf/tgamma/asin lean on this. */
static inline f32v f32_gather(const float* p, i32v idx) {
	f32v r;
	for (int i = 0; i < f32_lanes; i++) {
		r[i] = p[idx[i]];
	}
	return r;
}

/* A byte-table lookup: each lane's index selects 4 consecutive bytes. The
   index is masked to the lane count, as _mm256_permutevar8x32_ps masks to its
   low 3 bits. */
static inline uint8x16_t permute_byte_index(i32v idx, int lanes) {
	int32x4_t l = vandq_s32(idx, vdupq_n_s32(lanes - 1));
	int32x4_t b = vmulq_n_s32(l, 0x04040404);
	return vreinterpretq_u8_s32(vaddq_s32(b, vdupq_n_s32(0x03020100)));
}
static inline f32v f32_permute(f32v a, i32v idx) {
	return vreinterpretq_f32_u8(
		vqtbl1q_u8(vreinterpretq_u8_f32(a), permute_byte_index(idx, f32_lanes)));
}

static inline i32v f32_as_i32(f32v a)            { return vreinterpretq_s32_f32(a); }
static inline f32v i32_as_f32(i32v a)            { return vreinterpretq_f32_s32(a); }

static inline f32v f32_from_i32(i32v a)          { return vcvtq_f32_s32(a); }
/* vcvtq_s32_f32 truncates, so the rounding is done first, exactly as the x86
   backend composes _mm256_cvtps_epi32 with _mm256_round_ps. */
static inline i32v f32_to_i32_nearest(f32v a)    { return vcvtq_s32_f32(vrndnq_f32(a)); }
static inline i32v f32_to_i32_zero(f32v a)       { return vcvtq_s32_f32(a); }

/* ------------------------------------------------------------------ int32 */

static inline i32v i32_set1(int a)               { return vdupq_n_s32(a); }
static inline i32v i32_add(i32v a, i32v b)       { return vaddq_s32(a, b); }
static inline i32v i32_sub(i32v a, i32v b)       { return vsubq_s32(a, b); }

/* Reproduces _mm256_mul_epi32, which is NOT a lane-wise multiply: it takes the
   EVEN lanes, multiplies 32x32 into 64 bits, and leaves the odd lanes holding
   the high halves. The result looks wrong because it is -- see
   baseline/x86/semantics.txt, where lanes 1 and 3 come back zero. Reproduced
   rather than corrected so that the two backends agree and the baseline stays
   a clean oracle; fixing it is a separate change to both. */
static inline i32v i32_mul(i32v a, i32v b) {
	int32x2_t ae = vuzp1_s32(vget_low_s32(a), vget_high_s32(a));
	int32x2_t be = vuzp1_s32(vget_low_s32(b), vget_high_s32(b));
	return vreinterpretq_s32_s64(vmull_s32(ae, be));
}

static inline i32v i32_and(i32v a, i32v b)       { return vandq_s32(a, b); }
static inline i32v i32_or(i32v a, i32v b)        { return vorrq_s32(a, b); }
static inline i32v i32_xor(i32v a, i32v b)       { return veorq_s32(a, b); }
/* _mm256_andnot_si256(a, b) is (~a) & b; vbicq_s32(x, y) is x & ~y. */
static inline i32v i32_andnot(i32v a, i32v b)    { return vbicq_s32(b, a); }

/* LOGICAL right shifts, matching _mm256_srlv/srli_epi32 on a signed type:
   (-8) >> 3 is 536870911, and simd.hpp's bit manipulation relies on it. NEON
   shifts right by a NEGATIVE left-shift count. */
static inline i32v i32_srlv(i32v a, i32v n) {
	return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(a), vnegq_s32(n)));
}
static inline i32v i32_sllv(i32v a, i32v n)      { return vshlq_s32(a, n); }
static inline i32v i32_srli(i32v a, int n)       { return i32_srlv(a, vdupq_n_s32(n)); }
static inline i32v i32_slli(i32v a, int n)       { return vshlq_s32(a, vdupq_n_s32(n)); }

static inline i32v i32_cmpeq(i32v a, i32v b)     { return vreinterpretq_s32_u32(vceqq_s32(a, b)); }
static inline i32v i32_cmpgt(i32v a, i32v b)     { return vreinterpretq_s32_u32(vcgtq_s32(a, b)); }
static inline i32v i32_min(i32v a, i32v b)       { return vminq_s32(a, b); }
static inline i32v i32_max(i32v a, i32v b)       { return vmaxq_s32(a, b); }

static inline i32v i32_gather(const int* p, i32v idx) {
	i32v r;
	for (int i = 0; i < i32_lanes; i++) {
		r[i] = p[idx[i]];
	}
	return r;
}
static inline i32v i32_permute(i32v a, i32v idx) {
	return vreinterpretq_s32_u8(
		vqtbl1q_u8(vreinterpretq_u8_s32(a), permute_byte_index(idx, i32_lanes)));
}

/* ---------------------------------------------------------------- float64 */

static inline f64v f64_set1(double a)            { return vdupq_n_f64(a); }
static inline f64v f64_add(f64v a, f64v b)       { return vaddq_f64(a, b); }
static inline f64v f64_sub(f64v a, f64v b)       { return vsubq_f64(a, b); }
static inline f64v f64_mul(f64v a, f64v b)       { return vmulq_f64(a, b); }
static inline f64v f64_div(f64v a, f64v b)       { return vdivq_f64(a, b); }
static inline f64v f64_sqrt(f64v a)              { return vsqrtq_f64(a); }
static inline f64v f64_min(f64v a, f64v b)       { return vbslq_f64(vcltq_f64(a, b), a, b); }
static inline f64v f64_max(f64v a, f64v b)       { return vbslq_f64(vcgtq_f64(a, b), a, b); }
static inline f64v f64_floor(f64v a)             { return vrndmq_f64(a); }
static inline f64v f64_ceil(f64v a)              { return vrndpq_f64(a); }
static inline f64v f64_round_nearest(f64v a)     { return vrndnq_f64(a); }
static inline f64v f64_round_zero(f64v a)        { return vrndq_f64(a); }

static inline f64v f64_fmadd(f64v a, f64v b, f64v c) { return vfmaq_f64(c, a, b); }
static inline f64v f64_fmsub(f64v a, f64v b, f64v c) {
	return vfmaq_f64(vnegq_f64(c), a, b);
}

static inline i64v f64_cmp_eq(f64v a, f64v b) { return vreinterpretq_s64_u64(vceqq_f64(a, b)); }
static inline i64v f64_cmp_gt(f64v a, f64v b) { return vreinterpretq_s64_u64(vcgtq_f64(a, b)); }
static inline i64v f64_cmp_ge(f64v a, f64v b) { return vreinterpretq_s64_u64(vcgeq_f64(a, b)); }
static inline i64v f64_cmp_lt(f64v a, f64v b) { return vreinterpretq_s64_u64(vcltq_f64(a, b)); }
static inline i64v f64_cmp_le(f64v a, f64v b) { return vreinterpretq_s64_u64(vcleq_f64(a, b)); }
static inline i64v f64_cmp_neq(f64v a, f64v b) {
	uint64x2_t ord = vandq_u64(vceqq_f64(a, a), vceqq_f64(b, b));
	uint64x2_t ne = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(a, b))));
	return vreinterpretq_s64_u64(vandq_u64(ne, ord));
}

static inline f64v f64_blendv(f64v a, f64v b, f64v mask) {
	uint64x2_t m = vreinterpretq_u64_s64(vshrq_n_s64(vreinterpretq_s64_f64(mask), 63));
	return vbslq_f64(m, b, a);
}

static inline i64v f64_as_i64(f64v a)            { return vreinterpretq_s64_f64(a); }
static inline f64v i64_as_f64(i64v a)            { return vreinterpretq_f64_s64(a); }

static inline f64v f64_gather(const double* p, i64v idx) {
	f64v r;
	for (int i = 0; i < f64_lanes; i++) {
		r[i] = p[idx[i]];
	}
	return r;
}

/* One bit per lane, from each lane's sign bit. The generated asin(simd_f64)
   indexes a table with this, so the table has 2^lanes rows -- 4 here against
   16 on x86. */
static inline int f64_movemask(f64v a) {
	uint64x2_t s = vshrq_n_u64(vreinterpretq_u64_f64(a), 63);
	return (int) (vgetq_lane_u64(s, 0) | (vgetq_lane_u64(s, 1) << 1));
}

static inline i64v f64_to_i64_nearest(f64v a)    { return vcvtq_s64_f64(vrndnq_f64(a)); }

/* ------------------------------------------------------------------ int64 */

static inline i64v i64_set1(long long a)         { return vdupq_n_s64((int64_t) a); }
static inline i64v i64_add(i64v a, i64v b)       { return vaddq_s64(a, b); }
static inline i64v i64_sub(i64v a, i64v b)       { return vsubq_s64(a, b); }
static inline i64v i64_and(i64v a, i64v b)       { return vandq_s64(a, b); }
static inline i64v i64_or(i64v a, i64v b)        { return vorrq_s64(a, b); }
static inline i64v i64_xor(i64v a, i64v b)       { return veorq_s64(a, b); }
static inline i64v i64_andnot(i64v a, i64v b)    { return vbicq_s64(b, a); }
static inline i64v i64_srlv(i64v a, i64v n) {
	return vreinterpretq_s64_u64(vshlq_u64(vreinterpretq_u64_s64(a), vnegq_s64(n)));
}
static inline i64v i64_sllv(i64v a, i64v n)      { return vshlq_s64(a, n); }
static inline i64v i64_srli(i64v a, int n)       { return i64_srlv(a, vdupq_n_s64(n)); }
static inline i64v i64_slli(i64v a, int n)       { return vshlq_s64(a, vdupq_n_s64(n)); }
static inline i64v i64_cmpeq(i64v a, i64v b)     { return vreinterpretq_s64_u64(vceqq_s64(a, b)); }
static inline i64v i64_cmpgt(i64v a, i64v b)     { return vreinterpretq_s64_u64(vcgtq_s64(a, b)); }
static inline i64v i64_gather(const long long* p, i64v idx) {
	i64v r;
	for (int i = 0; i < i64_lanes; i++) {
		r[i] = p[idx[i]];
	}
	return r;
}

}
}
