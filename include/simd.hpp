#pragma once

#include <immintrin.h>
#include <limits>
#include <mutex>
#include <cfenv>
#include <cmath>
#include <initializer_list>
#include <cstdio>

#ifdef NDEBUG
#define CHECK_ALIGNMENT(ptr, sz)
#else
#define CHECK_ALIGNMENT(ptr, sz) \
	if( uintptr_t(ptr) % uintptr_t(sz) != 0 ) { \
		printf( "Alignment error in %s on line %i!\n", __FILE__, __LINE__ ); \
		abort(); \
	}
#endif

namespace simd {

class simd_i32;
class simd_f32;

class simd_i32 {
	union {
		__m256i v;
		int w[8];
	};
public:
	simd_i32() = default;
	simd_i32(const simd_i32&) = default;
	simd_i32(simd_i32&&) = default;
	simd_i32& operator=(const simd_i32&) = default;
	simd_i32& operator=(simd_i32&&) = default;
	simd_i32(const simd_f32& other);
	int operator[](int i) const {
		CHECK_ALIGNMENT(this, 32);
		return w[i];
	}
	int& operator[](int i) {
		CHECK_ALIGNMENT(this, 32);
		return w[i];
	}
	inline simd_i32(int a) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_set1_epi32(a);
	}
	inline simd_i32(const std::initializer_list<int>& list) {
		CHECK_ALIGNMENT(this, 32);
		int i = 0;
		for (auto j = list.begin(); j != list.end(); j++) {
			w[i++] = *j;
		}
	}
	inline simd_i32& gather(const int* ptr, simd_i32 indices) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_i32gather_epi32(ptr, indices.v, sizeof(int));
		return *this;
	}
	inline simd_i32 permute(const simd_i32& indices) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_permutevar8x32_epi32(v, indices.v);
		return result;
	}
	inline simd_i32& operator+=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_add_epi32(v, other.v);
		return *this;
	}
	inline simd_i32& operator-=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_sub_epi32(v, other.v);
		return *this;
	}
	inline simd_i32& operator*=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_mul_epi32(v, other.v);
		return *this;
	}
	inline simd_i32& operator&=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_and_si256(v, other.v);
		return *this;
	}
	inline simd_i32& operator^=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_xor_si256(v, other.v);
		return *this;
	}
	inline simd_i32& operator|=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_or_si256(v, other.v);
		return *this;
	}
	inline simd_i32 operator&&(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result = (*this) & other;
		return result;
	}
	inline simd_i32 operator||(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result = *this | other;
		return result;
	}
	inline simd_i32 operator!() const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result = (*this) == simd_i32(0);
		return result;
	}
	inline simd_i32 operator+(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_add_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator-(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_sub_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator~() const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_andnot_si256(v, simd_i32(0xFFFFFFFF).v);
		return result;
	}
	inline simd_i32 operator*(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_mul_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator&(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_and_si256(v, other.v);
		return result;
	}
	inline simd_i32 operator^(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_xor_si256(v, other.v);
		return result;
	}
	inline simd_i32 operator|(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_or_si256(v, other.v);
		return result;
	}
	inline simd_i32 operator>>(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_srlv_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator<<(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_sllv_epi32(v, other.v);
		return result;
	}
	inline simd_i32& operator>>=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_srlv_epi32(v, other.v);
		return *this;
	}
	inline simd_i32& operator<<=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_sllv_epi32(v, other.v);
		return *this;
	}
	inline simd_i32 operator>>(int i) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_srli_epi32(v, i);
		return result;
	}
	inline simd_i32 operator<<(int i) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_slli_epi32(v, i);
		return result;
	}
	inline simd_i32& operator>>=(int i) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_srli_epi32(v, i);
		return *this;
	}
	inline simd_i32& operator<<=(int i) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_slli_epi32(v, i);
		return *this;
	}
	inline simd_i32 operator-() const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i32(0) - *this;
	}
	inline simd_i32 operator+() const {
		CHECK_ALIGNMENT(this, 32);
		return *this;
	}
	inline simd_i32 operator==(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_cmpeq_epi32(v, other.v);
		return -result;
	}
	inline simd_i32 operator!=(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i32(1) - (*this == other);
	}
	inline simd_i32 operator>(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_cmpgt_epi32(v, other.v);
		return -result;
	}
	inline simd_i32 operator>=(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		return ((*this == other) + (*this > other)) > simd_i32(0);
	}
	inline simd_i32 operator<(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i32(1) - (*this >= other);
	}
	inline simd_i32 operator<=(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i32(1) - (*this > other);
	}
	inline static constexpr size_t size() {
		return 8;
	}
	inline simd_i32& pad(int n) {
		CHECK_ALIGNMENT(this, 32);
		const int& e = size();
		for (int i = n; i < e; i++) {
			w[i] = w[0];
		}
		return *this;
	}
	static inline simd_i32 mask(int n) {
		simd_i32 mk;
		for (int i = 0; i < n; i++) {
			mk[i] = 1;
		}
		for (int i = n; i < size(); i++) {
			mk[i] = 0;
		}
		return mk;
	}
	inline void set_NaN() {
		CHECK_ALIGNMENT(this, 32);
		for (int i = 0; i < size(); i++) {
			w[i] = std::numeric_limits<int>::signaling_NaN();
		}
	}
	friend simd_i32 max(simd_i32, simd_i32);
	friend simd_i32 min(simd_i32, simd_i32);
	friend simd_i32 lround(simd_f32);
	friend simd_f32 blend(simd_f32, simd_f32, simd_i32);
	friend simd_f32;
};

inline simd_i32 max(simd_i32 a, simd_i32 b) {
	a.v = _mm256_max_epi32(a.v, b.v);
	return a;
}

inline simd_i32 min(simd_i32 a, simd_i32 b) {
	a.v = _mm256_min_epi32(a.v, b.v);
	return a;
}

class simd_f32 {
	__m256 v;
public:
	simd_f32() = default;
	simd_f32(const simd_f32&) = default;
	simd_f32(simd_f32&&) = default;
	simd_f32& operator=(const simd_f32&) = default;
	simd_f32& operator=(simd_f32&&) = default;
	inline float operator[](int i) const {
		CHECK_ALIGNMENT(this, 32);
		return v[i];
	}
	inline float& operator[](int i) {
		CHECK_ALIGNMENT(this, 32);
		return v[i];
	}
	inline simd_f32(float a) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_broadcast_ss(&a);
	}
	inline simd_f32(const std::initializer_list<float>& list) {
		CHECK_ALIGNMENT(this, 32);
		int i = 0;
		for (auto j = list.begin(); j != list.end(); j++) {
			v[i++] = *j;
		}
	}
	inline simd_f32(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_cvtepi32_ps(other.v);
	}
	inline simd_f32& gather(const float* ptr, simd_i32 indices) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_i32gather_ps(ptr, indices.v, sizeof(float));
		return *this;
	}
	inline simd_f32 permute(const simd_i32& indices) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f32 result;
		result.v = _mm256_permutevar8x32_ps(v, indices.v);
		return result;
	}
	inline simd_f32& operator+=(const simd_f32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_add_ps(v, other.v);
		return *this;
	}
	inline simd_f32& operator-=(const simd_f32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_sub_ps(v, other.v);
		return *this;
	}
	inline simd_f32& operator*=(const simd_f32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_mul_ps(v, other.v);
		return *this;
	}
	inline simd_f32& operator/=(const simd_f32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_div_ps(v, other.v);
		return *this;
	}
	inline simd_f32 operator+(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f32 result;
		result.v = _mm256_add_ps(v, other.v);
		return result;
	}
	inline simd_f32 operator-(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f32 result;
		result.v = _mm256_sub_ps(v, other.v);
		return result;
	}
	inline simd_f32 operator*(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f32 result;
		result.v = _mm256_mul_ps(v, other.v);
		return result;
	}
	inline simd_f32 operator/(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f32 result;
		result.v = _mm256_div_ps(v, other.v);
		return result;
	}
	inline simd_f32 operator-() const {
		CHECK_ALIGNMENT(this, 32);
		return simd_f32(0) - *this;
	}
	inline simd_f32 operator+() const {
		CHECK_ALIGNMENT(this, 32);
		return *this;
	}
	inline simd_i32 operator==(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_EQ_OS));
		return -result;
	}
	inline simd_i32 operator!=(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_NEQ_OS));
		return -result;
	}
	inline simd_i32 operator>(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_GT_OS));
		return -result;
	}
	inline simd_i32 operator>=(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_GE_OS));
		return -result;
	}
	inline simd_i32 operator<(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_LT_OS));
		return -result;
	}
	inline simd_i32 operator<=(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_LE_OS));
		return -result;
	}
	inline static constexpr size_t size() {
		return 8;
	}
	inline simd_f32& pad(int n) {
		CHECK_ALIGNMENT(this, 32);
		const int& e = size();
		for (int i = n; i < e; i++) {
			v[i] = v[0];
		}
		return *this;
	}
	static inline simd_f32 mask(int n) {
		simd_f32 mk;
		for (int i = 0; i < n; i++) {
			mk[i] = 1.f;
		}
		for (int i = n; i < size(); i++) {
			mk[i] = 0.f;
		}
		return mk;
	}
	inline void set_NaN() {
		CHECK_ALIGNMENT(this, 32);
		for (int i = 0; i < size(); i++) {
			v[i] = std::numeric_limits<float>::signaling_NaN();
		}
	}
	friend simd_f32 rint(simd_f32 x);
	friend simd_f32 sqrt(simd_f32);
	friend simd_f32 rsqrt(simd_f32);
	friend simd_f32 fma(simd_f32, simd_f32, simd_f32);
	friend float reduce_sum(simd_f32);
	friend simd_f32 trunc(simd_f32);
	friend simd_f32 round(simd_f32);
	friend simd_i32 lround(simd_f32);
	friend simd_f32 floor(simd_f32);
	friend simd_f32 ceil(simd_f32);
	friend simd_f32 fmax(simd_f32, simd_f32);
	friend simd_f32 fmin(simd_f32, simd_f32);
	friend simd_f32 blend(simd_f32, simd_f32, simd_i32);
	friend simd_f32 frexp(simd_f32, simd_i32*);
	friend class simd_f32_2;
	friend class simd_i32;
};

simd_f32 log10(simd_f32);
simd_f32 tgamma(simd_f32);
simd_f32 log2(simd_f32);
simd_f32 log(simd_f32);
simd_f32 sin(simd_f32);
simd_f32 cos(simd_f32 x);

simd_f32 exp(simd_f32);
simd_f32 exp2(simd_f32);
simd_f32 expm1(simd_f32);
simd_f32 erfc(simd_f32);
simd_f32 erf(simd_f32);
simd_f32 cbrt(simd_f32);
simd_f32 log1p(simd_f32);
simd_f32 pow(simd_f32 y, simd_f32 x);
simd_f32 atan(simd_f32);
simd_f32 asin(simd_f32);
simd_f32 acos(simd_f32 x);

inline simd_f32 nextafter(simd_f32 x, simd_f32 y) {
	/* Comparisons yield 1 or 0 here, never -1, so the step has to be built
	   explicitly; floats are sign-magnitude, so the bit-space direction is
	   reversed for negative x; and +-0 and NaN do not fall out of either. */
	const simd_i32 dir = (y > x) - (x > y);
	simd_i32 i = (simd_i32&) x;
	i += blend(dir, simd_i32(0) - dir, x < simd_f32(0));
	simd_f32 r = (simd_f32&) i;
	/* copysign() is declared further down, so take y's sign bit directly:
	   sign(y) | 1 is the smallest denormal with y's sign. */
	simd_i32 tiny = (((simd_i32&) y) & simd_i32((int) 0x80000000)) | simd_i32(1);
	r = blend(r, (simd_f32&) tiny, x == simd_f32(0));
	r = blend(r, y, x == y);
	return blend(r, y, !(y == y));   /* y is NaN; y != y is false here */
}

inline simd_f32 remainder(simd_f32 n, simd_f32 d) {
	const auto z = n / d;
	return n - d * round(z);
}

inline simd_f32 fmod(simd_f32 n, simd_f32 d) {
	const auto z = n / d;
	return n - d * trunc(z);
}

inline simd_f32 remquo(simd_f32 n, simd_f32 d, simd_i32* q) {
	const auto z = n / d;
	const auto qr = round(z);
   *q = lround(qr);
	return n - d * qr;
}

inline simd_f32 fabs(simd_f32 x) {
	simd_i32 i = (((simd_i32&) x) & simd_i32(0x7FFFFFFF));
	return (simd_f32&) i;
}

inline simd_f32 abs(simd_f32 x) {
	return fabs(x);
}

inline simd_f32 blend(simd_f32 a, simd_f32 b, simd_i32 mask) {
	mask = -mask;
	a.v = _mm256_blendv_ps(a.v, b.v, ((simd_f32&) mask).v);
	return a;
}

inline simd_f32 sinh(simd_f32 x) {
	return simd_f32(0.5) * (expm1(x) - expm1(-x));
}

inline simd_f32 cosh(simd_f32 x) {
	const auto z = exp(x);
	return (simd_f32(0.5) * z + simd_f32(0.5) / z);
}

inline simd_f32 tanh(simd_f32 x) {
	return sinh(x) / cosh(x);
}

inline simd_f32 asinh(simd_f32 x) {
	const auto y = log(x + sqrt(x * x + simd_f32(1)));
	const auto expm1p = expm1(y);
	const auto expm1m = expm1(-y);
	const auto sinhy = (expm1p - expm1m) * simd_f32(0.5);
	const auto coshy = (simd_f32(2) + expm1p + expm1m) * simd_f32(0.5);
	return y + (x - sinhy) / coshy;

}

inline simd_f32 fmax(simd_f32 a, simd_f32 b) {
	a.v = _mm256_max_ps(a.v, b.v);
	return a;
}

inline simd_f32 fmin(simd_f32 a, simd_f32 b) {
	a.v = _mm256_min_ps(a.v, b.v);
	return a;
}

inline simd_f32 atanh(simd_f32 x) {
	const auto y = simd_f32(0.5) * log((simd_f32(1) + x) / (simd_f32(1) - x));
	const auto expm1p = expm1(y);
	const auto expm1m = expm1(-y);
	const auto sinhy = (expm1p - expm1m) * simd_f32(0.5);
	const auto coshy = (simd_f32(2) + expm1p + expm1m) * simd_f32(0.5);
	return y + (x - sinhy / coshy) / (coshy * coshy);
}

inline simd_f32 round(simd_f32 x) {
	simd_f32 result;
	result.v = _mm256_round_ps(x.v, _MM_FROUND_TO_NEAREST_INT);
	return result;
}

inline simd_i32 lround(simd_f32 x) {
	simd_i32 result;
	result.v = _mm256_cvtps_epi32(_mm256_round_ps(x.v, _MM_FROUND_TO_NEAREST_INT));
	return result;
}

inline simd_f32 floor(simd_f32 x) {
	simd_f32 result;
	result.v = _mm256_floor_ps(x.v);
	return result;
}

inline simd_f32 ceil(simd_f32 x) {
	simd_f32 result;
	result.v = _mm256_ceil_ps(x.v);
	return result;
}

inline simd_f32 fdim(simd_f32 x, simd_f32 y) {
	return (x > y) * (x - y);
}

inline float reduce_sum(simd_f32 x) {
	__m128 a = *((__m128 *) &x.v);
	__m128 b = *(((__m128 *) &x.v) + 1);
	a = _mm_add_ps(a, b);
	return a[0] + a[1] + a[2] + a[3];
}

inline simd_f32 fma(simd_f32 a, simd_f32 b, simd_f32 c) {
	simd_f32 result;
	result.v = _mm256_fmadd_ps(a.v, b.v, c.v);
	return result;
}

inline simd_i32::simd_i32(const simd_f32& other) {
	v = _mm256_cvtps_epi32(_mm256_round_ps(other.v, _MM_FROUND_TO_ZERO));
}

inline simd_f32 sqrt(simd_f32 x) {
	x.v = _mm256_sqrt_ps(x.v);
	return x;
}

inline simd_f32 rsqrt(simd_f32 x) {
	x.v = _mm256_rsqrt_ps(x.v);
	return x;
}

inline simd_f32 copysign(simd_f32 x, simd_f32 y) {
	simd_f32 result = fabs(x);
	simd_i32 i = ((simd_i32&) result) | (simd_i32(0x80000000) & (simd_i32&) (y));
	result = (simd_f32&) i;
	return result;
}

inline simd_f32 atan2(simd_f32 y, simd_f32 x) {
	return atan(y / x) + copysign(copysign(M_PI_2, x) - simd_f32(M_PI_2), y);
}

inline simd_f32 frexp(simd_f32 x, simd_i32* e) {
	simd_i32 i, j;
	simd_f32 y;
	i = (simd_i32&) x;
	j = i & simd_i32(0x807FFFFF);
	j |= simd_i32(127 << 23);
	i &= simd_i32(0x7F800000);
	i >>= int(23);
	i -= simd_i32(127);
	*e = i;
	y = (simd_f32&) j;
	(*e) = (*e) + simd_i32(1);
	y *= simd_f32(0.5);
	return y;
}

inline simd_f32 modf(simd_f32 x, simd_f32* i) {
	*i = simd_f32(x);
	x -= *i;
	return x;
}

inline simd_f32 tan(simd_f32 x) {
	return sin(x) / cos(x);
}

inline simd_i32 ilogb(simd_f32 x) {
	simd_i32 i = (simd_i32&) x;
	i >>= 23;
	i -= 127;
	return i;
}

inline simd_f32 logb(simd_f32 x) {
	return log2(x);
}

inline simd_f32 ldexp(simd_f32 x, simd_i32 e) {
	e += simd_i32(127);
	e <<= 23;
	x *= (simd_f32&) e;
	return x;
}

inline simd_f32 hypot(simd_f32 x, simd_f32 y) {
	simd_i32 ix = (simd_i32&) x;
	simd_i32 iy = (simd_i32&) y;
	ix >>= 23;
	iy >>= 23;
	simd_i32 i = (ix + iy) >> 1;
	simd_i32 j = simd_i32(254) - i;
	i <<= 23;
	j <<= 23;
	const simd_f32 a = (simd_f32&) i;
	const simd_f32 b = (simd_f32&) j;
	x *= b;
	y *= b;
	return a * sqrt(x * x + y * y);
}

inline simd_f32 trunc(simd_f32 x) {
	x.v = _mm256_round_ps(x.v, _MM_FROUND_TO_ZERO);
	return x;
}

inline simd_f32 scalbn(simd_f32 x, simd_i32 n) {
	simd_i32 i = (simd_i32&) x;
	simd_i32 j = (simd_i32&) x;
	simd_f32 y;
	i >>= 23;
	i &= 0xFF;
	i += n;
	i <<= 23;
	j = j & 0x807FFFFF;
	j |= i;
	return (simd_f32&) j;
}

inline simd_f32 rint(simd_f32 x) {
	switch (fegetround()) {
	case FE_DOWNWARD:
		return floor(x);
	case FE_TONEAREST:
		return round(x);
	case FE_TOWARDZERO:
		return trunc(x);
	case FE_UPWARD:
		return ceil(x);
	}
}

inline simd_f32 nearbyint(simd_f32 x) {
	return rint(x);
}

class simd_i64;
class simd_f64;

class simd_i64 {
	union {
		__m256i v;
		int64_t w[4];
	};
public:
	simd_i64() = default;
	simd_i64(const simd_i64&) = default;
	simd_i64(simd_i64&&) = default;
	simd_i64& operator=(const simd_i64&) = default;
	simd_i64& operator=(simd_i64&&) = default;
	simd_i64(const simd_f64& other);
	long long operator[](int i) const {
		CHECK_ALIGNMENT(this, 32);
		return w[i];
	}
	int64_t& operator[](int i) {
		CHECK_ALIGNMENT(this, 32);
		return w[i];
	}
	inline simd_i64(long long a) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_set_epi64x(a, a, a, a);
	}
	inline simd_i64(const std::initializer_list<long long>& list) {
		CHECK_ALIGNMENT(this, 32);
		int i = 0;
		for (auto j = list.begin(); j != list.end(); j++) {
			(*this)[i++] = *j;
		}
	}
	inline simd_i64& gather(const long long* ptr, simd_i64 indices) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_i64gather_epi64(ptr, indices.v, sizeof(long long));
		return *this;
	}
	inline simd_i64& operator+=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_add_epi64(v, other.v);
		return *this;
	}
	inline simd_i64& operator-=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_sub_epi64(v, other.v);
		return *this;
	}
	inline simd_i64& operator&=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_and_si256(v, other.v);
		return *this;
	}
	inline simd_i64& operator^=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_xor_si256(v, other.v);
		return *this;
	}
	inline simd_i64& operator|=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_or_si256(v, other.v);
		return *this;
	}
	inline simd_i64 operator&&(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result = (*this) & other;
		return result;
	}
	inline simd_i64 operator||(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result = (*this) | other;
		return result;
	}
	inline simd_i64 operator!() const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result = (*this) == simd_i64(0);
		return result;
	}
	inline simd_i64 operator+(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_add_epi64(v, other.v);
		return result;
	}
	inline simd_i64 operator-(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_sub_epi64(v, other.v);
		return result;
	}
	inline simd_i64 operator~() const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_andnot_si256(v, simd_i64(0xFFFFFFFFFFFFFFFFLL).v);
		return result;
	}
	inline simd_i64 operator&(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_and_si256(v, other.v);
		return result;
	}
	inline simd_i64 operator^(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_xor_si256(v, other.v);
		return result;
	}
	inline simd_i64 operator|(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_or_si256(v, other.v);
		return result;
	}
	inline simd_i64 operator>>(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_srlv_epi64(v, other.v);
		return result;
	}
	inline simd_i64 operator<<(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_sllv_epi64(v, other.v);
		return result;
	}
	inline simd_i64 operator>>(long long i) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_srli_epi64(v, i);
		return result;
	}
	inline simd_i64 operator<<(long long i) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_slli_epi64(v, i);
		return result;
	}
	inline simd_i64& operator>>=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_srlv_epi64(v, other.v);
		return *this;
	}
	inline simd_i64& operator<<=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_sllv_epi64(v, other.v);
		return *this;
	}
	inline simd_i64& operator>>=(unsigned long long i) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_srli_epi64(v, i);
		return *this;
	}
	inline simd_i64& operator<<=(unsigned long long i) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_slli_epi64(v, i);
		return *this;
	}
	inline simd_i64 operator-() const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i64(0) - *this;
	}
	inline simd_i64 operator+() const {
		CHECK_ALIGNMENT(this, 32);
		return *this;
	}
	inline simd_i64 operator==(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_cmpeq_epi64(v, other.v);
		return -result;
	}
	inline simd_i64 operator!=(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i64(1) - (*this == other);
	}
	inline simd_i64 operator>(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_cmpgt_epi64(v, other.v);
		return -result;
	}
	inline simd_i64 operator>=(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		return ((*this == other) + (*this > other)) > simd_i64(0);
	}
	inline simd_i64 operator<(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i64(1) - (*this >= other);
	}
	inline simd_i64 operator<=(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i64(1) - (*this > other);
	}
	inline static constexpr size_t size() {
		return 4;
	}
	inline simd_i64& pad(int n) {
		const int& e = size();
		for (int i = n; i < e; i++) {
			v[i] = v[0];
		}
		return *this;
	}
	static inline simd_i64 mask(int n) {
		simd_i64 mk;
		for (int i = 0; i < n; i++) {
			mk[i] = 1;
		}
		for (int i = n; i < size(); i++) {
			mk[i] = 0;
		}
		return mk;
	}
	inline void set_NaN() {
		CHECK_ALIGNMENT(this, 32);
		for (int i = 0; i < size(); i++) {
			v[i] = std::numeric_limits<int>::signaling_NaN();
		}
	}
	friend simd_f64 blend(simd_f64, simd_f64, simd_i64);
	friend simd_i64 lround(simd_f64);
	friend simd_f64;
};


class simd_f64 {
	__m256d v;
public:
	simd_f64() = default;
	simd_f64(const simd_f64&) = default;
	simd_f64(simd_f64&&) = default;
	simd_f64& operator=(const simd_f64&) = default;
	simd_f64& operator=(simd_f64&&) = default;
	inline double operator[](int i) const {
		CHECK_ALIGNMENT(this, 32);
		return v[i];
	}
	inline double& operator[](int i) {
		CHECK_ALIGNMENT(this, 32);
		return v[i];
	}
	inline simd_f64(double a) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_broadcast_sd(&a);
	}
	inline simd_f64(const std::initializer_list<double>& list) {
		CHECK_ALIGNMENT(this, 32);
		int i = 0;
		for (auto j = list.begin(); j != list.end(); j++) {
			v[i++] = *j;
		}
	}
	inline simd_f64(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v[0] = (double) other[0];
		v[1] = (double) other[1];
		v[2] = (double) other[2];
		v[3] = (double) other[3];
	}
	inline simd_f64 permute(simd_i64 indices) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f64 result;
		__m256i i = indices.v;
		__m256d y = v;
		result.v = __builtin_shuffle(v, i);
		return result;
	}
	inline simd_f64& gather(const double* ptr, simd_i64 indices) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_i64gather_pd(ptr, indices.v, sizeof(double));
		return *this;
	}
	inline simd_f64& operator+=(const simd_f64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_add_pd(v, other.v);
		return *this;
	}
	inline simd_f64& operator-=(const simd_f64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_sub_pd(v, other.v);
		return *this;
	}
	inline simd_f64& operator*=(const simd_f64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_mul_pd(v, other.v);
		return *this;
	}
	inline simd_f64& operator/=(const simd_f64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = _mm256_div_pd(v, other.v);
		return *this;
	}
	inline simd_f64 operator+(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f64 result;
		result.v = _mm256_add_pd(v, other.v);
		return result;
	}
	inline simd_f64 operator-(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f64 result;
		result.v = _mm256_sub_pd(v, other.v);
		return result;
	}
	inline simd_f64 operator*(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f64 result;
		result.v = _mm256_mul_pd(v, other.v);
		return result;
	}
	inline simd_f64 operator/(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f64 result;
		result.v = _mm256_div_pd(v, other.v);
		return result;
	}
	inline simd_f64 operator-() const {
		CHECK_ALIGNMENT(this, 32);
		return simd_f64(0) - *this;
	}
	inline simd_f64 operator+() const {
		CHECK_ALIGNMENT(this, 32);
		return *this;
	}
	inline simd_i64 operator==(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_EQ_OS));
		return -result;
	}
	inline simd_i64 operator!=(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_NEQ_OS));
		return -result;
	}
	inline simd_i64 operator>(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_GT_OS));
		return -result;
	}
	inline simd_i64 operator>=(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_GE_OS));
		return -result;
	}
	inline simd_i64 operator<(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_LT_OS));
		return -result;
	}
	inline simd_i64 operator<=(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_LE_OS));
		return -result;
	}
	inline static constexpr size_t size() {
		return 4;
	}
	inline simd_f64& pad(int n) {
		CHECK_ALIGNMENT(this, 32);
		const int& e = size();
		for (int i = n; i < e; i++) {
			v[i] = v[0];
		}
		return *this;
	}
	static inline simd_f64 mask(int n) {
		simd_f64 mk;
		for (int i = 0; i < n; i++) {
			mk[i] = 1.f;
		}
		for (int i = n; i < size(); i++) {
			mk[i] = 0.f;
		}
		return mk;
	}
	inline void set_NaN() {
		CHECK_ALIGNMENT(this, 32);
		for (int i = 0; i < size(); i++) {
			v[i] = std::numeric_limits<double>::signaling_NaN();
		}
	}
	friend simd_f64 rint(simd_f64 x);
	friend simd_f64 sqrt(simd_f64);
	friend simd_f64 rsqrt(simd_f64);
	friend simd_f64 fma(simd_f64, simd_f64, simd_f64);
	friend double reduce_sum(simd_f64);
	friend simd_f64 trunc(simd_f64);
	friend simd_f64 round(simd_f64);
	friend simd_i64 lround(simd_f64);
	friend simd_f64 floor(simd_f64);
	friend simd_f64 ceil(simd_f64);
	friend simd_f64 fmax(simd_f64, simd_f64);
	friend simd_f64 fmin(simd_f64, simd_f64);
	friend simd_f64 blend(simd_f64, simd_f64, simd_i64);
	friend simd_f64 asin(simd_f64 x);
	friend class simd_i64;
	friend class simd_f64_2;
};

simd_f64 pow(simd_f64 y, simd_f64 x);
simd_f64 log(simd_f64);
simd_f64 log10(simd_f64);
simd_f64 log2(simd_f64);
simd_f64 cos(simd_f64);
simd_f64 asin(simd_f64);
simd_f64 exp(simd_f64);
simd_f64 exp2(simd_f64);
simd_f64 expm1(simd_f64);
simd_f64 erfc(simd_f64);
simd_f64 erf(simd_f64);
simd_f64 cbrt(simd_f64);
simd_f64 log1p(simd_f64);

inline simd_f64 hypot(simd_f64 x, simd_f64 y) {
	simd_i64 ix = (simd_i64&) x;
	simd_i64 iy = (simd_i64&) y;
	ix >>= 52;
	iy >>= 52;
	simd_i64 i = (ix + iy) >> 1;
	simd_i64 j = simd_i64(2026) - i;
	i <<= 52;
	j <<= 52;
	const simd_f64 a = (simd_f64&) i;
	const simd_f64 b = (simd_f64&) j;
	x *= b;
	y *= b;
	return a * sqrt(x * x + y * y);
}

inline simd_f64 fabs(simd_f64 x) {
	simd_i64 i = (((simd_i64&) x) & simd_i64(0x7FFFFFFFFFFFFFFFLL));
	return (simd_f64&) i;
}

inline simd_f64 abs(simd_f64 x) {
	return fabs(x);
}

simd_f64 acos(simd_f64 x);

simd_f64 atan(simd_f64 x);

inline simd_f64 blend(simd_f64 a, simd_f64 b, simd_i64 mask) {
	mask = -mask;
	a.v = _mm256_blendv_pd(a.v, b.v, ((simd_f64&) mask).v);
	return a;
}

inline simd_f64 asinh(simd_f64 x) {
	const auto y = log(x + sqrt(x * x + simd_f64(1)));
	const auto expm1p = expm1(y);
	const auto expm1m = expm1(-y);
	const auto sinhy = (expm1p - expm1m) * simd_f64(0.5);
	const auto coshy = (simd_f64(2) + expm1p + expm1m) * simd_f64(0.5);
	return y + (x - sinhy) / coshy;

}

inline simd_f64 atanh(simd_f64 x) {
	const auto y = simd_f64(0.5) * log((simd_f64(1) + x) / (simd_f64(1) - x));
	const auto expm1p = expm1(y);
	const auto expm1m = expm1(-y);
	const auto sinhy = (expm1p - expm1m) * simd_f64(0.5);
	const auto coshy = (simd_f64(2) + expm1p + expm1m) * simd_f64(0.5);
	return y + (x - sinhy / coshy) / (coshy * coshy);
}

simd_f64 exp2(simd_f64 x);

inline simd_f64 copysign(simd_f64 x, simd_f64 y) {
	simd_f64 result = fabs(x);
	simd_i64 i = ((simd_i64&) result) | (simd_i64(0x8000000000000000LL) & (simd_i64&) (y));
	result = (simd_f64&) i;
	return result;
}

inline simd_f64 atan2(simd_f64 y, simd_f64 x) {
	return atan(y / x) + copysign(copysign(M_PI_2, x) - simd_f64(M_PI_2), y);
}

inline simd_f64 fmax(simd_f64 a, simd_f64 b) {
	a.v = _mm256_max_pd(a.v, b.v);
	return a;
}

inline simd_f64 fmin(simd_f64 a, simd_f64 b) {
	a.v = _mm256_min_pd(a.v, b.v);
	return a;
}

inline simd_f64 round(simd_f64 x) {
	simd_f64 result;
	result.v = _mm256_round_pd(x.v, _MM_FROUND_TO_NEAREST_INT);
	return result;
}


inline simd_i64 lround(simd_f64 x) {
	simd_i64 result;
	result.v = _mm256_cvtpd_epi64(_mm256_round_pd(x.v, _MM_FROUND_TO_NEAREST_INT));
	return result;
}

inline simd_f64 trunc(simd_f64 x) {
	x.v = _mm256_round_pd(x.v, _MM_FROUND_TO_ZERO);
	return x;
}

inline simd_f64 floor(simd_f64 x) {
	simd_f64 result;
	result.v = _mm256_floor_pd(x.v);
	return result;
}

inline simd_f64 ceil(simd_f64 x) {
	simd_f64 result;
	result.v = _mm256_ceil_pd(x.v);
	return result;
}

inline simd_f64 frexp(simd_f64 x, simd_i64* e) {
	simd_i64 i, j;
	simd_f64 y;
	i = (simd_i64&) x;
	j = i & simd_i64(0x800FFFFFFFFFFFFFULL);
	j |= simd_i64(1023ULL << 52ULL);
	i &= simd_i64(0x7FF0000000000000ULL);
	i >>= (long long) (52);
	i -= simd_i64(1023);
	*e = i;
	y = (simd_f64&) j;
	(*e) = (*e) + simd_i64(1);
	y *= simd_f64(0.5);
	return y;
}

inline simd_f64 ldexp(simd_f64 x, simd_i64 e) {
	e += simd_i64(1023);
	e <<= (long long) 52;
	x *= (simd_f64&) e;
	return x;
}

inline simd_f64 fma(simd_f64 a, simd_f64 b, simd_f64 c) {
	simd_f64 result;
	result.v = _mm256_fmadd_pd(a.v, b.v, c.v);
	return result;
}

inline simd_i64::simd_i64(const simd_f64& other) {
	v[0] = (long long) (other[0]);
	v[1] = (long long) (other[1]);
	v[2] = (long long) (other[2]);
	v[3] = (long long) (other[3]);
}

simd_f64 tgamma(simd_f64);
simd_f64 lgamma(simd_f64);
simd_f32 lgamma(simd_f32);
simd_f32 tgamma(simd_f32);

inline simd_f64 sqrt(simd_f64 x) {
	x.v = _mm256_sqrt_pd(x.v);
	return x;
}

inline simd_f64 rsqrt(simd_f64 x) {
	return simd_f64(1) / sqrt(x);
}

inline simd_f64 sinh(simd_f64 x) {
	return simd_f64(0.5) * (expm1(x) - expm1(-x));
}

inline simd_f64 cosh(simd_f64 x) {
	return simd_f64(0.5) * (exp(x) + exp(-x));
}

inline simd_f64 tanh(simd_f64 x) {
	return sinh(x) / cosh(x);
}

simd_f64 sin(simd_f64 x);

inline simd_f64 tan(simd_f64 x) {
	return sin(x) / cos(x);
}

inline simd_f64 modf(simd_f64 x, simd_f64* i) {
	*i = simd_f64(x);
	x -= *i;
	return x;
}

inline simd_i64 ilogb(simd_f64 x) {
	simd_i64 i = (simd_i64&) x;
	i >>= 52;
	i -= 1023;
	return i;
}

inline simd_f64 logb(simd_f64 x) {
	return log2(x);
}

inline simd_f64 scalbn(simd_f64 x, simd_i64 n) {
	simd_i64 i = (simd_i64&) x;
	simd_i64 j = (simd_i64&) x;
	simd_f64 y;
	i >>= 52;
	i &= 0xFFFFFFFFFFFFFLL;
	i += n;
	i <<= 52;
	j = j & 0x800FFFFFFFFFFFFFLL;
	j |= i;
	return (simd_f64&) j;
}

inline simd_f64 rint(simd_f64 x) {
	switch (fegetround()) {
	case FE_DOWNWARD:
		return floor(x);
	case FE_TONEAREST:
		return round(x);
	case FE_TOWARDZERO:
		return trunc(x);
	case FE_UPWARD:
		return ceil(x);
	}
}

inline simd_f64 nearbyint(simd_f64 x) {
	return rint(x);
}

struct simd_f32_2 {
	simd_f32 x;
	simd_f32 y;
	static inline simd_f32_2 __attribute__((optimize("O3"))) quick_two_sum(simd_f32 a_, simd_f32 b_) {
		simd_f32_2 r;
		const __m256& a = a_.v;
		const __m256& b = b_.v;
		r.x.v = _mm256_add_ps(a, b);
		r.y.v = _mm256_sub_ps(r.x.v, a);
		r.y.v = _mm256_sub_ps(b, r.y.v);
		return r;

	}
	static inline simd_f32_2 __attribute__((optimize("O3"))) two_sum(simd_f32 a_, simd_f32 b_) {
		simd_f32_2 r;
		const __m256& a = a_.v;
		const __m256& b = b_.v;
		__m256& s = r.x.v;
		__m256& e = r.y.v;
		s = _mm256_add_ps(a, b);
		const __m256 v = _mm256_sub_ps(s, a);
		e = _mm256_sub_ps(s, v);
		e = _mm256_add_ps(_mm256_sub_ps(a, e), _mm256_sub_ps(b, v));
		return r;
	}
	static inline simd_f32_2 __attribute__((optimize("O3"))) two_product(simd_f32 a_, simd_f32 b_) {
		simd_f32_2 r;
		const __m256& a = a_.v;
		const __m256& b = b_.v;
		r.x.v = _mm256_mul_ps(a, b);
		r.y.v = _mm256_fmsub_ps(a, b, r.x.v);
		return r;
	}
public:
	inline simd_f32_2& operator=(simd_f32 a) {
		const static float zero = 0.0;
		x = a;
		y.v = _mm256_broadcast_ss(&zero);
		return *this;
	}
	inline simd_f32_2(simd_f32 a, simd_f32 b) {
		x = a;
		y = b;
	}
	inline simd_f32_2(simd_f32 a) {
		*this = a;
	}
	inline simd_f32_2() {
	}
	inline operator simd_f32() const {
		return x + y;
	}
	inline simd_f32_2 operator+(simd_f32 other) const {
		simd_f32_2 s;
		s = two_sum(x, other);
		s.y += y;
		s = quick_two_sum(s.x, s.y);
		return s;
	}
	inline simd_f32_2 operator*(simd_f32 other) const {
		simd_f32_2 p;
		p = two_product(x, other);
		p.y = fma(y, other, p.y);
		p = quick_two_sum(p.x, p.y);
		return p;
	}
	inline simd_f32_2 operator-(simd_f32 other) const {
		return *this + -other;
	}
	inline simd_f32_2 operator+(simd_f32_2 other) const {
		simd_f32_2 s, t;
		s = two_sum(x, other.x);
		t = two_sum(y, other.y);
		s.y += t.x;
		s = quick_two_sum(s.x, s.y);
		s.y += t.y;
		s = quick_two_sum(s.x, s.y);
		return s;
	}
	inline simd_f32_2 operator*(simd_f32_2 other) const {
		simd_f32_2 p;
		p = two_product(x, other.x);
		p.y += x * other.y;
		p.y += y * other.x;
		p = quick_two_sum(p.x, p.y);
		return p;
	}
	inline simd_f32_2 operator-() const {
		simd_f32_2 r;
		r.x = -x;
		r.y = -y;
		return r;
	}
	inline simd_f32_2 operator/(const simd_f32_2 A) const {
		simd_f32_2 result;
		const simd_f32 xn = simd_f32(1) / A.x;
		const simd_f32 yn = x * xn;
		const simd_f32_2 diff = (*this - A * simd_f32(yn));
		const simd_f32_2 prod = two_product(xn, diff);
		return simd_f32_2(yn) + prod;
	}
	inline simd_f32_2 operator-(simd_f32_2 other) const {
		return *this + -other;
	}
};

inline simd_f32_2 operator+(simd_f32 a, simd_f32_2 b) {
	return b + a;
}

inline simd_f32_2 operator*(simd_f32 a, simd_f32_2 b) {
	return b * a;
}

inline simd_f32_2 operator-(simd_f32 a, simd_f32_2 b) {
	return -b + a;
}

inline simd_f32_2 sqr(simd_f32_2 A) {
	simd_f32_2 p;
	p = simd_f32_2::two_product(A.x, A.x);
	p.y += simd_f32(2) * A.x * A.y;
	p = simd_f32_2::quick_two_sum(p.x, p.y);
	return p;
}

inline simd_f32_2 sqrt(simd_f32_2 X) {
	simd_f32_2 Y = sqrt(X.x);
	Y = Y + (X / Y - Y) * simd_f32(0.5);
	return Y;
}

struct simd_f64_2 {
	simd_f64 x;
	simd_f64 y;
	static inline simd_f64_2 __attribute__((optimize("O3"))) quick_two_sum(simd_f64 a_, simd_f64 b_) {
		simd_f64_2 r;
		const __m256d& a = a_.v;
		const __m256d& b = b_.v;
		r.x.v = _mm256_add_pd(a, b);
		r.y.v = _mm256_sub_pd(r.x.v, a);
		r.y.v = _mm256_sub_pd(b, r.y.v);
		return r;

	}
	static inline simd_f64_2 __attribute__((optimize("O3"))) two_sum(simd_f64 a_, simd_f64 b_) {
		simd_f64_2 r;
		const __m256d& a = a_.v;
		const __m256d& b = b_.v;
		__m256d& s = r.x.v;
		__m256d& e = r.y.v;
		s = _mm256_add_pd(a, b);
		const __m256d v = _mm256_sub_pd(s, a);
		e = _mm256_sub_pd(s, v);
		e = _mm256_add_pd(_mm256_sub_pd(a, e), _mm256_sub_pd(b, v));
		return r;
	}
	static inline simd_f64_2 __attribute__((optimize("O3"))) two_product(simd_f64 a_, simd_f64 b_) {
		simd_f64_2 r;
		const __m256d& a = a_.v;
		const __m256d& b = b_.v;
		r.x.v = _mm256_mul_pd(a, b);
		r.y.v = _mm256_fmsub_pd(a, b, r.x.v);
		return r;
	}
public:
	inline simd_f64_2& operator=(simd_f64 a) {
		const static double zero = 0.0;
		x = a;
		y.v = _mm256_broadcast_sd(&zero);
		return *this;
	}
	inline simd_f64_2(simd_f64 a, simd_f64 b) {
		x = a;
		y = b;
	}
	inline simd_f64_2(simd_f64 a) {
		*this = a;
	}
	inline simd_f64_2() {
	}
	inline operator simd_f64() const {
		return x + y;
	}
	inline simd_f64_2 operator+(simd_f64 other) const {
		simd_f64_2 s;
		s = two_sum(x, other);
		s.y += y;
		s = quick_two_sum(s.x, s.y);
		return s;
	}
	inline simd_f64_2 operator*(simd_f64 other) const {
		simd_f64_2 p;
		p = two_product(x, other);
		p.y = fma(y, other, p.y);
		p = quick_two_sum(p.x, p.y);
		return p;
	}
	inline simd_f64_2 operator-(simd_f64 other) const {
		return *this + -other;
	}
	inline simd_f64_2 operator+(simd_f64_2 other) const {
		simd_f64_2 s, t;
		s = two_sum(x, other.x);
		t = two_sum(y, other.y);
		s.y += t.x;
		s = quick_two_sum(s.x, s.y);
		s.y += t.y;
		s = quick_two_sum(s.x, s.y);
		return s;
	}
	inline simd_f64_2 operator*(simd_f64_2 other) const {
		simd_f64_2 p;
		p = two_product(x, other.x);
		p.y += x * other.y;
		p.y += y * other.x;
		p = quick_two_sum(p.x, p.y);
		return p;
	}
	inline simd_f64_2 operator-() const {
		simd_f64_2 r;
		r.x = -x;
		r.y = -y;
		return r;
	}
	inline simd_f64_2 operator/(const simd_f64_2 A) const {
		simd_f64_2 result;
		const simd_f64 xn = simd_f64(1) / A.x;
		const simd_f64 yn = x * xn;
		const simd_f64_2 diff = (*this - A * simd_f64(yn));
		const simd_f64_2 prod = two_product(xn, diff);
		return simd_f64_2(yn) + prod;
	}
	inline simd_f64_2 operator-(simd_f64_2 other) const {
		return *this + -other;
	}
};

inline simd_f64_2 operator+(simd_f64 a, simd_f64_2 b) {
	return b + a;
}

inline simd_f64_2 operator*(simd_f64 a, simd_f64_2 b) {
	return b * a;
}

inline simd_f64_2 operator-(simd_f64 a, simd_f64_2 b) {
	return -b + a;
}

inline simd_f64_2 sqr(simd_f64_2 A) {
	simd_f64_2 p;
	p = simd_f64_2::two_product(A.x, A.x);
	p.y += simd_f64(2) * A.x * A.y;
	p = simd_f64_2::quick_two_sum(p.x, p.y);
	return p;
}

inline simd_f64_2 sqrt(simd_f64_2 X) {
	simd_f64_2 Y = sqrt(X.x);
	Y = Y + (X / Y - Y) * simd_f64(0.5);
	return Y;
}

inline simd_f64 remainder(simd_f64 n, simd_f64 d) {
	const auto z = n / d;
	return n - d * round(z);
}

inline simd_f64 remquo(simd_f64 n, simd_f64 d, simd_i64* q) {
	const auto z = n / d;
	const auto qr = round(z);
   *q = lround(qr);
	return n - d * qr;
}

inline simd_f64 fmod(simd_f64 n, simd_f64 d) {
	const auto z = n;
	return n - d * trunc(z);
}

inline simd_f32 acosh(simd_f32 x) {
	const auto Z = sqrt(simd_f32_2::two_product(x, x) - simd_f32(1));
	const auto P = simd_f32_2::two_sum(x, simd_f32(-1));
	simd_f32_2 Q, W;
	Q = simd_f32_2::two_sum(Z.x, P.x);
	W = simd_f32_2::two_sum(Z.y, P.y);
	Q.y += W.x;
	Q = simd_f32_2::quick_two_sum(Q.x, Q.y);
	Q.y += W.y;
	return log1p(Q.x + Q.y);
}

inline simd_f64 acosh(simd_f64 x) {
	const auto Z = sqrt(simd_f64_2::two_product(x, x) - simd_f64(1));
	const auto P = simd_f64_2::two_sum(x, simd_f64(-1));
	simd_f64_2 Q, W;
	Q = simd_f64_2::two_sum(Z.x, P.x);
	W = simd_f64_2::two_sum(Z.y, P.y);
	Q.y += W.x;
	Q = simd_f64_2::quick_two_sum(Q.x, Q.y);
	Q.y += W.y;
	return log1p(Q.x + Q.y);
}

inline simd_f64 nextafter(simd_f64 x, simd_f64 y) {
	/* See the simd_f32 overload for why each step is needed. */
	const simd_i64 dir = (y > x) - (x > y);
	simd_i64 i = (simd_i64&) x;
	i += blend(dir, simd_i64(0) - dir, x < simd_f64(0));
	simd_f64 r = (simd_f64&) i;
	simd_i64 tiny = (((simd_i64&) y) & simd_i64((long long) 0x8000000000000000ULL)) | simd_i64(1);
	r = blend(r, (simd_f64&) tiny, x == simd_f64(0));
	r = blend(r, y, x == y);
	return blend(r, y, !(y == y));   /* y is NaN; y != y is false here */
}

inline simd_f64 fdim(simd_f64 x, simd_f64 y) {
	return (x > y) * (x - y);
}

inline double reduce_sum(simd_f64 x) {
	__m128d a = *((__m128d *) &x.v);
	__m128d b = *(((__m128d *) &x.v) + 1);
	a = _mm_add_pd(a, b);
	return a[0] + a[1];
}

}

#include <vector>
