#pragma once

/* The vector primitives live in a per-architecture backend header; see
   PORT-AARCH64.md. Everything below is written against that interface, so a
   new architecture is a new backend rather than edits throughout this file. */
#if defined(__aarch64__) && defined(SIMD_NATIVE_NEON)
#include "simd_backend_neon.hpp"
#else
#include "simd_backend_x86.hpp"
#endif

#include <limits>
#include <type_traits>
#include <mutex>
#include <cfenv>
#include <cmath>
#include <initializer_list>
#include <cstdio>

#ifdef NDEBUG
#define CHECK_ALIGNMENT(ptr, sz)
#else
/* The `sz` argument is ignored: every call site passes a literal 32, the AVX2
   object size, which is wrong wherever the vector type is aligned differently
   (16 under SIMDe on NEON, and again in a native NEON backend). Taking the
   alignment from the pointee is correct everywhere and leaves the 128 call
   sites alone. The parameter is kept so those call sites still compile. */
#define CHECK_ALIGNMENT(ptr, sz) \
	if( uintptr_t(ptr) % alignof(std::remove_reference_t<decltype(*(ptr))>) != 0 ) { \
		printf( "Alignment error in %s on line %i!\n", __FILE__, __LINE__ ); \
		abort(); \
	}
#endif

namespace simd {

class simd_i32;
class simd_f32;

class simd_i32 {
	union {
		backend::i32v v;
		int w[backend::i32_lanes];
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
		v = backend::i32_set1(a);
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
		v = backend::i32_gather(ptr, indices.v);
		return *this;
	}
	inline simd_i32 permute(const simd_i32& indices) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::i32_permute(v, indices.v);
		return result;
	}
	inline simd_i32& operator+=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i32_add(v, other.v);
		return *this;
	}
	inline simd_i32& operator-=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i32_sub(v, other.v);
		return *this;
	}
	inline simd_i32& operator*=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i32_mul(v, other.v);
		return *this;
	}
	inline simd_i32& operator&=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i32_and(v, other.v);
		return *this;
	}
	inline simd_i32& operator^=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i32_xor(v, other.v);
		return *this;
	}
	inline simd_i32& operator|=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i32_or(v, other.v);
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
		result.v = backend::i32_add(v, other.v);
		return result;
	}
	inline simd_i32 operator-(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::i32_sub(v, other.v);
		return result;
	}
	inline simd_i32 operator~() const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::i32_andnot(v, simd_i32(0xFFFFFFFF).v);
		return result;
	}
	inline simd_i32 operator*(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::i32_mul(v, other.v);
		return result;
	}
	inline simd_i32 operator&(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::i32_and(v, other.v);
		return result;
	}
	inline simd_i32 operator^(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::i32_xor(v, other.v);
		return result;
	}
	inline simd_i32 operator|(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::i32_or(v, other.v);
		return result;
	}
	inline simd_i32 operator>>(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::i32_srlv(v, other.v);
		return result;
	}
	inline simd_i32 operator<<(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::i32_sllv(v, other.v);
		return result;
	}
	inline simd_i32& operator>>=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i32_srlv(v, other.v);
		return *this;
	}
	inline simd_i32& operator<<=(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i32_sllv(v, other.v);
		return *this;
	}
	inline simd_i32 operator>>(int i) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::i32_srli(v, i);
		return result;
	}
	inline simd_i32 operator<<(int i) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::i32_slli(v, i);
		return result;
	}
	inline simd_i32& operator>>=(int i) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i32_srli(v, i);
		return *this;
	}
	inline simd_i32& operator<<=(int i) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i32_slli(v, i);
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
		result.v = backend::i32_cmpeq(v, other.v);
		return -result;
	}
	inline simd_i32 operator!=(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i32(1) - (*this == other);
	}
	inline simd_i32 operator>(const simd_i32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::i32_cmpgt(v, other.v);
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
		return (size_t) backend::i32_lanes;
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
	friend simd_f32 from_bits(simd_i32);
	friend simd_i32 to_bits(simd_f32);
	friend simd_i32 max(simd_i32, simd_i32);
	friend simd_i32 min(simd_i32, simd_i32);
	friend simd_i32 lround(simd_f32);
	friend simd_f32 blend(simd_f32, simd_f32, simd_i32);
	friend simd_f32;
};

inline simd_i32 max(simd_i32 a, simd_i32 b) {
	a.v = backend::i32_max(a.v, b.v);
	return a;
}

inline simd_i32 min(simd_i32 a, simd_i32 b) {
	a.v = backend::i32_min(a.v, b.v);
	return a;
}

class simd_f32 {
#if defined(SIMD_BACKEND_LANE_UNION)
	/* Only where the backend's vector type cannot be subscripted. On the
	   native backend the bare member is essential: adding the array alongside
	   it defeats GCC's register promotion of this class and measured 2.5x
	   slower on x86. See PORT-AARCH64.md. */
	union {
		backend::f32v v;
		float w[backend::f32_lanes];
	};
	inline float& lane(int i) { return w[i]; }
	inline float lane(int i) const { return w[i]; }
#else
	backend::f32v v;
	inline float& lane(int i) { return v[i]; }
	inline float lane(int i) const { return v[i]; }
#endif
public:
	simd_f32() = default;
	simd_f32(const simd_f32&) = default;
	simd_f32(simd_f32&&) = default;
	simd_f32& operator=(const simd_f32&) = default;
	simd_f32& operator=(simd_f32&&) = default;
	inline float operator[](int i) const {
		CHECK_ALIGNMENT(this, 32);
		return lane(i);
	}
	inline float& operator[](int i) {
		CHECK_ALIGNMENT(this, 32);
		return lane(i);
	}
	inline simd_f32(float a) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f32_set1(a);
	}
	inline simd_f32(const std::initializer_list<float>& list) {
		CHECK_ALIGNMENT(this, 32);
		int i = 0;
		for (auto j = list.begin(); j != list.end(); j++) {
			lane(i++) = *j;
		}
	}
	inline simd_f32(const simd_i32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f32_from_i32(other.v);
	}
	inline simd_f32& gather(const float* ptr, simd_i32 indices) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f32_gather(ptr, indices.v);
		return *this;
	}
	inline simd_f32 permute(const simd_i32& indices) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f32 result;
		result.v = backend::f32_permute(v, indices.v);
		return result;
	}
	inline simd_f32& operator+=(const simd_f32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f32_add(v, other.v);
		return *this;
	}
	inline simd_f32& operator-=(const simd_f32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f32_sub(v, other.v);
		return *this;
	}
	inline simd_f32& operator*=(const simd_f32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f32_mul(v, other.v);
		return *this;
	}
	inline simd_f32& operator/=(const simd_f32& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f32_div(v, other.v);
		return *this;
	}
	inline simd_f32 operator+(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f32 result;
		result.v = backend::f32_add(v, other.v);
		return result;
	}
	inline simd_f32 operator-(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f32 result;
		result.v = backend::f32_sub(v, other.v);
		return result;
	}
	inline simd_f32 operator*(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f32 result;
		result.v = backend::f32_mul(v, other.v);
		return result;
	}
	inline simd_f32 operator/(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f32 result;
		result.v = backend::f32_div(v, other.v);
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
		result.v = backend::f32_cmp_eq(v, other.v);
		return -result;
	}
	inline simd_i32 operator!=(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::f32_cmp_neq(v, other.v);
		return -result;
	}
	inline simd_i32 operator>(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::f32_cmp_gt(v, other.v);
		return -result;
	}
	inline simd_i32 operator>=(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::f32_cmp_ge(v, other.v);
		return -result;
	}
	inline simd_i32 operator<(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::f32_cmp_lt(v, other.v);
		return -result;
	}
	inline simd_i32 operator<=(const simd_f32& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i32 result;
		result.v = backend::f32_cmp_le(v, other.v);
		return -result;
	}
	inline static constexpr size_t size() {
		return (size_t) backend::f32_lanes;
	}
	inline simd_f32& pad(int n) {
		CHECK_ALIGNMENT(this, 32);
		const int& e = size();
		for (int i = n; i < e; i++) {
			lane(i) = lane(0);
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
			lane(i) = std::numeric_limits<float>::signaling_NaN();
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
	friend simd_f32 from_bits(simd_i32);
	friend simd_i32 to_bits(simd_f32);
	friend class simd_f32_2;
	friend class simd_i32;
};

/* Reinterpret a vector's bits between the float and integer classes. These
   replace casts of the form to_bits(x), which were undefined behaviour and
   had started to miscompile (TODO item 7). They lower to a register-level
   reinterpret with no memory traffic. */
inline simd_i32 to_bits(simd_f32 x) {
	simd_i32 r;
	r.v = backend::f32_as_i32(x.v);
	return r;
}

inline simd_f32 from_bits(simd_i32 i) {
	simd_f32 r;
	r.v = backend::i32_as_f32(i.v);
	return r;
}

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
	simd_i32 i = to_bits(x);
	i += blend(dir, simd_i32(0) - dir, x < simd_f32(0));
	simd_f32 r = from_bits(i);
	/* copysign() is declared further down, so take y's sign bit directly:
	   sign(y) | 1 is the smallest denormal with y's sign. */
	simd_i32 tiny = (to_bits(y) & simd_i32((int) 0x80000000)) | simd_i32(1);
	r = blend(r, from_bits(tiny), x == simd_f32(0));
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
	simd_i32 i = (to_bits(x) & simd_i32(0x7FFFFFFF));
	return from_bits(i);
}

inline simd_f32 abs(simd_f32 x) {
	return fabs(x);
}

inline simd_f32 blend(simd_f32 a, simd_f32 b, simd_i32 mask) {
	mask = -mask;
	a.v = backend::f32_blendv(a.v, b.v, backend::i32_as_f32(mask.v));
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
	a.v = backend::f32_max(a.v, b.v);
	return a;
}

inline simd_f32 fmin(simd_f32 a, simd_f32 b) {
	a.v = backend::f32_min(a.v, b.v);
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
	result.v = backend::f32_round_nearest(x.v);
	return result;
}

inline simd_i32 lround(simd_f32 x) {
	simd_i32 result;
	result.v = backend::f32_to_i32_nearest(x.v);
	return result;
}

inline simd_f32 floor(simd_f32 x) {
	simd_f32 result;
	result.v = backend::f32_floor(x.v);
	return result;
}

inline simd_f32 ceil(simd_f32 x) {
	simd_f32 result;
	result.v = backend::f32_ceil(x.v);
	return result;
}

inline simd_f32 fdim(simd_f32 x, simd_f32 y) {
	return (x > y) * (x - y);
}

inline float reduce_sum(simd_f32 x) {
	constexpr int H = backend::f32_lanes / 2;
	float a[H];
	for (int i = 0; i < H; i++) {
		a[i] = x.lane(i) + x.lane(i + H);
	}
	float s = a[0];
	for (int i = 1; i < H; i++) {
		s += a[i];
	}
	return s;
}

inline simd_f32 fma(simd_f32 a, simd_f32 b, simd_f32 c) {
	simd_f32 result;
	result.v = backend::f32_fmadd(a.v, b.v, c.v);
	return result;
}

inline simd_i32::simd_i32(const simd_f32& other) {
	v = backend::f32_to_i32_zero(other.v);
}

inline simd_f32 sqrt(simd_f32 x) {
	x.v = backend::f32_sqrt(x.v);
	return x;
}

inline simd_f32 rsqrt(simd_f32 x) {
	x.v = backend::f32_rsqrt(x.v);
	return x;
}

inline simd_f32 copysign(simd_f32 x, simd_f32 y) {
	simd_f32 result = fabs(x);
	simd_i32 i = to_bits(result) | (simd_i32(0x80000000) & to_bits(y));
	result = from_bits(i);
	return result;
}

inline simd_f32 atan2(simd_f32 y, simd_f32 x) {
	return atan(y / x) + copysign(copysign(M_PI_2, x) - simd_f32(M_PI_2), y);
}

inline simd_f32 frexp(simd_f32 x, simd_i32* e) {
	simd_i32 i, j;
	simd_f32 y;
	i = to_bits(x);
	j = i & simd_i32(0x807FFFFF);
	j |= simd_i32(127 << 23);
	i &= simd_i32(0x7F800000);
	i >>= int(23);
	i -= simd_i32(127);
	*e = i;
	y = from_bits(j);
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
	simd_i32 i = to_bits(x);
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
	x *= from_bits(e);
	return x;
}

inline simd_f32 hypot(simd_f32 x, simd_f32 y) {
	simd_i32 ix = to_bits(x);
	simd_i32 iy = to_bits(y);
	ix >>= 23;
	iy >>= 23;
	simd_i32 i = (ix + iy) >> 1;
	simd_i32 j = simd_i32(254) - i;
	i <<= 23;
	j <<= 23;
	const simd_f32 a = from_bits(i);
	const simd_f32 b = from_bits(j);
	x *= b;
	y *= b;
	return a * sqrt(x * x + y * y);
}

inline simd_f32 trunc(simd_f32 x) {
	x.v = backend::f32_round_zero(x.v);
	return x;
}

inline simd_f32 scalbn(simd_f32 x, simd_i32 n) {
	simd_i32 i = to_bits(x);
	simd_i32 j = to_bits(x);
	simd_f32 y;
	i >>= 23;
	i &= 0xFF;
	i += n;
	i <<= 23;
	j = j & 0x807FFFFF;
	j |= i;
	return from_bits(j);
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
		backend::i64v v;
		int64_t w[backend::i64_lanes];
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
		v = backend::i64_set1(a);
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
		v = backend::i64_gather(ptr, indices.v);
		return *this;
	}
	inline simd_i64& operator+=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i64_add(v, other.v);
		return *this;
	}
	inline simd_i64& operator-=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i64_sub(v, other.v);
		return *this;
	}
	inline simd_i64& operator&=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i64_and(v, other.v);
		return *this;
	}
	inline simd_i64& operator^=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i64_xor(v, other.v);
		return *this;
	}
	inline simd_i64& operator|=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i64_or(v, other.v);
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
		result.v = backend::i64_add(v, other.v);
		return result;
	}
	inline simd_i64 operator-(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::i64_sub(v, other.v);
		return result;
	}
	inline simd_i64 operator~() const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::i64_andnot(v, simd_i64(0xFFFFFFFFFFFFFFFFLL).v);
		return result;
	}
	inline simd_i64 operator&(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::i64_and(v, other.v);
		return result;
	}
	inline simd_i64 operator^(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::i64_xor(v, other.v);
		return result;
	}
	inline simd_i64 operator|(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::i64_or(v, other.v);
		return result;
	}
	inline simd_i64 operator>>(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::i64_srlv(v, other.v);
		return result;
	}
	inline simd_i64 operator<<(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::i64_sllv(v, other.v);
		return result;
	}
	inline simd_i64 operator>>(long long i) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::i64_srli(v, i);
		return result;
	}
	inline simd_i64 operator<<(long long i) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::i64_slli(v, i);
		return result;
	}
	inline simd_i64& operator>>=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i64_srlv(v, other.v);
		return *this;
	}
	inline simd_i64& operator<<=(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i64_sllv(v, other.v);
		return *this;
	}
	inline simd_i64& operator>>=(unsigned long long i) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i64_srli(v, i);
		return *this;
	}
	inline simd_i64& operator<<=(unsigned long long i) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::i64_slli(v, i);
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
		result.v = backend::i64_cmpeq(v, other.v);
		return -result;
	}
	inline simd_i64 operator!=(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		return simd_i64(1) - (*this == other);
	}
	inline simd_i64 operator>(const simd_i64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::i64_cmpgt(v, other.v);
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
		return (size_t) backend::i64_lanes;
	}
	inline simd_i64& pad(int n) {
		const int& e = size();
		for (int i = n; i < e; i++) {
			w[i] = w[0];
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
			w[i] = std::numeric_limits<int>::signaling_NaN();
		}
	}
	friend simd_f64 blend(simd_f64, simd_f64, simd_i64);
	friend simd_i64 lround(simd_f64);
	friend simd_i64 to_bits(simd_f64);
	friend simd_f64 from_bits(simd_i64);
	friend int movemask(simd_i64);
	friend simd_f64;
};


class simd_f64 {
#if defined(SIMD_BACKEND_LANE_UNION)
	/* Only where the backend's vector type cannot be subscripted. On the
	   native backend the bare member is essential: adding the array alongside
	   it defeats GCC's register promotion of this class and measured 2.5x
	   slower on x86. See PORT-AARCH64.md. */
	union {
		backend::f64v v;
		double w[backend::f64_lanes];
	};
	inline double& lane(int i) { return w[i]; }
	inline double lane(int i) const { return w[i]; }
#else
	backend::f64v v;
	inline double& lane(int i) { return v[i]; }
	inline double lane(int i) const { return v[i]; }
#endif
public:
	simd_f64() = default;
	simd_f64(const simd_f64&) = default;
	simd_f64(simd_f64&&) = default;
	simd_f64& operator=(const simd_f64&) = default;
	simd_f64& operator=(simd_f64&&) = default;
	inline double operator[](int i) const {
		CHECK_ALIGNMENT(this, 32);
		return lane(i);
	}
	inline double& operator[](int i) {
		CHECK_ALIGNMENT(this, 32);
		return lane(i);
	}
	inline simd_f64(double a) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f64_set1(a);
	}
	inline simd_f64(const std::initializer_list<double>& list) {
		CHECK_ALIGNMENT(this, 32);
		int i = 0;
		for (auto j = list.begin(); j != list.end(); j++) {
			lane(i++) = *j;
		}
	}
	inline simd_f64(const simd_i64& other) {
		CHECK_ALIGNMENT(this, 32);
		lane(0) = (double) other[0];
		lane(1) = (double) other[1];
		lane(2) = (double) other[2];
		lane(3) = (double) other[3];
	}
	inline simd_f64 permute(simd_i64 indices) const {
		CHECK_ALIGNMENT(this, 32);
		/* __builtin_shuffle needs a GCC vector type, which SIMDe's simde__m256d
		   is not. Same semantics: result[k] = v[indices[k]], the index taken
		   modulo the lane count as the builtin does. */
		simd_f64 result;
		for (int k = 0; k < (int) size(); k++) {
			result.lane(k) = lane(indices[k] & (backend::f64_lanes - 1));
		}
		return result;
	}
	inline simd_f64& gather(const double* ptr, simd_i64 indices) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f64_gather(ptr, indices.v);
		return *this;
	}
	inline simd_f64& operator+=(const simd_f64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f64_add(v, other.v);
		return *this;
	}
	inline simd_f64& operator-=(const simd_f64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f64_sub(v, other.v);
		return *this;
	}
	inline simd_f64& operator*=(const simd_f64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f64_mul(v, other.v);
		return *this;
	}
	inline simd_f64& operator/=(const simd_f64& other) {
		CHECK_ALIGNMENT(this, 32);
		v = backend::f64_div(v, other.v);
		return *this;
	}
	inline simd_f64 operator+(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f64 result;
		result.v = backend::f64_add(v, other.v);
		return result;
	}
	inline simd_f64 operator-(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f64 result;
		result.v = backend::f64_sub(v, other.v);
		return result;
	}
	inline simd_f64 operator*(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f64 result;
		result.v = backend::f64_mul(v, other.v);
		return result;
	}
	inline simd_f64 operator/(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_f64 result;
		result.v = backend::f64_div(v, other.v);
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
		result.v = backend::f64_cmp_eq(v, other.v);
		return -result;
	}
	inline simd_i64 operator!=(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::f64_cmp_neq(v, other.v);
		return -result;
	}
	inline simd_i64 operator>(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::f64_cmp_gt(v, other.v);
		return -result;
	}
	inline simd_i64 operator>=(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::f64_cmp_ge(v, other.v);
		return -result;
	}
	inline simd_i64 operator<(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::f64_cmp_lt(v, other.v);
		return -result;
	}
	inline simd_i64 operator<=(const simd_f64& other) const {
		CHECK_ALIGNMENT(this, 32);
		simd_i64 result;
		result.v = backend::f64_cmp_le(v, other.v);
		return -result;
	}
	inline static constexpr size_t size() {
		return (size_t) backend::f64_lanes;
	}
	inline simd_f64& pad(int n) {
		CHECK_ALIGNMENT(this, 32);
		const int& e = size();
		for (int i = n; i < e; i++) {
			lane(i) = lane(0);
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
			lane(i) = std::numeric_limits<double>::signaling_NaN();
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
	friend simd_f64 from_bits(simd_i64);
	friend simd_i64 to_bits(simd_f64);
	friend class simd_i64;
	friend class simd_f64_2;
};

inline simd_i64 to_bits(simd_f64 x) {
	simd_i64 r;
	r.v = backend::f64_as_i64(x.v);
	return r;
}

inline simd_f64 from_bits(simd_i64 i) {
	simd_f64 r;
	r.v = backend::i64_as_f64(i.v);
	return r;
}

/* Sign bit of each lane, one bit per lane. The generated asin(simd_f64) uses
   it to index a table with one row per mask value. */
inline int movemask(simd_i64 i) {
	return backend::f64_movemask(backend::i64_as_f64(i.v));
}

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
	simd_i64 ix = to_bits(x);
	simd_i64 iy = to_bits(y);
	ix >>= 52;
	iy >>= 52;
	simd_i64 i = (ix + iy) >> 1;
	simd_i64 j = simd_i64(2026) - i;
	i <<= 52;
	j <<= 52;
	const simd_f64 a = from_bits(i);
	const simd_f64 b = from_bits(j);
	x *= b;
	y *= b;
	return a * sqrt(x * x + y * y);
}

inline simd_f64 fabs(simd_f64 x) {
	simd_i64 i = (to_bits(x) & simd_i64(0x7FFFFFFFFFFFFFFFLL));
	return from_bits(i);
}

inline simd_f64 abs(simd_f64 x) {
	return fabs(x);
}

simd_f64 acos(simd_f64 x);

simd_f64 atan(simd_f64 x);

inline simd_f64 blend(simd_f64 a, simd_f64 b, simd_i64 mask) {
	mask = -mask;
	a.v = backend::f64_blendv(a.v, b.v, backend::i64_as_f64(mask.v));
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
	simd_i64 i = to_bits(result) | (simd_i64(0x8000000000000000LL) & to_bits(y));
	result = from_bits(i);
	return result;
}

inline simd_f64 atan2(simd_f64 y, simd_f64 x) {
	return atan(y / x) + copysign(copysign(M_PI_2, x) - simd_f64(M_PI_2), y);
}

inline simd_f64 fmax(simd_f64 a, simd_f64 b) {
	a.v = backend::f64_max(a.v, b.v);
	return a;
}

inline simd_f64 fmin(simd_f64 a, simd_f64 b) {
	a.v = backend::f64_min(a.v, b.v);
	return a;
}

inline simd_f64 round(simd_f64 x) {
	simd_f64 result;
	result.v = backend::f64_round_nearest(x.v);
	return result;
}


inline simd_i64 lround(simd_f64 x) {
	simd_i64 result;
	result.v = backend::f64_to_i64_nearest(x.v);
	return result;
}

inline simd_f64 trunc(simd_f64 x) {
	x.v = backend::f64_round_zero(x.v);
	return x;
}

inline simd_f64 floor(simd_f64 x) {
	simd_f64 result;
	result.v = backend::f64_floor(x.v);
	return result;
}

inline simd_f64 ceil(simd_f64 x) {
	simd_f64 result;
	result.v = backend::f64_ceil(x.v);
	return result;
}

inline simd_f64 frexp(simd_f64 x, simd_i64* e) {
	simd_i64 i, j;
	simd_f64 y;
	i = to_bits(x);
	j = i & simd_i64(0x800FFFFFFFFFFFFFULL);
	j |= simd_i64(1023ULL << 52ULL);
	i &= simd_i64(0x7FF0000000000000ULL);
	i >>= (long long) (52);
	i -= simd_i64(1023);
	*e = i;
	y = from_bits(j);
	(*e) = (*e) + simd_i64(1);
	y *= simd_f64(0.5);
	return y;
}

inline simd_f64 ldexp(simd_f64 x, simd_i64 e) {
	e += simd_i64(1023);
	e <<= (long long) 52;
	x *= from_bits(e);
	return x;
}

inline simd_f64 fma(simd_f64 a, simd_f64 b, simd_f64 c) {
	simd_f64 result;
	result.v = backend::f64_fmadd(a.v, b.v, c.v);
	return result;
}

inline simd_i64::simd_i64(const simd_f64& other) {
	w[0] = (long long) (other[0]);
	w[1] = (long long) (other[1]);
	w[2] = (long long) (other[2]);
	w[3] = (long long) (other[3]);
}

simd_f64 tgamma(simd_f64);
simd_f64 lgamma(simd_f64);
simd_f32 lgamma(simd_f32);
simd_f32 tgamma(simd_f32);

inline simd_f64 sqrt(simd_f64 x) {
	x.v = backend::f64_sqrt(x.v);
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
	simd_i64 i = to_bits(x);
	i >>= 52;
	i -= 1023;
	return i;
}

inline simd_f64 logb(simd_f64 x) {
	return log2(x);
}

inline simd_f64 scalbn(simd_f64 x, simd_i64 n) {
	simd_i64 i = to_bits(x);
	simd_i64 j = to_bits(x);
	simd_f64 y;
	i >>= 52;
	i &= 0xFFFFFFFFFFFFFLL;
	i += n;
	i <<= 52;
	j = j & 0x800FFFFFFFFFFFFFLL;
	j |= i;
	return from_bits(j);
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
		const backend::f32v& a = a_.v;
		const backend::f32v& b = b_.v;
		r.x.v = backend::f32_add(a, b);
		r.y.v = backend::f32_sub(r.x.v, a);
		r.y.v = backend::f32_sub(b, r.y.v);
		return r;

	}
	static inline simd_f32_2 __attribute__((optimize("O3"))) two_sum(simd_f32 a_, simd_f32 b_) {
		simd_f32_2 r;
		const backend::f32v& a = a_.v;
		const backend::f32v& b = b_.v;
		backend::f32v& s = r.x.v;
		backend::f32v& e = r.y.v;
		s = backend::f32_add(a, b);
		const backend::f32v v = backend::f32_sub(s, a);
		e = backend::f32_sub(s, v);
		e = backend::f32_add(backend::f32_sub(a, e), backend::f32_sub(b, v));
		return r;
	}
	static inline simd_f32_2 __attribute__((optimize("O3"))) two_product(simd_f32 a_, simd_f32 b_) {
		simd_f32_2 r;
		const backend::f32v& a = a_.v;
		const backend::f32v& b = b_.v;
		r.x.v = backend::f32_mul(a, b);
		r.y.v = backend::f32_fmsub(a, b, r.x.v);
		return r;
	}
public:
	inline simd_f32_2& operator=(simd_f32 a) {
		const static float zero = 0.0;
		x = a;
		y.v = backend::f32_set1(zero);
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
		const backend::f64v& a = a_.v;
		const backend::f64v& b = b_.v;
		r.x.v = backend::f64_add(a, b);
		r.y.v = backend::f64_sub(r.x.v, a);
		r.y.v = backend::f64_sub(b, r.y.v);
		return r;

	}
	static inline simd_f64_2 __attribute__((optimize("O3"))) two_sum(simd_f64 a_, simd_f64 b_) {
		simd_f64_2 r;
		const backend::f64v& a = a_.v;
		const backend::f64v& b = b_.v;
		backend::f64v& s = r.x.v;
		backend::f64v& e = r.y.v;
		s = backend::f64_add(a, b);
		const backend::f64v v = backend::f64_sub(s, a);
		e = backend::f64_sub(s, v);
		e = backend::f64_add(backend::f64_sub(a, e), backend::f64_sub(b, v));
		return r;
	}
	static inline simd_f64_2 __attribute__((optimize("O3"))) two_product(simd_f64 a_, simd_f64 b_) {
		simd_f64_2 r;
		const backend::f64v& a = a_.v;
		const backend::f64v& b = b_.v;
		r.x.v = backend::f64_mul(a, b);
		r.y.v = backend::f64_fmsub(a, b, r.x.v);
		return r;
	}
public:
	inline simd_f64_2& operator=(simd_f64 a) {
		const static double zero = 0.0;
		x = a;
		y.v = backend::f64_set1(zero);
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
	simd_i64 i = to_bits(x);
	i += blend(dir, simd_i64(0) - dir, x < simd_f64(0));
	simd_f64 r = from_bits(i);
	simd_i64 tiny = (to_bits(y) & simd_i64((long long) 0x8000000000000000ULL)) | simd_i64(1);
	r = blend(r, from_bits(tiny), x == simd_f64(0));
	r = blend(r, y, x == y);
	return blend(r, y, !(y == y));   /* y is NaN; y != y is false here */
}

inline simd_f64 fdim(simd_f64 x, simd_f64 y) {
	return (x > y) * (x - y);
}

inline double reduce_sum(simd_f64 x) {
	constexpr int H = backend::f64_lanes / 2;
	double a[H];
	for (int i = 0; i < H; i++) {
		a[i] = x.lane(i) + x.lane(i + H);
	}
	double s = a[0];
	for (int i = 1; i < H; i++) {
		s += a[i];
	}
	return s;
}

}

#include <vector>
