#include "simd.hpp"
#include <stdio.h>
#include <stdlib.h>

#include <limits>
#include <functional>
#include <cmath>
#include <thread>
#include <fenv.h>
#include "hiprec.hpp"
#include <future>
#include <vector>
#include "polynomial.hpp"

#include <complex>
#include <valarray>

class timer {
	double start_time;
	double time;
public:
	inline timer() {
		time = 0.0;
	}
	inline void stop() {
		struct timespec res;
		clock_gettime(CLOCK_MONOTONIC, &res);
		const double stop_time = res.tv_sec + res.tv_nsec / 1e9;
		time += stop_time - start_time;
	}
	inline void start() {
		struct timespec res;
		clock_gettime(CLOCK_MONOTONIC, &res);
		start_time = res.tv_sec + res.tv_nsec / 1e9;
	}
	inline void reset() {
		time = 0.0;
	}
	inline double read() {
		return time;
	}
};

double rand1() {
	return (rand() + 0.5) / RAND_MAX;
}

struct test_result_t {
	double avg_err;
	double max_err;
	double speed;
};

#define List(...) {__VA_ARGS__}

hiprec_real zetam1(int s) {
	return zeta(s) - hiprec_real(1);
}

hiprec_real polygamma(int n, hiprec_real x) {
	using real = hiprec_real;
	static bool init = false;
	static constexpr int M = 256;
	static constexpr double base = 1e38;
	static std::vector<std::vector<hiprec_real>> coeffs;
	static std::vector<hiprec_real> factorial(1, 1);
	if (!init) {
		init = true;
		coeffs.resize(1);
		coeffs[0].resize(M);
		coeffs[0][0] = digamma(1.0);
		for (int m = 1; m < M; m++) {
			coeffs[0][m] = pow(-hiprec_real(1), hiprec_real(m + 1)) * (zetam1(m + 1) + hiprec_real(1));
		}
	}
	if (n >= coeffs.size()) {
		int p = coeffs.size();
		int N = n + 1;
		coeffs.resize(n + 1);
		while (p != N) {
			hiprec_real pfac = 1.0L;
			for (int l = 1; l <= p; l++) {
				pfac *= hiprec_real(l);
			}
			factorial.push_back(pfac);
			coeffs[p].resize(M);
			for (int m = 0; m < M - 1; m++) {
				coeffs[p][m] = coeffs[p - 1][m + 1] * hiprec_real((m + 1));
			}
			int m = M - 1;
			coeffs[p][m] = pow(-hiprec_real(1), hiprec_real(p + m + 1)) * (zetam1(p + m + 1) + hiprec_real(1)) * pfac;
			p++;
		}
	}
	hiprec_real y = 0.0;
	int xi = roundl(x);
	x -= hiprec_real(xi);
	xi -= 1;
	hiprec_real x1 = x;
	for (int m = M - 1; m >= 0; m--) {
		y = y * x1 + coeffs[n][m];
	}
	hiprec_real sgn = pow(-hiprec_real(1), hiprec_real(n));

	while (xi > 0) {
		y += hiprec_real(sgn * factorial[n]) * pow(hiprec_real(x) + hiprec_real(xi), hiprec_real(-(n + 1)));
		xi--;
	}
	if (xi < 0) {
		xi = -xi;
	}
	while (xi > 0) {
		y -= sgn * factorial[n] * pow(hiprec_real(x) - hiprec_real(xi - 1), -hiprec_real(n + 1));
		xi--;
	}
	return y;
}

double epserr(double x0, double y0) {
	int i, j;
	frexp(x0, &i);
	i--;
	frexp(y0, &j);
	j--;
	i = std::max(i, j);
	double err = (x0 - y0) / ldexp(1.0, i);
	return fabs(err);
}

float epserr(float x0, float y0) {
	int i, j;
	frexpf(x0, &i);
	i--;
	frexpf(y0, &j);
	j--;
	i = std::max(i, j);
	float err = (x0 - y0) / ldexpf(1.0f, i);
	return fabsf(err);
}

/* Number of vector-width samples per accuracy/speed test: N = 1 << N_bit_shift.
   Each test allocates buffers proportional to N, so lowering this lowers the
   memory footprint and the runtime. */
constexpr int N_bit_shift = 21;

template<class T, class V, class F1, class F2>
test_result_t test_function(const F1& ref, const F2& test, T a, T b, bool relerr) {
	constexpr int N = 1 << N_bit_shift;
	constexpr int size = V::size();
	static T* xref;
	static V* xtest;
	static T* yref;
	static V* ytest;
	bool flag = true;
	if (xref == nullptr) {
		flag = (posix_memalign((void**) &xref, 32, sizeof(T) * size * N) != 0) && flag;
		flag = (posix_memalign((void**) &xtest, 32, sizeof(V) * N) != 0) && flag;
		flag = (posix_memalign((void**) &yref, 32, sizeof(T) * size * N) != 0) && flag;
		flag = (posix_memalign((void**) &ytest, 32, sizeof(V) * N) != 0) && flag;
	}

	int nthreads = 1;
	for (int i = 0; i < size * N; i++) {
		do {
			xref[i] = rand1() * ((double) b - (double) a) + (double) a;
		} while (xref[i] == 0.0);
	}
	for (int i = 0; i < size * N; i += size) {
		for (int j = 0; j < size; j++) {
			xtest[i / size][j] = xref[i + j];
		}
	}
	timer tref, ttest;
	std::vector<std::future<void>> futs;
	futs.reserve(nthreads);
	tref.start();
	for (int i = 0; i < nthreads; i++) {
		futs.push_back(std::async([&]() {
			for (int i = 0; i < size * N; i++) {
				yref[i] = ref(xref[i]);
			}
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	tref.stop();
	futs.resize(0);
	ttest.start();
	for (int i = 0; i < nthreads; i++) {
		futs.push_back(std::async([&]() {
			for (int i = 0; i < N; i++) {
				ytest[i] = test(xtest[i]);
			}
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	ttest.stop();
	double ref_time = tref.read();
	double test_time = ttest.read();
	double speed = ref_time / test_time;
	double err = 0.0;
	double max_err = 0.0;
	for (int i = 0; i < size * N; i += size) {
		for (int j = 0; j < size; j++) {
			T a = yref[i + j];
			T b = ytest[i / size][j];
			double this_err = epserr(b, a);
			err += this_err;
			if (this_err / std::numeric_limits<double>::epsilon() > 3) {
				//	printf("%e %e %e %e\n", xref[i + j], a, b, this_err / std::numeric_limits<double>::epsilon());
			}
			max_err = std::max((double) max_err, (double) this_err);
		}
	}
	err /= N * size;
	test_result_t result;
	result.speed = speed;
	result.max_err = max_err / std::numeric_limits<T>::epsilon();
	result.avg_err = err / std::numeric_limits<T>::epsilon();
	/* Every TEST macro expansion instantiates this template afresh, because the
	   lambdas give each call site its own closure types -- so these statics are
	   per-test, not shared. Release them here; otherwise ~1 GiB of buffers
	   accumulates per test until the process is OOM-killed. */
	free(xref);
	free(xtest);
	free(yref);
	free(ytest);
	xref = nullptr;
	xtest = nullptr;
	yref = nullptr;
	ytest = nullptr;
	return result;
}

template<class T, class V, class F1, class F2>
test_result_t test_function(const F1& ref, const F2& test, T a0, T b0, T a1, T b1, bool relerr) {
	constexpr int N = 1 << N_bit_shift;
	constexpr int size = V::size();
	static T* xref;
	static V* xtest;
	static T* yref;
	static V* ytest;
	static T* zref;
	static V* ztest;
	bool flag = true;
	if (xref == nullptr) {
		flag = (posix_memalign((void**) &xref, 32, sizeof(T) * size * N) != 0) && flag;
		flag = (posix_memalign((void**) &xtest, 32, sizeof(V) * N) != 0) && flag;
		flag = (posix_memalign((void**) &yref, 32, sizeof(T) * size * N) != 0) && flag;
		flag = (posix_memalign((void**) &ytest, 32, sizeof(V) * N) != 0) && flag;
		flag = (posix_memalign((void**) &zref, 32, sizeof(T) * size * N) != 0) && flag;
		flag = (posix_memalign((void**) &ztest, 32, sizeof(V) * N) != 0) && flag;
	}
	int nthreads = 1;
	for (int i = 0; i < size * N; i++) {
		do {
			xref[i] = rand1() * ((double) b0 - (double) a0) + (double) a0;
		} while (xref[i] == 0.0);
		do {
			yref[i] = rand1() * ((double) b1 - (double) a1) + (double) a1;
		} while (yref[i] == 0.0);
	}
	for (int i = 0; i < size * N; i += size) {
		for (int j = 0; j < size; j++) {
			xtest[i / size][j] = xref[i + j];
			ytest[i / size][j] = yref[i + j];
		}
	}
	timer tref, ttest;
	std::vector<std::future<void>> futs;
	futs.reserve(nthreads);
	tref.start();
	for (int i = 0; i < nthreads; i++) {
		futs.push_back(std::async([&]() {
			for (int i = 0; i < size * N; i++) {
				zref[i] = ref(xref[i], yref[i]);
			}
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	tref.stop();
	futs.resize(0);
	ttest.start();
	for (int i = 0; i < nthreads; i++) {
		futs.push_back(std::async([&]() {
			for (int i = 0; i < N; i++) {
				ztest[i] = test(xtest[i], ytest[i]);
			}
		}));
	}
	for (auto& f : futs) {
		f.get();
	}
	ttest.stop();
	double ref_time = tref.read();
	double test_time = ttest.read();
	double speed = ref_time / test_time;
	double err = 0.0;
	double max_err = 0.0;
	for (int i = 0; i < size * N; i += size) {
		for (int j = 0; j < size; j++) {
			T a = zref[i + j];
			T b = ztest[i / size][j];
			double this_err = epserr(b, a);
			err += this_err;
			max_err = std::max((double) max_err, (double) this_err);
		}
	}
	err /= N * size;
	test_result_t result;
	result.speed = speed;
	result.max_err = max_err / std::numeric_limits<T>::epsilon();
	result.avg_err = err / std::numeric_limits<T>::epsilon();
	/* Per-instantiation statics; see the single-argument overload above. */
	free(xref);
	free(xtest);
	free(yref);
	free(ytest);
	free(zref);
	free(ztest);
	xref = nullptr;
	xtest = nullptr;
	yref = nullptr;
	ytest = nullptr;
	zref = nullptr;
	ztest = nullptr;
	return result;
}

#define TEST1(type, vtype, name, Ref, Test, a, b, rel) \
{ \
	auto ref = [=](type x) { \
		return Ref(x); \
	}; \
	auto test = [=](vtype x) { \
		return Test(x); \
	}; \
	auto res = test_function<type, vtype, decltype(ref), decltype(test)>(ref, test, (a), (b), rel); \
	printf("%6s %12e %12e %12e\n", #name, res.speed, res.avg_err, res.max_err); \
}

#define TEST2(type, vtype, name, Ref, Test, a0, b0, a1, b1, rel) \
{ \
	auto ref = [=](type x, type y) { \
		return Ref(x, y); \
	}; \
	auto test = [=](vtype x, vtype y) { \
		return Test(x, y); \
	}; \
	auto res = test_function<type, vtype, decltype(ref), decltype(test)>(ref, test, (a0), (b0), (a1), (b1), rel); \
	printf("%6s %12e %12e %12e\n", #name, res.speed, res.avg_err, res.max_err); \
}

namespace simd {
}

void cordic(double theta, double& sintheta, double& costheta) {
	constexpr int N = 28;
	static double dphi1[N];
	static double dphi2[N];
	static double dphi3[N];
	static double taneps[N];
	static double twopiinv;
	static double factor;
	static std::once_flag once;
	static double dtheta1;
	static double dtheta2;
	static double dtheta3;
	static hiprec_real pi_exact = hiprec_real(4) * atan(hiprec_real(1));
	static double dbin = pi_exact * pow(hiprec_real(2), -hiprec_real(N));
	std::call_once(once, []() {
		hiprec_real factor_ = 1.0;
		hiprec_real dtheta0 = hiprec_real(2) * pi_exact;
		dtheta1 = dtheta0;
		dtheta2 = dtheta0 - hiprec_real(dtheta1);
		dtheta3 = dtheta0 - hiprec_real(dtheta1) - hiprec_real(dtheta2);
		for( int n = 0; n < N; n++) {
			hiprec_real eps = pi_exact * pow(hiprec_real(2), -hiprec_real(n));
			taneps[n] = tan(eps);
			hiprec_real phi0 = eps;
			dphi1[n] = phi0;
			dphi2[n] = phi0 - hiprec_real(dphi1[n]);
			dphi3[n] = phi0 - hiprec_real(dphi1[n]) - hiprec_real(dphi2[n]);
			if( n != 1 ) {
				factor_ *= hiprec_real(1) / sqrt(hiprec_real(1) + tan(eps) * tan(eps));
			}
		}
		factor = factor_;
	});
	int index = std::round(theta / dtheta1);
	theta = std::fma(double(-index), dtheta3, std::fma(double(-index), dtheta2, std::fma(double(-index), dtheta1, theta)));
	double x = 1.0;
	double y = 0.0;
	double phi1 = 0.0;
	double phi2 = 0.0;
	double phi3 = 0.0;
	for (int n = 0; n < N; n++) {
		double eps = taneps[n];
		double x0 = x;
		double y0 = y;
		if (phi3 + phi2 + phi1 + dphi1[n] / 2.0 < theta) {
			if (n == 1) {
				std::swap(x, y);
				y = -y;
			} else {
				x = std::fma(eps, -y0, x);
				y = std::fma(eps, x0, y);
			}
			phi1 += dphi1[n];
			phi2 += dphi2[n];
			phi3 += dphi3[n];
		} else if (phi3 + phi2 + phi1 - dphi1[n] / 2.0 > theta) {
			if (n == 1) {
				std::swap(x, y);
				x = -x;
			} else {
				x = std::fma(eps, y0, x);
				y = std::fma(eps, -x0, y);
			}
			phi1 -= dphi1[n];
			phi2 -= dphi2[n];
			phi3 -= dphi3[n];
		}
	}
	x = (double) x * factor;
	y = (double) y * factor;
	double eps = ((theta - phi1) - phi2) - phi3;
	costheta = x;
	sintheta = y;
	sintheta = std::fma(eps, x, sintheta);
	costheta = std::fma(eps, -y, costheta);
}

double cos_test(double theta) {
	constexpr int N = 11;
	static std::once_flag once;
	static double coeff[N];
	static hiprec_real pi_exact = hiprec_real(4) * atan(hiprec_real(1));
	static double pi1 = pi_exact;
	static double pi2 = pi_exact - hiprec_real(pi1);
	std::call_once(once, []() {
		hiprec_real fac(-1);
		for( int n = 0; n < N; n++) {
			coeff[n] = hiprec_real(1) / fac;
			fac *= hiprec_real(2 * n + 3);
			fac *= -hiprec_real(2 * n + 2);
		}
	});
	theta = abs(theta);
	int i = std::floor(theta / M_PI_2);
	i |= 1;
	int sgn = -2 * (i / 2 - 2 * (i / 4)) + 1;
	double x = sgn * std::fma(-double(i), pi2 * 0.5, std::fma(-double(i), pi1 * 0.5, theta));
	double x2 = x * x;
	double y = coeff[N - 1];
	for (int n = N - 2; n >= 0; n--) {
		y = std::fma(x2, y, coeff[n]);
	}
	y *= x;
	return y;
}

float acos_test(float x) {
	constexpr int N = 18;
	constexpr float coeffs[] = { 0, 2., 0.3333333333, 0.08888888889, 0.02857142857, 0.01015873016, 0.003848003848, 0.001522287237, 0.0006216006216,
			0.0002600159463, 0.0001108489034, 0.00004798653828, 0.00002103757656, 9.321264693e-6, 4.167443738e-6, 1.877744765e-6, 8.517995405e-7, 3.886999686e-7 };
	float sgn = copysign(1.0f, x);
	x = abs(x);
	x = 1.f - x;
	float y = coeffs[N - 1];
	for (int n = N - 2; n >= 0; n--) {
		y = std::fma(x, y, coeffs[n]);
	}
	y = sqrt(std::max(y, 0.f));
	long double exact_pi = 4.0L * atanl(1.0L);
	float pi1 = exact_pi;
	float pi2 = exact_pi - (long double) pi1;
	if (sgn == -1.f) {
		y = pi1 - y;
		y += pi2;
	}
	return y;
}

float log2_test(float x) {
	constexpr int N = 5;
	int i = std::ilogbf(x * sqrtf(2));
	x = ldexp(x, -i);
	printf("%e\n", x - 1.0);
	float y = (x - 1.0) / (x + 1.0);
	float y2 = y * y;
	float z = 2.0 / (2 * (N - 1) + 1);
	for (int n = N - 2; n >= 0; n--) {
		z = fma(z, y2, 2.0 / (2 * n + 1) / log(2));
	}
	z *= y;
	return z + i;
}

hiprec_real factorial(int n) {
	constexpr static int N = 100;
	static hiprec_real results[N];
	static bool have[N];
	static bool init = false;
	if (!init) {
		for (int n = 0; n < N; n++) {
			have[n] = false;
		}
		init = true;
	}
	hiprec_real res;
	if (n < N) {
		if (!have[n]) {
			if (n < 2) {
				res = hiprec_real(1);
			} else {
				res = hiprec_real(n) * factorial(n - 1);
			}
			have[n] = true;
			results[n] = res;
		} else {
			res = results[n];
		}
	} else {
		if (n < 2) {
			res = hiprec_real(1);
		} else {
			res = hiprec_real(n) * factorial(n - 1);
		}

	}
	return res;
}

float pow_test(float x, float y) {
	constexpr int Nlogbits = 11;
	constexpr int Ntable = 1 << Nlogbits;
	static float log2hi[Ntable];
	static float log2lo[Ntable];
	static bool init = false;
	float lomax = 0.0;
	if (!init) {
		for (int n = 0; n < Ntable; n++) {
			int i = (n << (23 - Nlogbits)) | (127 << 23);
			float a = (float&) i;
			log2hi[n] = log2(hiprec_real(a));
			log2lo[n] = log2(hiprec_real(a)) - hiprec_real(log2hi[n]);
			lomax = std::max(lomax, (float) fabs(log2lo[n]));
		}
		init = true;
	}
	int i = (int&) x;
	int xi = (i >> 23) - 127;
	int index = (i & 0x7FF000) >> (23 - Nlogbits);
	//printf("%i\n", index);
	int j = (i & 0x7FF000) | (127 << 23);
	int k = (i & 0x7FFFFF) | (127 << 23);
	float a = (float&) j;
	float b = (float&) k;
	float eps = (b - a) / a;
	float z = -0.25f;
	z = fmaf(z, eps, 0.5f);
	z *= eps;
	float z2 = z * z;
	//= (4.121985831e-01);
	float log1peps = 2.885390082e+00;
	log1peps *= z;
	float loghi = log2hi[index];
	float loglo = log2lo[index];
	//loghi += log1peps;
	float arg1 = y * loghi;
	float arg1err = fmaf(y, loghi, -arg1);
	float arg2 = y * loglo;
	float pwr = exp2f(y);
	float theta = 1.f;
	float err = 0.0;
	for (int i = 0; i < 7; i++) {
		if (xi & 1) {
			theta *= pwr;
		}
		pwr *= pwr;
		xi >>= 1;
	}
//	printf( "%e %e %e \n", log2hi[index],log2lo[index], log1peps);
	//theta += err;
	x = logf(2) * (y * log1peps + (arg2 + arg1err));
	double p = 1.0f / 6.0f;
	p = fmaf(p, x, 0.5f);
	p = fmaf(p, x, 1.0f);
	p = fmaf(p, x, 1.0f);
	z = exp2f(arg1) * theta * p;
	return z;
}

struct double2 {
	double x;
	double y;
	double2 operator*(double2 other) const {
		double2 res;
		res.x = x * other.x;
		res.y = y * other.y + fma(x, other.x, -res.x);
		res.y += x * other.y + y * other.x;
		return res;
	}
};
struct double3 {
	double x, y, z;
};

double_2 sqr(double x) {
	double_2 p;
	p.x = x * x;
	p.y = fma(x, x, -p.x);
	return p;
}

double_2 sqrt(double_2 A) {
	double xn = 1.0 / sqrt(A.x);
	double yn = A.x * xn;
	double_2 ynsqr = sqr(yn);
	double diff = (A - ynsqr).x;
	double_2 prod;
	prod.x = xn * diff;
	prod.y = fma(xn, diff, -prod.x);
	prod.x *= 0.5;
	prod.y *= 0.5;
	prod.x += yn;
	return double_2::quick_two_sum(prod.x, prod.y);
}

std::vector<double> generate_gamma(int N, int M) {
	constexpr double toler = 0.5 * std::numeric_limits<double>::epsilon();
	hiprec_real a = hiprec_real(1);
	hiprec_real b = hiprec_real(2);
	std::function<hiprec_real(hiprec_real)> func = [a,b](hiprec_real x) {
		const auto sum = a + b;
		const auto dif = b - a;
		const auto half = hiprec_real(0.5);
		x = half*(sum + dif * x);
		return gamma(x);
	};
	std::vector<hiprec_real> co = ChebyCoeffs(func, toler, 0);
	co.resize(co.size() + N - 1, 0.0);
	for (int m = 0; m < co.size(); m++) {
		printf("%e\n", (double) co[m]);
		co[m] = pow(hiprec_real(2), hiprec_real(m)) * co[m];
	}
	std::valarray<hiprec_real> A(co.data(), co.size());
	auto B = A;
	for (int n = 0; n < N - 1; n++) {
		auto num = (hiprec_real(n + 1) + hiprec_real(1) / hiprec_real(2));
		for (int m = 0; m < B.size(); m++) {
			if (m > 0) {
				B[m] = num * A[m] + A[m - 1];
			} else {
				B[m] = num * A[m];
			}
		}
		A = B;
	}
	co.resize(0);
	for (int i = 0; i < std::min((int) A.size(), M); i++) {
		co.push_back(A[i]);
	}
	co.resize(M, 0.0);
	std::vector<double> res(co.size());
	printf("\n");
	for (int i = 0; i < co.size(); i++) {
		res[i] = co[i];
		printf("%e\n", res[i] / res[0]);
	}
	return res;
}

hiprec_real loggamma_deriv(int n, hiprec_real z) {
	if (n == 0) {
		return hiprec_real(1) / (gamma(z));
	} else {
		hiprec_real z0 = z;
		hiprec_real z1 = z0 * hiprec_real(1.0 + 10.0 * std::numeric_limits<float>::epsilon());
		hiprec_real dz = z1 - z0;
		z0 = loggamma_deriv(n - 1, z0);
		z1 = loggamma_deriv(n - 1, z1);
		return (z1 - z0) / dz;
	}
}
;
double factorial2(int n) {
	if (n < 0) {
		printf("%i\n", n);
		return 0.0;
	}
	if (n == 0) {
		return 1.0;
	}
	return factorial2(n - 1) * n;
}
;

double binomial(int n, int k) {
	if (n < k) {
		n++;
		return (n - k) / (double) n * binomial(n, k);
	}
	return factorial2(n) / factorial2(k) / factorial2(n - k);
}
;

double lgamma_root(double x0) {
	double max = x0 + double(0.25);
	double min = x0 - double(0.25);
	min = nextafter(min, x0);
	max = nextafter(max, x0);
	double err;
	do {
		double mid = (max + min) * double(0.5L);
		double fmid = lgamma(mid);
		double fmax = lgamma(max);
		if (fmid * fmax > double(0)) {
			max = mid;
		} else {
			min = mid;
		}
		if (max == nextafter(min, max) || max == min) {
			return mid;
		}
	} while (1);
	return 0.0;
}

std::vector<double> gammainv_coeffs(int N, hiprec_real x) {
	using ptype = std::vector<int>;
	struct dtype {
		ptype poly;
		hiprec_real c0;
	};
	std::vector<double> res;
	const auto derivative = [N](const std::vector<dtype>& d0) {
		std::vector<dtype> d1;
		for( int n = 0; n < d0.size(); n++) {
			dtype d;
			d = d0[n];
			d.c0 *= -1;
			d.poly[0]++;
			bool found = false;
			for( int p = 0; p < d1.size(); p++) {
				if( d1[p].poly == d.poly) {
					d1[p].c0 += d.c0;
					found = true;
					break;
				}
			}
			if( !found ) {
				d1.push_back(d);
			}
			for( int m = 0; m < N; m++) {
				if( d0[n].poly[m]) {
					dtype d;
					d = d0[n];
					d.c0 *= d.poly[m];
					d.poly[m]--;
					d.poly[m+1]++;
					bool found = false;
					for( int p = 0; p < d1.size(); p++) {
						if( d1[p].poly == d.poly) {
							d1[p].c0 += d.c0;
							found = true;
							break;
						}
					}
					if( !found ) {
						d1.push_back(d);
					}
				}
			}
		}
		return d1;
	};
	std::vector<std::vector<dtype>> deriv(N);
	std::vector<hiprec_real> dpoly(N - 1);
	dtype d;
	d.poly.resize(N, 0);
	d.c0 = 1;
	deriv[0].push_back(d);
	for (int n = 1; n < N; n++) {
		deriv[n] = derivative(deriv[n - 1]);
		dpoly[n - 1] = polygamma(n - 1, x);
	}
	const auto evaluate = [&dpoly, x, N](const std::vector<dtype>& ds) {
		hiprec_real sum = 0.0;
		hiprec_real gam = gamma(hiprec_real(x));
		hiprec_real gaminv;
		int xi = round((double) x);
		if( (double) x == xi && xi <= 0 ) {
			gaminv = 0.0;
		} else {
			gaminv = hiprec_real(1) / gam;
		}
		for( const auto& d : ds) {
			hiprec_real term = hiprec_real(1);
			for( int n = 0; n < N - 1; n++) {
				for( int m = 0; m < d.poly[n]; m++) {
					term *= dpoly[n];
				}
			}
			term *= d.c0;
			term *= gaminv;
			sum += term;
		}
		return sum;
	};
	for (int n = 0; n < N; n++) {
		auto c0 = evaluate(deriv[n]);
		c0 /= factorial(n);
		res.push_back(c0);
	}
	return res;
}

float tgamma_test(float x_) {
	using namespace simd;
	static bool init = false;
	constexpr int NCHEBY = 11;
	constexpr int Ntot = NCHEBY + 1;
	constexpr int M = 19;
	constexpr int Mstr = 14;
	constexpr int Msin = 10;
	static float coeffs[M][Ntot];
	static float sincoeffs[Msin];
	static float einvhi, einvlo;
	if (!init) {
		init = true;
		einvhi = exp(-hiprec_real(1));
		einvlo = exp(-hiprec_real(1)) - hiprec_real(einvhi);
		std::function<hiprec_real(hiprec_real)> func = [](hiprec_real x) {
			const auto sum = 0;
			const auto dif = 1;
			const auto half = hiprec_real(0.5);
			static const hiprec_real pi = hiprec_real(4) * atan(hiprec_real(1));
			x = half*x;
			if( x == hiprec_real(0.0)) {
				return hiprec_real(0);
			} else {
				return (sin(pi * x ) / (pi));
			}
		};
		auto chebies = ChebyCoeffs2(func, 2 * Msin + 1, -1);
		chebies.resize(2 * Msin, 0.0);
		for (int i = 0; i < 2 * Msin - 1; i += 2) {
			sincoeffs[i / 2] = (float) chebies[i + 1];
		}
		static hiprec_real A[2 * M];
		A[0] = 0;
		A[1] = hiprec_real(1);
		A[2] = -digamma(hiprec_real(1));
		for (int k = 3; k < M; k++) {
			hiprec_real sum = -digamma(hiprec_real(1)) * A[k - 1];
			for (int n = 2; n < k; n++) {
				auto sgn = pow(hiprec_real(-1), hiprec_real(n + 1));
				sum += sgn * zeta(n) * A[k - n];
			}
			sum /= hiprec_real(k - 1);
			A[k] = sum;
		}
		for (int n = 0; n < M; n++) {
			coeffs[n][0] = A[n];
		}
		for (int n = 1; n < NCHEBY; n++) {
			auto co = gammainv_coeffs(M, n);
			for (int m = 0; m < co.size(); m++) {
				coeffs[m][n] = co[m];
			}
		}
		A[0] = hiprec_real(0.5) * sqrt(hiprec_real(2));
		for (int n = 1; n < 2 * M; n++) {
			hiprec_real sum = 0.0;
			for (int k = 1; k < n; k++) {
				sum += A[k] * A[n - k] / hiprec_real(k + 1);
			}
			sum = 1.0 / n * A[n - 1] - sum;
			sum /= A[0] * (hiprec_real(1) + hiprec_real(1) / (hiprec_real(n) + hiprec_real(1)));
			A[n] = sum;
		}
		for (int n = 0; n < Mstr; n++) {
			auto pi = hiprec_real(4) * atan(hiprec_real(1));
			auto scale = hiprec_real(2) * sqrt(pi * exp(hiprec_real(-1)));
			coeffs[n][Ntot - 1] = (A[2 * n] * scale * gamma(hiprec_real(n) + hiprec_real(0.5)) / gamma(hiprec_real(0.5)));
		}
		for (int n = Mstr; n < M; n++) {
			coeffs[n][Ntot - 1] = 0.0;
		}
	}
	float_2 Einv;
	Einv.x = einvhi;
	Einv.y = einvlo;
	float x = x_;
	float y, z, x0, c, r, sgn, x2;
	int ic, asym, neg;
	float_2 A;
	x0 = x;
	neg = (x <= float(-0.5));
	x = neg ? -x : x;
	asym = x > float(8.5);
	x2 = round(x);
	ic = asym ? float(Ntot - 1) : x2 ;
	z = asym ? float(1) / x : x - x2;
	y = coeffs[M - 1][ic];
	for (int m = M - 2; m >= 0; m--) {
		y = fma(y, z, coeffs[m][ic]);
	}
	A = x;
	A = A * Einv;
	c = x - float(0.5);
	x2 = pow(A.x, c);
	x2 *= (float(1) + c * A.y / A.x);
	y = asym ? y * x2 : float(1) / y;
	r = x0 - floor(x0);
	r = r > float(0.5) ? float(1) - r : r;
	sgn = int(floor(x0)) & int(1) ? float(1) : float(-1);
	x2 = float(4) * r * r;
	z = float(sincoeffs[Msin - 1]);
	for (int m = Msin - 2; m >= 0; m--) {
		z = fma(z, x2, float(sincoeffs[m]));
	}
	z *= float(2) * r;
	y = neg ? sgn / (y * z * x0) : y;
	return y;
}

simd::simd_f64 lgamma_test(simd::simd_f64 x) {
	using namespace simd;
	static bool init = false;
	constexpr int NCHEBY = 12;
	static constexpr int NROOTS = 26;
	constexpr int Ntot = NROOTS + NCHEBY + 1;
	constexpr int M = 21;
	constexpr int Msin = 20;
	constexpr int Mlog = 10;
	static double coeffs[M][Ntot];
	static double bias[Ntot];
	static double Xc[Ntot];
	static double factor = 10.0;
	static double rootlocs[NROOTS];
	static double rootbegin[NROOTS];
	static double rootend[NROOTS];
	static double logsincoeffs[Msin];
	if (!init) {
		init = true;
		double x0 = 2.0;
		int n = 0;
		while (x0 > -15) {
			double xrt = lgamma_root(x0);
			if (n == 0) {
				xrt = 2.0;
			} else if (n == 1) {
				xrt = 1.0;
			}
			//	printf("%.16e\n", xrt);
			rootlocs[n] = xrt;
			Xc[n + NCHEBY] = xrt;
			double x1 = round(x0);
			auto co = gammainv_coeffs(M, xrt);
			const double eps = 0.5 * std::numeric_limits<double>::epsilon();
			double span1 = std::min(pow(eps * fabs(co[0] / co.back()), 1.0 / (co.size() - 1)), 0.5);
			double a = xrt < -0.5 ? std::min(((fabs(xrt)) / 5.5), 0.80) : 1.0;
			double span2 = (0.5 + a * 0.5) * fabs(xrt - round(xrt));
			span2 = nextafter(span2, 0.0);
			if (n % 2 == 0) {
				rootbegin[n] = xrt - span1;
				rootend[n] = xrt + span2;
			} else {
				rootbegin[n] = xrt - span2;
				rootend[n] = xrt + span1;
			}
			if (x0 == 2.0) {
				x0 = 1.0;
			} else if (x0 == 1.0) {
				x0 = -2.25;
			} else {
				x0 -= 0.5;
			}
			for (int m = 0; m < M; m++) {
				coeffs[m][n + NCHEBY] = co[m];
			}
			if (rootend[n] > rootbegin[n - 1]) {
				auto avg = 0.5 * (Xc[n + NCHEBY] + Xc[n + NCHEBY - 1]);
				rootend[n] = rootbegin[n - 1] = avg;
			}
			//		printf( "%e %e %e\n", rootbegin[n], xrt, rootend[n]);
			n++;
		}
		std::function<hiprec_real(hiprec_real)> func = [](hiprec_real x) {
			const auto sum = 0;
			const auto dif = 1;
			const auto half = hiprec_real(0.5);
			static const hiprec_real pi = hiprec_real(4) * atan(hiprec_real(1));
			x = half*x;
			if( x == hiprec_real(0.0)) {
				return hiprec_real(0);
			} else {
				return log(sin(pi * x ) / (pi * x));
			}
		};
		auto chebies = ChebyCoeffs(func, std::numeric_limits<double>::epsilon() * 0.5, 1);
		chebies.resize(2 * Msin, 0.0);
		for (int i = 0; i < 2 * Msin; i += 2) {
			logsincoeffs[i / 2] = (double) (chebies[i] * pow(hiprec_real(2), hiprec_real(i)));
		}
		static hiprec_real A[2 * M];
		A[0] = hiprec_real(0.5) * sqrt(hiprec_real(2));
		for (int n = 1; n < 2 * M; n++) {
			hiprec_real sum = 0.0;
			for (int k = 1; k < n; k++) {
				sum += A[k] * A[n - k] / hiprec_real(k + 1);
			}
			sum = 1.0 / n * A[n - 1] - sum;
			sum /= A[0] * (hiprec_real(1) + hiprec_real(1) / (hiprec_real(n) + hiprec_real(1)));
			A[n] = sum;
		}
		for (int n = 0; n < M; n++) {
			coeffs[n][Ntot - 1] = (A[2 * n] * sqrt(hiprec_real(2)) * gamma(hiprec_real(n) + hiprec_real(0.5)) / gamma(hiprec_real(0.5)));
		}
		for (int n = 0; n < NCHEBY; n++) {
			hiprec_real a = -hiprec_real(3.5) + hiprec_real(n);
			hiprec_real b = -hiprec_real(3.5) + hiprec_real(n + 1);
			hiprec_real xc = hiprec_real(0.5) * (a + b);
			hiprec_real span = (b - a) * hiprec_real(0.5);
			Xc[n] = xc;
			std::function<hiprec_real(hiprec_real)> func = [a,b](hiprec_real x) {
				const auto sum = a + b;
				const auto dif = b - a;
				const auto half = hiprec_real(0.5);
				x = half*(sum + dif * x);
				return hiprec_real(1) / gamma(x);
			};
			double norm = 1;
			auto chebies = ChebyCoeffs2(func, M + 1, 0);
			chebies.resize(M, 0.0);
			for (int m = 0; m < chebies.size(); m++) {
				coeffs[m][n] = chebies[m] * pow(span, hiprec_real(-m));
			}
		}
		for (int nn = 0; nn < Ntot - 1; nn++) {
			if (fabs(coeffs[0][nn]) < 1e-3) {
				coeffs[0][nn] = 0.0;
				bias[nn] = 0.0;
			} else if (fabs(coeffs[0][nn] + 1.0) < 0.1) {
				bias[nn] = -1.0;
				double y = log(hiprec_real(1) / -gamma(hiprec_real(Xc[nn])));
				y = exp(hiprec_real(y)) - hiprec_real(1);
				coeffs[0][nn] = -y;
			} else if (fabs(coeffs[0][nn] - 1.0) < 0.1) {
				bias[nn] = 1.0;
				double y = log(hiprec_real(1) / gamma(hiprec_real(Xc[nn])));
				y = exp(hiprec_real(y)) - hiprec_real(1);
				coeffs[0][nn] = y;
			} else {
				bias[nn] = 0.0;
			}
		}
		bias[Ntot - 1] = 0.0;

	}
	const auto logxor1px = [](simd_f64 x, simd_f64 xm1) {
		simd_f64 x1, z, z2, y, x0;
		simd_i64 k, j;
		x0 = x * simd_f64(M_SQRT2);
		frexp(x0, &j);
		x1 = simd_f64(2) * frexp(x, &k);
		j = j - simd_f64(1);
		k = k - simd_f64(1);
		k -= j;
		x1 = ldexp(x1, k);
		z = blend( (x1 - simd_f64(1)) / (x1 + simd_f64(1)), xm1 / (xm1 + simd_f64(2)), j == simd_i64(0));
		z2 = z * z;
		y = simd_f64(2.0 / (1.0 + 2.0 *(Mlog-1)));
		for( int n = Mlog - 2; n >= 0; n--) {
			y = fma(y, z2, simd_f64(2.0 / (1.0 + 2.0 * n)));
		}
		y *= z;
		static const simd_f64 log2(log(2));
		y += simd_f64(j) * log2;
		return y;
	};
	simd_f64 y, z, x0, zm1, logx, r, b, c, x2;
	simd_i64 ic, nearroot, nearroot1, nearroot2, asym, neg, yneg;
	simd_f64_2 Y;
	x0 = x;
	ic = simd_i64(fmin(simd_f64(2) * fmax(floor(-x - simd_f64(1)), simd_f64(0)), simd_f64(NROOTS - 2)));
	nearroot1 = (x > c.gather(rootbegin, ic) && x < c.gather(rootend, ic));
	nearroot2 = (x > c.gather(rootbegin, ic + simd_i64(1)) && x < c.gather(rootend, ic + simd_i64(1)));
	nearroot = nearroot1 || nearroot2;
	neg = !nearroot && (x <= simd_f64(-3.5));
	x = blend(x, -x, neg);
	asym = !nearroot && (x >= simd_f64(7.5));
	ic = blend(blend(simd_i64(round(x)) + simd_i64(3), simd_i64(Ntot - 1), asym), ic + simd_i64(NCHEBY) + nearroot2, nearroot);
	z = blend(x - c.gather(Xc, ic), simd_f64(1) / x, asym);
	y = c.gather(coeffs[M - 1], ic);
	for (int m = M - 2; m >= 0; m--) {
		y = fma(y, z, c.gather(coeffs[m], ic));
	}
	logx = log(x);
	y = blend(y, simd_f64(1) / y, asym);
	b.gather(bias, ic);
	yneg = b + y < simd_f64(0);
	z = blend(b + y, -y - b, yneg);
	zm1 = blend(y - (simd_f64(1) - b), -y - (simd_f64(1) + b), yneg);
	y = -logxor1px(z, zm1);
	y += blend(simd_f64(0), -x + (x - simd_f64(0.5)) * logx + simd_f64(0.5 * log(2.0 * M_PI)), asym);
	r = x0 - floor(x0);
	r = blend(r, simd_f64(1) - r, r > simd_f64(0.5));
	x2 = r * r;
	z = simd_f64(logsincoeffs[Msin - 1]);
	for (int m = Msin - 2; m >= 0; m--) {
		z = fma(z, x2, simd_f64(logsincoeffs[m]));
	}
	Y = simd_f64_2::quick_two_sum(log(abs(r)), z);
	Y = Y + simd_f64_2::two_sum(y, logx);
	y = blend(y, -Y.x, neg);
	return y;
}

std::vector<hiprec_real> cot_derivs(int N, hiprec_real x) {
	std::vector<hiprec_real> res(N);
	struct term_t {
		hiprec_real cot;
		hiprec_real csc;
		hiprec_real c0;
	};
	const auto derivative = [](const std::vector<term_t>& d0s ) {
		std::vector<term_t> d1;
		for( auto d0 : d0s) {
			term_t d;
			d = d0;
			d.c0 *= -d.csc;
			d.cot = d.cot + hiprec_real(1);
			d1.push_back(d);
			d = d0;
			d.c0 *= -d.cot;
			d.cot =d.cot - hiprec_real(1);
			d.csc=d.csc + hiprec_real(2);
			d1.push_back(d);
		}
		std::vector<term_t> D;
		for( int i = 0; i < d1.size(); i++) {
			term_t d = d1[i];
			for( int j = i + 1; j < d1.size(); j++) {
				if( d1[j].cot == d1[i].cot && d1[j].csc == d1[i].csc) {
					d.c0 += d1[j].c0;
					d1[j] = d1.back();
					d1.pop_back();
					j--;
				}
			}
			D.push_back(d);
		}
		return D;
	};
	const auto evaluate = [](const std::vector<term_t>& D, hiprec_real x) {
		hiprec_real sum = 0.0;
		for( auto d : D ) {
			hiprec_real term = 1.0;
			hiprec_real sine = sin(x);
			hiprec_real csc = hiprec_real(1) / sine;
			hiprec_real cot = cos(x) * csc;
			for( int k = 0; k < d.cot; k++) {
				term *= cot;
			}
			for( int k = 0; k < d.csc; k++) {
				term *= csc;
			}
			term *= d.c0;
			sum += term;
		}
		return sum;
	};
	std::vector<std::vector<term_t>> derivs(N);
	term_t d0;
	d0.cot = 1.0;
	d0.csc = 0.0;
	d0.c0 = 1.0;
	derivs[0].push_back(d0);
	for (int n = 1; n < N; n++) {
		derivs[n] = derivative(derivs[n - 1]);
	}
	for (int n = 0; n < N; n++) {
		res[n] = evaluate(derivs[n], x);
	}
	return res;
}

std::vector<double> logsin_expansion(int N) {
	hiprec_real x = .5;
	int M = 4 * N;
	auto derivs = cot_derivs(M, 0.5);
	std::vector<hiprec_real> C(N, 0.0);
	std::vector<hiprec_real> A(M);
	hiprec_real pi = hiprec_real(4) * atan(hiprec_real(1));
	A[0] = log(sin(x) / x);
	A[1] = derivs[0] - hiprec_real(1) / x;
	for (int n = 2; n < M; n++) {
		A[n] = derivs[n - 1];
		A[n] += pow(hiprec_real(-1), hiprec_real(n)) * factorial(n - 1) * pow(x, hiprec_real(-n));
		A[n] /= factorial(n);
	}
	for (int n = 0; n < N; n++) {
		C[n] = 0.0;
		for (int k = n; k < M; k++) {
			C[n] += A[k] * factorial(k) / factorial(n) / factorial(k - n) * pow(-x, hiprec_real(k - n));
		}
	}
	std::vector<double> res;
	for (int n = 0; n < N; n++) {
		res.push_back(C[n]);
	}
	return res;
}

int main() {
	using namespace simd;
//	gammainv_coeffs(20, 16.0);
//return 0;
	srand(time(NULL));
//	TEST1(float, simd_f32, tgamma, tgamma, tgamma, -33, 33.000, true);
//	TEST2(float, simd_f32, pow, powf, pow, 1e-3, 1e3, .01, 10, true);
//TEST1(double, simd_f64, log2, log2, log2_precise, .0001, 100000, true);
//	TEST1(double, simd_f64, exp, exp, exp, -600.0, 600.0, true);
//	TEST1(double, simd_f64, tgamma, tgammal, tgamma_test, 0, 2.000, true);


//	TEST2(float, simd_f32, pow, pow, pow, .1, 10, -30, 30, true);

	printf("Testing SIMD Functions with N = 1 << %d = %d\n", N_bit_shift, 1 << N_bit_shift);
	printf("\nSingle Precision\n");
	printf("name   speed        avg err      max err\n");

	TEST1(float, simd_f32, asin, asinf, asin, -1, 1, true);
	TEST1(float, simd_f32, acos, acosf, acos, -1, 1, true);
	TEST1(float, simd_f32, atan, atanf, atan, -10.0, 10.0, true);
	TEST1(float, simd_f32, acosh, acoshf, acosh, 1.001, 10.0, true);
	TEST1(float, simd_f32, asinh, asinhf, asinh, .001, 10, true);
	TEST1(float, simd_f32, atanh, atanhf, atanh, 0.001, 0.999, true);
	TEST1(float, simd_f32, exp, expf, exp, -86.0, 86.0, true);
	TEST1(float, simd_f32, exp2, exp2f, exp2, -125.0, 125.0, true);
	TEST1(float, simd_f32, expm1, expm1f, expm1, -2.0, 2.0, true);
	TEST1(float, simd_f32, log, logf, log, exp(-1), exp(40), true);
	TEST1(float, simd_f32, log2, log2f, log2, 0.00001, 100000, true);
	TEST1(float, simd_f32, log1p, log1pf, log1p, exp(-3), exp(3), true);
	TEST1(float, simd_f32, erf, erff, erf, -7, 7, true);
	TEST1(float, simd_f32, erfc, erfcf, erfc, -8.9, 8.9, true);
	TEST1(float, simd_f32, tgamma, tgammaf, tgamma, -33, 33.0, true);
	TEST1(float, simd_f32, cosh, coshf, cosh, -10.0, 10.0, true);
	TEST1(float, simd_f32, sinh, sinhf, sinh, -10.0, 10.0, true);
	TEST1(float, simd_f32, tanh, tanhf, tanh, -10.0, 10.0, true);
	TEST1(float, simd_f32, sin, sinf, sin, -2 * M_PI, 2 * M_PI, true);
	TEST1(float, simd_f32, cos, cosf, cos, -2 * M_PI, 2 * M_PI, true);
	TEST1(float, simd_f32, tan, tanf, tan, -2 * M_PI, 2 * M_PI, true);
	printf("\nDouble Precision\n");
	printf("name   speed        avg err      max err\n");
	TEST1(double, simd_f64, asin, asin, asin, -1, 1, true);
	TEST1(double, simd_f64, acos, acos, acos, -1 + 1e-6, 1 - 1e-6, true);
	TEST1(double, simd_f64, atan, atan, atan, -10.0, 10.0, true);
	TEST1(double, simd_f64, acosh, acosh, acosh, 1.001, 10.0, true);
	TEST1(double, simd_f64, asinh, asinh, asinh, .001, 10, true);
	TEST1(double, simd_f64, atanh, atanh, atanh, 0.001, 0.999, true);
	TEST2(double, simd_f64, pow, pow, pow, .1, 10, -300, 300, true);
	TEST1(double, simd_f64, exp, exp, exp, -600.0, 600.0, true);
	TEST1(double, simd_f64, exp2, exp2, exp2, -1000.0, 1000.0, true);
	TEST1(double, simd_f64, expm1, expm1, expm1, -2.0, 2.0, true);
	TEST1(double, simd_f64, log, log, log, exp(-1), exp(40), true);
	TEST1(double, simd_f64, log2, log2, log2, .0001, 100000, true);
	TEST1(double, simd_f64, log1p, log1p, log1p, exp(-3), exp(3), true);
	TEST1(double, simd_f64, erf, erf, erf, -9, 9, true);
	TEST1(double, simd_f64, erfc, erfc, erfc, -25.0, 25.0, true);
	TEST1(double, simd_f64, cosh, cosh, cosh, -10.0, 10.0, true);
	TEST1(double, simd_f64, sinh, sinh, sinh, -10.0, 10.0, true);
	TEST1(double, simd_f64, tanh, tanh, tanh, -10.0, 10.0, true);
	TEST1(double, simd_f64, sin, sin, sin, -2.0 * M_PI, 2.0 * M_PI, true);
	TEST1(double, simd_f64, cos, cos, cos, -2.0 * M_PI, 2.0 * M_PI, true);
	TEST1(double, simd_f64, tan, tan, tan, -2.0 * M_PI, 2.0 * M_PI, true);

	/*

	 TEST1(float, simd_f32, cbrt, cbrtf, cbrt, 1.0 / 4000, 4000, true);
	 TEST1(float, simd_f32, sqrt, sqrtf, sqrt, 0, std::numeric_limits<int>::max(), true);
	 TEST1(float, simd_f32, cvt, cvt32_ref, cvt32_test, 1, +10000000, true);*/

	/*	TEST1(double, simd_f64, exp2, exp2, exp2, -1000.0, 1000.0, true);
	 TEST1(double, simd_f64, cbrt, cbrt, cbrt, 1.0 / 4000, 4000, true);
	 TEST1(double, simd_f64, sqrt, sqrt, sqrt, 0, std::numeric_limits<long long>::max(), true);
	 TEST1(double, simd_f64, cvt, cvt64_ref, cvt64_test, 1LL, +1000000000LL, true);*/

	return (0);
}

