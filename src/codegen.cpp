#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "../include/simd.hpp"
#include "../include/hiprec.hpp"
#include "../include/polynomial.hpp"

#define SYSTEM(...) if( system(__VA_ARGS__) != 0 ) {fprintf(stderr, "SYSTEM error %s %i\n", __FILE__, __LINE__); abort(); }

template<class ...Args>
std::string print2str(const char* fstr, Args&&...args) {
	std::string result;
	char* str;
	if (!asprintf(&str, fstr, std::forward<Args>(args)...)) {
		fprintf(stderr, "Error in %s on line %i\n", __FILE__, __LINE__);
		abort();
	}
	result = str;
	free(str);
	return result;
}

hiprec_real factorial(int n) {
	if (n <= 1) {
		return hiprec_real(1);
	} else {
		return hiprec_real(n) * factorial(n - 1);
	}
}

/* Inputs are resolved relative to SIMD_ROOT_DIR, which CMake sets to this
   project's source directory. The default preserves the standalone build,
   where the codegen runs from a build directory one level below the root. */
#ifndef SIMD_ROOT_DIR
#define SIMD_ROOT_DIR ".."
#endif

void include(FILE* fp, std::string filename) {
	constexpr int N = 1024;
	char buffer[N];
	FILE* fp0 = fopen(filename.c_str(), "rt");
	if (!fp0) {
		fprintf(stderr, "Unable to open %s\n", filename.c_str());
		abort();
	}
	while (!feof(fp0)) {
		if (!fgets(buffer, N, fp0)) {
			break;
		}
		fprintf(fp, "%s", buffer);
	}
	fclose(fp0);
}

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

void float_funcs(FILE* fp) {
	include(fp, std::string(SIMD_ROOT_DIR) + "/include/code.hpp");

	/* cos */
	{
		constexpr int N = 6;
		float coeff[N];
		hiprec_real pi_exact = hiprec_real(4) * atan(hiprec_real(1));
		float pi1 = pi_exact;
		float pi2 = pi_exact - hiprec_real(pi1);
		hiprec_real fac(1);
		for (int n = 0; n < N; n++) {
			coeff[n] = hiprec_real(1) / fac;
			fac *= hiprec_real(2 * n + 2);
			fac *= -hiprec_real(2 * n + 3);
		}
		fprintf(fp, "\nsimd_f32 cos(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 x0, s, x2, y;\n");
		fprintf(fp, "\tx = abs(x);\n");
		fprintf(fp, "\tx0 = floor(x * simd_f32(%.9e));\n", 1.0 / M_PI);
		fprintf(fp, "\ts = simd_f32(2) * (x0 - simd_f32(2) * floor(x0 * simd_f32(0.5))) - simd_f32(1);\n");
		fprintf(fp, "\tx0 += simd_f32(0.5);\n");
		fprintf(fp, "\tx = s * fma(-x0, simd_f32(%.9e), fma(-x0, simd_f32(%.9e), x));\n", pi2, pi1);
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\ty = simd_f32(%.9e);\n", coeff[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x2, y, simd_f32(%.9e));\n", coeff[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	/* sin */
	{
		constexpr int N = 6;
		float coeff[N];
		hiprec_real pi_exact = hiprec_real(4) * atan(hiprec_real(1));
		float pi1 = pi_exact;
		float pi2 = pi_exact - hiprec_real(pi1);
		hiprec_real fac(1);
		for (int n = 0; n < N; n++) {
			coeff[n] = hiprec_real(1) / fac;
			fac *= hiprec_real(2 * n + 2);
			fac *= -hiprec_real(2 * n + 3);
		}
		fprintf(fp, "\nsimd_f32 sin(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 x0, s, x2, y;\n");
		fprintf(fp, "\tx0 = round(x * simd_f32(%.9e));\n", 1.0 / M_PI);
		fprintf(fp, "\ts = -simd_f32(2) * (x0 - simd_f32(2) * floor(x0 * simd_f32(0.5))) + simd_f32(1);\n");
		fprintf(fp, "\tx = s * fma(-x0, simd_f32(%.9e), fma(-x0, simd_f32(%.9e), x));\n", pi2, pi1);
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\ty = simd_f32(%.9e);\n", coeff[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x2, y, simd_f32(%.9e));\n", coeff[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}

	/* erf */
	{
		constexpr int M = 3;
		static std::vector<hiprec_real> co[M];
		const hiprec_real xmax(4.1);
		static constexpr double toler = std::numeric_limits<float>::epsilon() * 0.5;
		int N = 0;
		for (int i = 0; i < M; i++) {
			hiprec_real a = hiprec_real(i) * xmax / hiprec_real(M);
			hiprec_real b = hiprec_real(i + 1) * xmax / hiprec_real(M);
			std::function<hiprec_real(hiprec_real)> func = [a,b](hiprec_real x) {
				const auto sum = a + b;
				const auto dif = b - a;
				const auto half = hiprec_real(0.5);
				return erf(half*(sum + dif * x));
			};
			co[i] = ChebyCoeffs(func, toler, 0);
			N = std::max(N, (int) co[i].size());
		}
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < M; m++) {
				if (co[m].size() < n + 1) {
					co[m].push_back(0.0);
				}
			}
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 erf(simd_f32 x) {\n");
		fprintf(fp, "\tstatic const simd_f32 co[] = {\n");
		for (int n = 0; n < N; n += 2) {
			fprintf(fp, "\t\t{");
			for (int i = n; i < n + 2; i++) {
				double c1, c2, c3, c4;
				if (i < N) {
					int m = i / 2;
					const auto pi = hiprec_real(4) * atan(hiprec_real(1));
					c1 = (i % 2 == 1) ? (double) (hiprec_real(2) / sqrt(pi) * hiprec_real(m % 2 == 0 ? 1 : -1) / factorial(m) / hiprec_real(2 * m + 1)) : 0.0;
					c2 = co[0][i];
					c3 = co[1][i];
					c4 = co[2][i];
					const double factor = (double) (pow(hiprec_real(2) * hiprec_real(M) / xmax, hiprec_real(i)));
//					c1 *= factor;// * ((1<<(i)) * 0.5);
					//	c1 *= 2.0;
					c2 *= factor;
					c3 *= factor;
					c4 *= factor;
				} else {
					c1 = c3 = c4 = c2 = 99.0;
				}
				fprintf(fp, "%.9e, %.9e, %.9e, %.9e", c1, c2, c3, c4);
				if (i != n + 1) {
					fprintf(fp, ",");
				}
				fprintf(fp, " ");
			}
			fprintf(fp, "}");
			if (n + 2 < N) {
				fprintf(fp, ",");
			}
			fprintf(fp, "\n");
		}
		auto sqrtpi = sqrt(hiprec_real(4) * atan(hiprec_real(1)));
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tsimd_f32 y, s, z, x2;\n");
		fprintf(fp, "\tsimd_i32 i0, i1, l;\n");
		fprintf(fp, "\ts = copysign(simd_f32(1), x);\n");
		fprintf(fp, "\tx = fabs(x);\n");
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\tl = x < simd_f32(0.15);\n");
		fprintf(fp, "\tz = simd_f32(1) - simd_f32(l);\n");
		fprintf(fp, "\tx = fmin(x, simd_f32(%.9e));\n", (double) xmax - 0.00001);
		fprintf(fp, "\ti0 = x * simd_f32(%.9e);\n", (double) (hiprec_real(M) / xmax));
		fprintf(fp, "\ti0 += simd_i32(1);\n");
		fprintf(fp, "\ti0 -= l;\n");
		fprintf(fp, "\ti1 = i0 + simd_i32(4);\n");
		fprintf(fp, "\tx -= z * simd_f32(%.9e) * i0;\n", (double) (xmax / hiprec_real(M)));
		fprintf(fp, "\tx -= z * simd_f32(%.9e);\n", (double) (xmax * hiprec_real(0.5) / hiprec_real(M)) - (double) (xmax / hiprec_real(M)));
//		fprintf(fp, "\tx *= simd_f32(%.9e);\n",);
//		fprintf(fp, "\tx = blend(x, x0, l);\n");
		fprintf(fp, "\ty = co[%i].permute(i%i);\n", (N - 1) / 2, (N - 1) % 2);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, co[%i].permute(i%i));\n", n / 2, n % 2);
		}
		fprintf(fp, "\treturn s * y;\n");
		fprintf(fp, "}\n");
	}

	/* tgamma */
	{
		static bool init = false;
		constexpr int NCHEBY = 11;
		constexpr int Ntot = NCHEBY + 1;
		constexpr int M = 19;
		constexpr int Mstr = 5;
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
		fprintf(fp, "\nsimd_f32 tgamma(simd_f32 x) {\n");
		fprintf(fp, "\tstatic const simd_f32_2 Einv(%.9ef, %.9ef);\n", einvhi, einvlo);
		fprintf(fp, "\tstatic constexpr float coeffs[][%i] = {\n", Ntot);
		for (int m = 0; m < M; m++) {
			fprintf(fp, "\t\t{");
			for (int n = 0; n < Ntot; n++) {
				fprintf(fp, "%17.9ef%s", coeffs[m][n], n != Ntot - 1 ? ", " : "");
			}
			fprintf(fp, "}%s\n", m != M - 1 ? "," : "");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tsimd_f32 y, z, x0, c, r, sgn, x2;\n"
				"\tsimd_i32 ic, asym, neg;\n"
				"\tsimd_f32_2 A;\n"
				"\tx0 = x;\n"
				"\tneg = (x <= simd_f32(-0.5));\n"
				"\tx = blend(x, -x, neg);\n"
				"\tasym = x > simd_f32(8.5);\n"
				"\tx2 = round(x);\n"
				"\tic = blend(x2, simd_f32(%i), asym);\n", Ntot - 1);
		fprintf(fp, "\tz = blend(x - x2, simd_f32(1) / x, asym);\n"
				"");
		fprintf(fp, "\ty.gather(coeffs[%i], ic);\n", M - 1);
		for (int m = M - 2; m >= 0; m--) {
			fprintf(fp, "\ty = fma(y, z, c.gather(coeffs[%i], ic));\n", m);
		}
		fprintf(fp, "\tA = x;\n"
				"\tA = A * Einv;\n"
				"\tc = x - simd_f32(0.5);\n"
				"\tx2 = pow(A.x, c);\n"
				"\tx2 *= (simd_f32(1) + c * A.y / A.x);\n"
				"\ty = blend(simd_f32(1) / y, y * x2, asym);\n"
				"\tr = x0 - floor(x0);\n"
				"\tr = blend(r, simd_f32(1) - r, r > simd_f32(0.5));\n"
				"\tsgn = blend(simd_f32(-1), simd_f32(1), simd_i32(floor(x0)) & simd_i32(1));\n"
				"\tx2 = simd_f32(4) * r * r;\n");
		fprintf(fp, "\tz = simd_f32(%.9e);\n", sincoeffs[Msin - 1]);
		for (int m = Msin - 2; m >= 0; m--) {
			fprintf(fp, "\tz = fma(z, x2, simd_f32(%.9e));\n", sincoeffs[m]);
		}
		fprintf(fp, "\tz *= simd_f32(2) * r;\n"
				"\ty = blend(y, sgn / (y * z * x0), neg);\n"
				"\treturn y;\n"
				"}\n");

	}


	/* lgamma */
	{
		static bool init = false;
		constexpr int NCHEBY = 12;
		static constexpr int NROOTS = 16;
		constexpr int Ntot = NROOTS + NCHEBY + 1;
		constexpr int M = 15;
		constexpr int Msin = 8;
		constexpr int Mlog = 4;
		static float coeffs[M][Ntot];
		static float bias[Ntot];
		static float Xc[Ntot];
		static float factor = 10.0;
		static float rootbegin[NROOTS];
		static float rootend[NROOTS];
		static float logsincoeffs[Msin];
		if (!init) {
			init = true;
			float x0 = 2.0;
			int n = 0;
			while (x0 > -9) {
				float xrt = lgamma_root(x0);
				if (n == 0) {
					xrt = 2.0;
				} else if (n == 1) {
					xrt = 1.0;
				}
				Xc[n + NCHEBY] = xrt;
				float x1 = round(x0);
				auto co = gammainv_coeffs(M, xrt);
				const float eps = 0.5 * std::numeric_limits<float>::epsilon();
				float span1 = std::min(pow(eps * fabs(co[0] / co.back()), 1.0 / (co.size() - 1)), 0.5);
				float a = xrt < -0.5 ? std::min(((fabs(xrt)) / 5.5), 0.80) : 1.0;
				float span2 = (0.5 + a * 0.5) * fabs(xrt - round(xrt));
				span2 = nextafterf(span2, 0.0);
				if (n == 0) {
					rootbegin[n] = 1.75;
					rootend[n] = 2.25;
				} else if (n == 1) {
					rootbegin[n] = 0.75;
					rootend[n] = 1.25;
				} else {
					if (n % 2 == 0) {
						rootbegin[n] = xrt - span1;
						rootend[n] = xrt + span2;
					} else {
						rootbegin[n] = xrt - span2;
						rootend[n] = xrt + span1;
					}
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
				if (x0 < 0.0) {
					if (rootend[n] > rootbegin[n - 1]) {
						auto avg = 0.5 * (Xc[n + NCHEBY] + Xc[n + NCHEBY - 1]);
						rootend[n] = rootbegin[n - 1] = avg;
					}
				}
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
			auto chebies = ChebyCoeffs(func, std::numeric_limits<float>::epsilon() * 0.5, 1);
			chebies.resize(2 * Msin, 0.0);
			for (int i = 0; i < 2 * Msin; i += 2) {
				logsincoeffs[i / 2] = (float) (chebies[i] * pow(hiprec_real(2), hiprec_real(i)));
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
				float norm = 1;
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
					float y = log(hiprec_real(1) / -gamma(hiprec_real(Xc[nn])));
					y = exp(hiprec_real(y)) - hiprec_real(1);
					coeffs[0][nn] = -y;
				} else if (fabs(coeffs[0][nn] - 1.0) < 0.1) {
					bias[nn] = 1.0;
					float y = log(hiprec_real(1) / gamma(hiprec_real(Xc[nn])));
					y = exp(hiprec_real(y)) - hiprec_real(1);
					coeffs[0][nn] = y;
				} else {
					bias[nn] = 0.0;
				}
			}
			bias[Ntot - 1] = 0.0;

		}
		fprintf(fp, "\n\nsimd_f32 lgamma(simd_f32 x) {\n");
		fprintf(fp, "\tstatic constexpr float coeffs[][%i] = {\n", Ntot);
		for (int m = 0; m < M; m++) {
			fprintf(fp, "\t\t{");
			for (int n = 0; n < Ntot; n++) {
				fprintf(fp, "%17.9e%s", coeffs[m][n], n != Ntot - 1 ? ", " : "");
			}
			fprintf(fp, "}%s\n", m != M - 1 ? "," : "");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tstatic constexpr float bias[] = {");
		for (int n = 0; n < Ntot; n++) {
			fprintf(fp, "%17.9e%s", bias[n], n != Ntot - 1 ? ", " : "");
		}
		fprintf(fp, "};\n");
		fprintf(fp, "\tstatic constexpr float Xc[] = {");
		for (int n = 0; n < Ntot; n++) {
			fprintf(fp, "%17.9e%s", Xc[n], n != Ntot - 1 ? ", " : "");
		}
		fprintf(fp, "};\n");
		fprintf(fp, "\tstatic constexpr float rootbegin[] = {");
		for (int n = 0; n < NROOTS; n++) {
			fprintf(fp, "%17.9e%s", rootbegin[n], n != NROOTS - 1 ? ", " : "");
		}
		fprintf(fp, "};\n");
		fprintf(fp, "\tstatic constexpr float rootend[] = {");
		for (int n = 0; n < NROOTS; n++) {
			fprintf(fp, "%17.9e%s", rootend[n], n != NROOTS - 1 ? ", " : "");
		}
		fprintf(fp, "};\n");
		fprintf(fp, "\tstatic const auto logxor1px = [](simd_f32 x, simd_f32 xm1) {\n"
				"\tsimd_f32 x1, z, z2, y, x0;\n"
				"\tsimd_i32 k, j;\n"
				"\tx0 = x * simd_f32(M_SQRT2);\n"
				"\tfrexp(x0, &j);\n"
				"\tx1 = simd_f32(2) * frexp(x, &k);\n"
				"\tj = j - simd_f32(1);\n"
				"\tk = k - simd_f32(1);\n"
				"\tk -= j;\n"
				"\tx1 = ldexp(x1, k);\n"
				"\tz = blend( (x1 - simd_f32(1)) / (x1 + simd_f32(1)), xm1 / (xm1 + simd_f32(2)), j == simd_i32(0));\n"
				"\tz2 = z * z;\n"
				"\ty = simd_f32(%.9e);\n", (float) (2.0L / (1.0L + 2.0L * (long double) (Mlog - 1))));
		for (int n = Mlog - 2; n >= 0; n--) {
			fprintf(fp, "\t\ty = fma(y, z2, simd_f32(%.9e));\n", (float) (2.0L / (1.0L + 2.0L * (long double) n)));
		}
		fprintf(fp, "\t\ty *= z;\n"
				"\ty += simd_f32(j) * simd_f32(%.9e);\n", (float) logl(2));
		fprintf(fp, "\t\treturn y;\n"
				"\t};\n"
				"");
		fprintf(fp, "\tsimd_f32 y, z, x0, zm1, logx, r, b, c, x2;\n"
				"\tsimd_i32 ic, nearroot, nearroot1, nearroot2, asym, neg, yneg;\n"
				"\tsimd_f32_2 Y;\n"
				"\tx0 = x;\n"
				"\tic = simd_i32(fmin(simd_f32(2) * fmax(floor(-x - simd_f32(1)), simd_f32(0)), simd_f32(%i)));\n", NROOTS - 2);
		fprintf(fp, "\tnearroot1 = (x > c.gather(rootbegin, ic) && x < c.gather(rootend, ic));\n"
				"\tnearroot2 = (x > c.gather(rootbegin, ic + simd_i32(1)) && x < c.gather(rootend, ic + simd_i32(1)));\n"
				"\tnearroot = nearroot1 || nearroot2;\n"
				"\tneg = !nearroot && (x <= simd_f32(-3.5));\n"
				"\tx = blend(x, -x, neg);\n"
				"\tasym = !nearroot && (x >= simd_f32(7.5));\n"
				"\tic = blend(blend(simd_i32(round(x)) + simd_i32(3), simd_i32(%i), asym), ic + simd_i32(%i) + nearroot2, nearroot);\n", Ntot - 1, NCHEBY);

		fprintf(fp, "\tz = blend(x - c.gather(Xc, ic), simd_f32(1) / x, asym);\n"
				"");
		fprintf(fp, "\ty = c.gather(coeffs[%i], ic);\n", M - 1);
		for (int m = M - 2; m >= 0; m--) {
			fprintf(fp, "\ty = fma(y, z, c.gather(coeffs[%i], ic));\n", m);
		}
		fprintf(fp, "\tlogx = log(x);\n"
				"\ty = blend(y, simd_f32(1) / y, asym);\n"
				"\tb.gather(bias, ic);\n"
				"\tyneg = b + y < simd_f32(0);\n"
				"\tz = blend(b + y, -y - b, yneg);\n"
				"\tzm1 = blend(y - (simd_f32(1) - b), -y - (simd_f32(1) + b), yneg);\n"
				"\ty = -logxor1px(z, zm1);\n"
				"\ty += blend(simd_f32(0), -x + (x - simd_f32(0.5)) * logx + simd_f32(%.9e), asym);\n", (float) (0.5L * log(2.0L * M_PIl)));
		fprintf(fp, "\tr = x0 - floor(x0);\n"
				"\tr = blend(r, simd_f32(1) - r, r > simd_f32(0.5));\n"
				"\tx2 = r * r;\n"
				"\tz = simd_f32(%.9e);\n", logsincoeffs[Msin - 1]);
		for (int m = Msin - 2; m >= 0; m--) {
			fprintf(fp, "\tz = fma(z, x2, simd_f32(%.9e));\n", logsincoeffs[m]);
		}
		fprintf(fp, "\tY = simd_f32_2::quick_two_sum(log(abs(r)), z);\n"
				"\tY = Y + simd_f32_2::two_sum(y, logx);\n"
				"\ty = blend(y, -Y.x, neg);\n"
				"\treturn y;\n");
		fprintf(fp, "}\n");
	}
	/* asin */
	{
		static std::vector<hiprec_real> co1;
		static std::vector<hiprec_real> co2;
		static const hiprec_real z0(0.5);
		static constexpr double toler = std::numeric_limits<float>::epsilon() * 0.5;
		auto func3 = [](hiprec_real y) {
			const auto x = hiprec_real(1) - y * y;
			const auto a = asin(x);
			static const auto b = hiprec_real(2) * atan(hiprec_real(1));
			const auto c = sqrt(hiprec_real(1) - x);
			return (b - a)/c;
		};
		auto func2 = [func3](hiprec_real x) {
			const static auto z1 = sqrt(hiprec_real(1) - z0);
			return func3((z1 + x * hiprec_real(z1)) / hiprec_real(2));
		};
		co2 = ChebyCoeffs(func2, toler, 0);
		do {
			int n = co1.size();
			co1.push_back(
					factorial(hiprec_real(2 * n))
							/ (pow(hiprec_real(2), hiprec_real(2 * n)) * factorial(hiprec_real(n)) * factorial(hiprec_real(n)) * (hiprec_real(2 * n + 1))));

		} while (std::abs(double(co1.back()) * pow(double(z0), 2 * co1.size() - 1)) > toler);
		auto tmp = co1;
		co1.resize(0);
		for (int i = 0; i < tmp.size(); i++) {
			co1.push_back(0.0);
			co1.push_back(tmp[i]);
		}
		int N = std::max(co1.size(), co2.size());
		for (int n = 0; n < N; n++) {
			if (co1.size() < n + 1) {
				co1.push_back(0.0);
			}
			if (co2.size() < n + 1) {
				co2.push_back(0.0);
			}
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 asin(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 y, s, z, x0, x1;\n");
		fprintf(fp, "\tsimd_i32 i0, i1, i2, i3;\n");
		fprintf(fp, "\tstatic const simd_f32 co[] = {\n");
		for (int n = 0; n < N; n += 4) {
			fprintf(fp, "\t\t{");
			for (int i = n; i < n + 4; i++) {
				double c1, c2;
				if (i < N) {
					c1 = co1[i];
					c2 = co2[i];
				} else {
					c1 = 99.0;
					c2 = 99.0;
				}
				fprintf(fp, "%.9e, %.9e", c1, c2);
				if (i != n + 3) {
					fprintf(fp, ",");
				}
				fprintf(fp, " ");
			}
			fprintf(fp, "}");
			if (n + 4 < N) {
				fprintf(fp, ",");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\ts = copysign(simd_f32(1), x);\n");
		fprintf(fp, "\tx = fabs(x);\n");
		fprintf(fp, "\ti0 = x > simd_f32(%.9e);\n", (double) z0);
		fprintf(fp, "\tz = sqrt(simd_f32(1) - x);\n");
		fprintf(fp, "\tx0 = x;\n");
		fprintf(fp, "\tx1 = simd_f32(%.9e) * z - simd_f32(1);\n", (double) (hiprec_real(2) / sqrt(hiprec_real(1) - z0)));
		fprintf(fp, "\tx = blend(x0, x1, i0);\n");
		fprintf(fp, "\ti1 = i0 + simd_i32(2);\n");
		fprintf(fp, "\ti2 = i0 + simd_i32(4);\n");
		fprintf(fp, "\ti3 = i0 + simd_i32(6);\n");
		fprintf(fp, "\ty = co[%i].permute(i%i);\n", (N - 1) / 4, (N - 1) % 4);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, co[%i].permute(i%i));\n", n / 4, n % 4);
		}
		fprintf(fp, "\tz = simd_f32(%.9e) - y * z;\n", (double) (hiprec_real(2) * atan(hiprec_real(1))));
		fprintf(fp, "\ty = blend(y, z, i0);\n");
		fprintf(fp, "\treturn s * y;\n");
		fprintf(fp, "}\n");
	}
	/* acos */
#define List(...) {__VA_ARGS__}
	{
		constexpr int N = 18;
		constexpr float coeffs[] =
				{ 0, 2., 0.3333333333, 0.08888888889, 0.02857142857, 0.01015873016, 0.003848003848, 0.001522287237, 0.0006216006216, 0.0002600159463,
						0.0001108489034, 0.00004798653828, 0.00002103757656, 9.321264693e-6, 4.167443738e-6, 1.877744765e-6, 8.517995405e-7, 3.886999686e-7 };
		hiprec_real exact_pi = hiprec_real(4) * atan(hiprec_real(1));
		float pi1 = exact_pi;
		float pi2 = exact_pi - hiprec_real(pi1);
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 acos(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 y, s, z;\n");
		fprintf(fp, "\ts = copysign(simd_f32(1), x);\n");
		fprintf(fp, "\tx = simd_f32(1) - abs(x);\n");
		fprintf(fp, "\ty = simd_f32(%.9e);\n", coeffs[N - 1]);
		for (int n = N - 2; n >= 1; n--) {
			fprintf(fp, "\ty = fma(x, y, simd_f32(%.9e));\n", coeffs[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\ty = sqrt(y);\n");
		fprintf(fp, "\tz = (simd_f32(%.9e) - y) + simd_f32(%.9e);\n", pi1, pi2);
		fprintf(fp, "\ty = blend(y, z, s < simd_f32(0));\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	/* log2 */
	{
		constexpr int N = 4;
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 log2(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 y, y2, z, x0;\n");
		fprintf(fp, "\tsimd_i32 i, j, k;\n");
		fprintf(fp, "\tx0 = x * simd_f32(M_SQRT2);\n");
		fprintf(fp, "\tj = (to_bits(x0) & simd_i32(0x7F800000));\n");
		fprintf(fp, "\tk = (to_bits(x) & simd_i32(0x7F800000));\n");
		fprintf(fp, "\tj >>= simd_i32(23);\n");
		fprintf(fp, "\tk >>= simd_i32(23);\n");
		fprintf(fp, "\tj -= simd_i32(127);\n");
		fprintf(fp, "\tk -= j;\n");
		fprintf(fp, "\tk <<= simd_i32(23);\n");
		fprintf(fp, "\ti = to_bits(x);\n");
		fprintf(fp, "\ti = (i & simd_i32(0x007FFFFF)) | k;\n");
		fprintf(fp, "\tx = from_bits(i);\n");
		fprintf(fp, "\ty = (x - simd_f32(1)) / (x + simd_f32(1));\n");
		fprintf(fp, "\ty2 = y * y;\n");
		fprintf(fp, "\tz = simd_f32(%.9e);\n", 2.0 / (2 * (N - 1) + 1) / log(2));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\tz = fma(z, y2, simd_f32(%.9e));\n", 2.0 / (2 * n + 1) / log(2));
		}
		fprintf(fp, "\tz *= y;\n");
		fprintf(fp, "\tz += simd_f32(j);\n");
		fprintf(fp, "\treturn z;\n");
		fprintf(fp, "}\n");
	}
	/* log */
	{
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 log(simd_f32 x) {\n");
		fprintf(fp, "\treturn log2(x) * simd_f32(%.9e);\n", 1.0 / log2(exp(1)));
		fprintf(fp, "}\n");
	}
	/* log10 */
	{
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 log10(simd_f32 x) {\n");
		fprintf(fp, "\treturn log2(x) * simd_f32(%.9e);\n", 1.0 / log2(10.0));
		fprintf(fp, "}\n");
	}

	/* expm1 */
	{
		constexpr int N = 8;
		int factorial[N];
		factorial[0] = 1;
		factorial[1] = 1;
		for (int n = 2; n < N; n++) {
			factorial[n] = factorial[n - 1] * n;
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 expm1(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 y;\n");
		fprintf(fp, "\tsimd_i32 l;\n");
		fprintf(fp, "\tl = abs(x) < simd_f32(1.0/3.0);\n");
		fprintf(fp, "\ty = simd_f32(%.9e);\n", 1.0 / factorial[N - 1]);
		for (int n = N - 2; n >= 1; n--) {
			fprintf(fp, "\ty = fma(y, x, simd_f32(%.9e));\n", 1.0 / factorial[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\ty = blend(exp(x) - simd_f32(1), y, l);");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}
	/* log1p */
	{
		constexpr int N = 4;
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 log1p(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 y, z, z2;\n");
		fprintf(fp, "\tsimd_i32 l;\n");
		fprintf(fp, "\tl = abs(x) < simd_f32(1.0/3.0);\n");
		fprintf(fp, "\tz = x / (x + simd_f32(2));\n");
		fprintf(fp, "\tz2 = z * z;\n");
		fprintf(fp, "\ty = simd_f32(%.9e);\n", 2.0 / (2 * (N - 1) + 1));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, z2, simd_f32(%.9e));\n", 2.0 / (2 * n + 1));
		}
		fprintf(fp, "\ty *= z;\n");
		fprintf(fp, "\ty = blend(log(x + simd_f32(1)), y, l);");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}

	/* exp */
	{
		constexpr int N = 9;
		double c0[N];
		int nf = 1;
		for (int n = 0; n < N; n++) {
			c0[n] = 1.0 / nf;
			nf *= (n + 1);
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 exp(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 x0, y, zero;\n");
		fprintf(fp, "\tsimd_i32 i;\n");
		fprintf(fp, "\tx = fmax(simd_f32(-87), fmin(simd_f32(87), x));\n");
		fprintf(fp, "\tx0 = round(x * simd_f32(%.9e));\n", 1.0 / M_LN2);
		fprintf(fp, "\tx -= x0 * simd_f32(%.9e);\n", M_LN2);
		fprintf(fp, "\ty = simd_f32(%.9e);\n", c0[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x, y, simd_f32(%.9e));\n", c0[n]);
		}
		fprintf(fp, "\ti = (simd_i32(x0) + simd_i32(127)) << int(23);\n");
		fprintf(fp, "\ty *= from_bits(i);\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}

	{/* exp2 */
		constexpr int N = 9;
		double c0[N];
		int nf = 1;
		for (int n = 0; n < N; n++) {
			c0[n] = 1.0 / nf;
			nf *= (n + 1);
		}
		double log2 = log(2);
		double factor = 1.0;
		for (int n = 0; n < N; n++) {
			c0[n] *= factor;
			factor *= log2;
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f32 exp2(simd_f32 x) {\n");
		fprintf(fp, "\tsimd_f32 x0, y, zero;\n");
		fprintf(fp, "\tsimd_i32 i;\n");
		fprintf(fp, "\tx = fmax(simd_f32(-127), fmin(simd_f32(127), x));\n");
		fprintf(fp, "\tx0 = round(x);\n");
		fprintf(fp, "\tx -= x0;\n");
		fprintf(fp, "\ty = simd_f32(%.9e);\n", c0[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x, y, simd_f32(%.9e));\n", c0[n]);
		}
		fprintf(fp, "\ti = (simd_i32(x0) + simd_i32(127)) << int(23);\n");
		fprintf(fp, "\ty *= from_bits(i);\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}

	/* erfc */
	{

		constexpr int N = 10;
		constexpr int M = 7;
		const hiprec_real xmax = 9.1;
		static hiprec_real c0[N][M];
		hiprec_real a[M];
		for (int n = 0; n < M; n++) {
			hiprec_real xmin = pow(hiprec_real(2), hiprec_real(n) / hiprec_real(2)) - hiprec_real(1);
			hiprec_real xmax1 = pow(hiprec_real(2), hiprec_real(n + 1) / hiprec_real(2)) - hiprec_real(1);
			if (n != M - 1) {
				a[n] = (xmax1 + xmin) / hiprec_real(2);
			} else {
				a[n] = (xmax + xmin) / hiprec_real(2);
			}
		}
		const hiprec_real pi = hiprec_real(4) * atan(hiprec_real(1));
		for (int m = 0; m < M; m++) {
			c0[0][m] = exp(a[m] * a[m]) * erfc(a[m]);
			c0[1][m] = hiprec_real(2) * (hiprec_real(-1) / sqrt(pi) + a[m] * c0[0][m]);
			for (int n = 2; n < N; n++) {
				c0[n][m] = hiprec_real(2) / hiprec_real(n) * (a[m] * c0[n - 1][m] + c0[n - 2][m]);
			}
		}
		fprintf(fp, "simd_f32 erfc(simd_f32 x) {\n");
		fprintf(fp, "\tconstexpr static float x0[] = {");
		for (int m = 0; m < M; m++) {
			fprintf(fp, "%24.9e", double(a[m]));
			if (m != M - 1) {
				fprintf(fp, ", ");
			}
		}
		fprintf(fp, "};\n");
		fprintf(fp, "\tconstexpr static float coeff[%i][%i] = {\n", N, M);
		for (int n = 0; n < N; n++) {
			fprintf(fp, "\t\t{");
			for (int m = 0; m < M; m++) {
				fprintf(fp, "%24.9e", double(c0[n][m]));
				if (m != M - 1) {
					fprintf(fp, ", ");
				}
			}
			fprintf(fp, "}");
			if (n != N - 1) {
				fprintf(fp, ", ");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tsimd_f32 r, z, y, e, a, q, c, neg, rng;\n");
		fprintf(fp, "\tsimd_i32 i;\n");
		fprintf(fp, "\tneg = simd_f32(x < simd_f32(0));\n");
		fprintf(fp, "\trng = simd_f32(x < simd_f32(%.9e));\n", double(xmax));
		fprintf(fp, "\tx = fmin(fabs(x), simd_f32(%.9e));\n", double(xmax));
		fprintf(fp, "\tq = x + simd_f32(1);\n");
		fprintf(fp, "\tx = fmin(x, simd_f32(%.9e));\n", double(xmax));
		fprintf(fp, "\tq *= q;\n");
		fprintf(fp, "\ti = ((to_bits(q) & simd_i32(0x7F800000)) >> int(23)) - simd_i32(127);\n");
		fprintf(fp, "\ty = x * x ;\n");
		fprintf(fp, "\tz = fma(x, x, -y);\n");
		fprintf(fp, "\te = exp(-y) * (simd_f32(1) - z);\n");
		fprintf(fp, "\ta.gather(x0, i);\n");
		fprintf(fp, "\tx -= a;\n");
		fprintf(fp, "\ty = c.gather(coeff[%i], i);\n", N - 1);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, c.gather(coeff[%i], i));\n", n);
		}
		fprintf(fp, "\ty *= e * rng;\n");
		fprintf(fp, "\treturn fma(simd_f32(1) - neg, y, neg * (simd_f32(2) - y));\n");
		fprintf(fp, "}\n");

	}

	/* pow */
	{
		fprintf(fp, "\nsimd_f32 pow(simd_f32 x, simd_f32 y) {\n");
		constexpr int M = 5;
		constexpr int N = 1;
		float coeff[M];
		float coeffx[N];
		float coeffy[N];
		for (int m = N; m < M; m++) {
			coeff[m] = (hiprec_real(2) / (hiprec_real(2 * m + 1))) / log(hiprec_real(2));
		}
		for (int m = 0; m < N; m++) {
			auto exact = (hiprec_real(2) / (hiprec_real(2 * m + 1))) / log(hiprec_real(2));
			coeffx[m] = exact;
			coeffy[m] = exact - hiprec_real(coeffx[m]);
		}
		fprintf(fp, "\tsimd_f32 z, x2, x0;\n");
		fprintf(fp, "\tsimd_i32 i, j, k;\n");
		fprintf(fp, "\tsimd_f32_2 X2, X, Z;\n");
		fprintf(fp, "\tx0 = x * simd_f32(M_SQRT2);\n");
		fprintf(fp, "\tj = (to_bits(x0) & simd_i32(0x7F800000));\n");
		fprintf(fp, "\tk = (to_bits(x) & simd_i32(0x7F800000));\n");
		fprintf(fp, "\tj >>= simd_i32(23);\n");
		fprintf(fp, "\tk >>= simd_i32(23);\n");
		fprintf(fp, "\tj -= simd_i32(127);\n");
		fprintf(fp, "\tk -= j;\n");
		fprintf(fp, "\tk <<= simd_i32(23);\n");
		fprintf(fp, "\ti = to_bits(x);\n");
		fprintf(fp, "\ti = (i & simd_i32(0x7FFFFFULL)) | k;\n");
		fprintf(fp, "\tX = from_bits(i);\n");
		fprintf(fp, "\tX = (X - simd_f32(1)) / (X + simd_f32(1));\n");
		fprintf(fp, "\tX2 = X * X;\n");
		fprintf(fp, "\tx2 = X2.x;\n");
		fprintf(fp, "\tz = simd_f32(%.9e);\n", coeff[M - 1]);
		for (int m = M - 2; m >= N; m--) {
			fprintf(fp, "\tz = fma(z, x2, simd_f32(%.9e));\n", coeff[m]);
		}
		fprintf(fp, "\tZ = simd_f32_2::two_product(z, x2) + simd_f32_2(%.9e, %.9e);\n", coeffx[N - 1], coeffy[N - 1]);
		for (int m = N - 2; m >= 0; m--) {
			fprintf(fp, "\tZ = Z * X2 + simd_f32_2(%.9e, %.9e);\n", coeffx[m], coeffy[m]);
		}
		fprintf(fp, "\tZ = y * (X * Z + simd_f32(j));\n");
		fprintf(fp, "\tz = exp2(Z.x) * (simd_f32(1) + simd_f32(M_LN2) * Z.y);\n");
		fprintf(fp, "\treturn z;\n");
		fprintf(fp, "}\n\n");
	}

}
void double_funcs(FILE* fp) {
	include(fp, std::string(SIMD_ROOT_DIR) + "/include/code64.hpp");

	/* erf */
	{
		constexpr int M = 6;
		static std::vector<hiprec_real> co[M];
		const hiprec_real xmax(6.5);
		static constexpr double toler = std::numeric_limits<double>::epsilon() * 0.5;
		int N = 0;
		for (int i = 0; i < M; i++) {
			hiprec_real a = hiprec_real(i) * xmax / hiprec_real(M);
			hiprec_real b = hiprec_real(i + 1) * xmax / hiprec_real(M);
			std::function<hiprec_real(hiprec_real)> func = [a,b](hiprec_real x) {
				const auto sum = a + b;
				const auto dif = b - a;
				const auto half = hiprec_real(0.5);
				return erf(half*(sum + dif * x));
			};
			co[i] = ChebyCoeffs(func, toler, 0);
			N = std::max(N, (int) co[i].size());
		}
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < M; m++) {
				if (co[m].size() < n + 1) {
					co[m].push_back(0.0);
				}
			}
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 erf(simd_f64 x) {\n");
		fprintf(fp, "\tstatic const double co[][%i] = {\n", M + 1);
		for (int n = 0; n < N; n++) {
			fprintf(fp, "\t\t{");
			double c1, c2, c3, c4, c5, c6, c7;
			if (n < N) {
				int m = n / 2;
				const auto pi = hiprec_real(4) * atan(hiprec_real(1));
				c1 = (n % 2 == 1) ? (double) (hiprec_real(2) / sqrt(pi) * hiprec_real(m % 2 == 0 ? 1 : -1) / factorial(m) / hiprec_real(2 * m + 1)) : 0.0;
				c2 = co[0][n];
				c3 = co[1][n];
				c4 = co[2][n];
				c5 = co[3][n];
				c6 = co[4][n];
				c7 = co[5][n];
				const double factor = (double) (pow(hiprec_real(2) * hiprec_real(M) / xmax, hiprec_real(n)));
				c2 *= factor;
				c3 *= factor;
				c4 *= factor;
				c5 *= factor;
				c6 *= factor;
				c7 *= factor;
			} else {
				c7 = c5 = c6 = c1 = c3 = c4 = c2 = 99.0;
			}
			fprintf(fp, "%24.17e, %24.17e, %24.17e, %24.17e, %24.17e, %24.17e, %24.17e", c1, c2, c3, c4, c5, c6, c7);
			fprintf(fp, "}");
			if (n + 1 < N) {
				fprintf(fp, ",");
			}
			fprintf(fp, "\n");
		}
		auto sqrtpi = sqrt(hiprec_real(4) * atan(hiprec_real(1)));
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tsimd_f64 y, s, z, x2, c;\n");
		fprintf(fp, "\tsimd_i64 i0, i1, l;\n");
		fprintf(fp, "\ts = copysign(simd_f64(1), x);\n");
		fprintf(fp, "\tx = fabs(x);\n");
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\tl = x < simd_f64(0.4);\n");
		fprintf(fp, "\tz = simd_f64(1) - simd_f64(l);\n");
		fprintf(fp, "\tx = fmin(x, simd_f64(%.17e));\n", (double) xmax - 0.00001);
		fprintf(fp, "\ti0 = x * simd_f64(%.17e);\n", (double) (hiprec_real(M) / xmax));
		fprintf(fp, "\ti0 += simd_i64(1);\n");
		fprintf(fp, "\ti0 -= l;\n");
		fprintf(fp, "\tx -= z * simd_f64(%.17e) * i0;\n", (double) (xmax / hiprec_real(M)));
		fprintf(fp, "\tx -= z * simd_f64(%.17e);\n", (double) (xmax * hiprec_real(0.5) / hiprec_real(M)) - (double) (xmax / hiprec_real(M)));
		fprintf(fp, "\ty = c.gather(co[%i], i0);\n", (N - 1));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, c.gather(co[%i], i0));\n", n);
		}
		fprintf(fp, "\treturn s * y;\n");
		fprintf(fp, "}\n");
	}
	/* tgamma */
	{
		static bool init = false;
		constexpr int NCHEBY = 11;
		constexpr int Ntot = NCHEBY + 1;
		constexpr int M = 19;
		constexpr int Mstr = 14;
		constexpr int Msin = 10;
		static double coeffs[M][Ntot];
		static double sincoeffs[Msin];
		static double einvhi, einvlo;
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
				sincoeffs[i / 2] = (double) chebies[i + 1];
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
		fprintf(fp, "\nsimd_f64 tgamma(simd_f64 x) {\n");
		fprintf(fp, "\tstatic const simd_f64_2 Einv(%.17e, %.17e);\n", einvhi, einvlo);
		fprintf(fp, "\tstatic constexpr double coeffs[][%i] = {\n", Ntot);
		for (int m = 0; m < M; m++) {
			fprintf(fp, "\t\t{");
			for (int n = 0; n < Ntot; n++) {
				fprintf(fp, "%25.17e%s", coeffs[m][n], n != Ntot - 1 ? ", " : "");
			}
			fprintf(fp, "}%s\n", m != M - 1 ? "," : "");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tsimd_f64 y, z, x0, c, r, sgn, x2;\n"
				"\tsimd_i64 ic, asym, neg;\n"
				"\tsimd_f64_2 A;\n"
				"\tx0 = x;\n"
				"\tneg = (x <= simd_f64(-0.5));\n"
				"\tx = blend(x, -x, neg);\n"
				"\tasym = x > simd_f64(8.5);\n"
				"\tx2 = round(x);\n"
				"\tic = blend(x2, simd_f64(%i), asym);\n", Ntot - 1);
		fprintf(fp, "\tz = blend(x - x2, simd_f64(1) / x, asym);\n"
				"");
		fprintf(fp, "\ty.gather(coeffs[%i], ic);\n", M - 1);
		for (int m = M - 2; m >= 0; m--) {
			fprintf(fp, "\ty = fma(y, z, c.gather(coeffs[%i], ic));\n", m);
		}
		fprintf(fp, "\tA = x;\n"
				"\tA = A * Einv;\n"
				"\tc = x - simd_f64(0.5);\n"
				"\tx2 = pow(A.x, c);\n"
				"\tx2 *= (simd_f64(1) + c * A.y / A.x);\n"
				"\ty = blend(simd_f64(1) / y, y * x2, asym);\n"
				"\tr = x0 - floor(x0);\n"
				"\tr = blend(r, simd_f64(1) - r, r > simd_f64(0.5));\n"
				"\tsgn = blend(simd_f64(-1), simd_f64(1), simd_i64(floor(x0)) & simd_i64(1));\n"
				"\tx2 = simd_f64(4) * r * r;\n");
		fprintf(fp, "\tz = simd_f64(%.17e);\n", sincoeffs[Msin - 1]);
		for (int m = Msin - 2; m >= 0; m--) {
			fprintf(fp, "\tz = fma(z, x2, simd_f64(%.17e));\n", sincoeffs[m]);
		}
		fprintf(fp, "\tz *= simd_f64(2) * r;\n"
				"\ty = blend(y, sgn / (y * z * x0), neg);\n"
				"\treturn y;\n"
				"}\n");

	}

	/* lgamma */
	{
		static bool init = false;
		constexpr int NCHEBY = 12;
		static constexpr int NROOTS = 32;
		constexpr int Ntot = NROOTS + NCHEBY + 1;
		constexpr int M = 21;
		constexpr int Msin = 20;
		constexpr int Mlog = 10;
		static double coeffs[M][Ntot];
		static double bias[Ntot];
		static double Xc[Ntot];
		static double factor = 10.0;
		static double rootbegin[NROOTS];
		static double rootend[NROOTS];
		static double logsincoeffs[Msin];
		if (!init) {
			init = true;
			double x0 = 2.0;
			int n = 0;
			while (x0 > -17) {
				double xrt = lgamma_root(x0);
				if (n == 0) {
					xrt = 2.0;
				} else if (n == 1) {
					xrt = 1.0;
				}
				Xc[n + NCHEBY] = xrt;
				double x1 = round(x0);
				auto co = gammainv_coeffs(M, xrt);
				const double eps = 0.5 * std::numeric_limits<double>::epsilon();
				double span1 = std::min(pow(eps * fabs(co[0] / co.back()), 1.0 / (co.size() - 1)), 0.5);
				double a = xrt < -0.5 ? std::min(((fabs(xrt)) / 5.5), 0.80) : 1.0;
				double span2 = (0.5 + a * 0.5) * fabs(xrt - round(xrt));
				span2 = nextafter(span2, 0.0);
				if (n == 0) {
					rootbegin[n] = 1.75;
					rootend[n] = 2.25;
				} else if (n == 1) {
					rootbegin[n] = 0.75;
					rootend[n] = 1.25;
				} else {
					if (n % 2 == 0) {
						rootbegin[n] = xrt - span1;
						rootend[n] = xrt + span2;
					} else {
						rootbegin[n] = xrt - span2;
						rootend[n] = xrt + span1;
					}
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
				if (x0 < 0.0) {
					if (rootend[n] > rootbegin[n - 1]) {
						auto avg = 0.5 * (Xc[n + NCHEBY] + Xc[n + NCHEBY - 1]);
						rootend[n] = rootbegin[n - 1] = avg;
					}
				}
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
		fprintf(fp, "\n\nsimd_f64 lgamma(simd_f64 x) {\n");
		fprintf(fp, "\tstatic constexpr double coeffs[][%i] = {\n", Ntot);
		for (int m = 0; m < M; m++) {
			fprintf(fp, "\t\t{");
			for (int n = 0; n < Ntot; n++) {
				fprintf(fp, "%25.17e%s", coeffs[m][n], n != Ntot - 1 ? ", " : "");
			}
			fprintf(fp, "}%s\n", m != M - 1 ? "," : "");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tstatic constexpr double bias[] = {");
		for (int n = 0; n < Ntot; n++) {
			fprintf(fp, "%25.17e%s", bias[n], n != Ntot - 1 ? ", " : "");
		}
		fprintf(fp, "};\n");
		fprintf(fp, "\tstatic constexpr double Xc[] = {");
		for (int n = 0; n < Ntot; n++) {
			fprintf(fp, "%25.17e%s", Xc[n], n != Ntot - 1 ? ", " : "");
		}
		fprintf(fp, "};\n");
		fprintf(fp, "\tstatic constexpr double rootbegin[] = {");
		for (int n = 0; n < NROOTS; n++) {
			fprintf(fp, "%25.17e%s", rootbegin[n], n != NROOTS - 1 ? ", " : "");
		}
		fprintf(fp, "};\n");
		fprintf(fp, "\tstatic constexpr double rootend[] = {");
		for (int n = 0; n < NROOTS; n++) {
			fprintf(fp, "%25.17e%s", rootend[n], n != NROOTS - 1 ? ", " : "");
		}
		fprintf(fp, "};\n");
		fprintf(fp, "\tconst auto logxor1px = [](simd_f64 x, simd_f64 xm1) {\n"
				"\t\tsimd_f64 x1, z, z2, y, x0;\n"
				"\t\tsimd_i64 k, j;\n"
				"\t\tx0 = x * simd_f64(M_SQRT2);\n"
				"\t\tfrexp(x0, &j);\n"
				"\t\tx1 = simd_f64(2) * frexp(x, &k);\n"
				"\t\tj = j - simd_f64(1);\n"
				"\t\tk = k - simd_f64(1);\n"
				"\t\tk -= j;\n"
				"\t\tx1 = ldexp(x1, k);\n"
				"\t\tz = blend( (x1 - simd_f64(1)) / (x1 + simd_f64(1)), xm1 / (xm1 + simd_f64(2)), j == simd_i64(0));\n"
				"\t\tz2 = z * z;\n"
				"\t\ty = simd_f64(%.17e);\n", (double) (2.0L / (1.0L + 2.0L * (long double) (Mlog - 1))));
		for (int n = Mlog - 2; n >= 0; n--) {
			fprintf(fp, "\t\ty = fma(y, z2, simd_f64(%.17e));\n", (double) (2.0L / (1.0L + 2.0L * (long double) n)));
		}
		fprintf(fp, "\t\ty *= z;\n"
				"\t\ty += simd_f64(j) * simd_f64(%.17e);\n", (double) logl(2));
		fprintf(fp, "\t\treturn y;\n"
				"\t};\n"
				"");
		fprintf(fp, "\tsimd_f64 y, z, x0, zm1, logx, r, b, c, x2;\n"
				"\tsimd_i64 ic, nearroot, nearroot1, nearroot2, asym, neg, yneg;\n"
				"\tsimd_f64_2 Y;\n"
				"\tx0 = x;\n"
				"\tic = simd_i64(fmin(simd_f64(2) * fmax(floor(-x - simd_f64(1)), simd_f64(0)), simd_f64(%i)));\n", NROOTS - 2);
		fprintf(fp, "\tnearroot1 = (x > c.gather(rootbegin, ic) && x < c.gather(rootend, ic));\n"
				"\tnearroot2 = (x > c.gather(rootbegin, ic + simd_i64(1)) && x < c.gather(rootend, ic + simd_i64(1)));\n"
				"\tnearroot = nearroot1 || nearroot2;\n"
				"\tneg = !nearroot && (x <= simd_f64(-3.5));\n"
				"\tx = blend(x, -x, neg);\n"
				"\tasym = !nearroot && (x >= simd_f64(7.5));\n"
				"\tic = blend(blend(simd_i64(round(x)) + simd_i64(3), simd_i64(%i), asym), ic + simd_i64(%i) + nearroot2, nearroot);\n", Ntot - 1, NCHEBY);

		fprintf(fp, "\tz = blend(x - c.gather(Xc, ic), simd_f64(1) / x, asym);\n"
				"");
		fprintf(fp, "\ty = c.gather(coeffs[%i], ic);\n", M - 1);
		for (int m = M - 2; m >= 0; m--) {
			fprintf(fp, "\ty = fma(y, z, c.gather(coeffs[%i], ic));\n", m);
		}
		fprintf(fp, "\tlogx = log(x);\n"
				"\ty = blend(y, simd_f64(1) / y, asym);\n"
				"\tb.gather(bias, ic);\n"
				"\tyneg = b + y < simd_f64(0);\n"
				"\tz = blend(b + y, -y - b, yneg);\n"
				"\tzm1 = blend(y - (simd_f64(1) - b), -y - (simd_f64(1) + b), yneg);\n"
				"\ty = -logxor1px(z, zm1);\n"
				"\ty += blend(simd_f64(0), -x + (x - simd_f64(0.5)) * logx + simd_f64(%.17e), asym);\n", (double) (0.5L * log(2.0L * M_PIl)));
		fprintf(fp, "\tr = x0 - floor(x0);\n"
				"\tr = blend(r, simd_f64(1) - r, r > simd_f64(0.5));\n"
				"\tx2 = r * r;\n"
				"\tz = simd_f64(%.17e);\n", logsincoeffs[Msin - 1]);
		for (int m = Msin - 2; m >= 0; m--) {
			fprintf(fp, "\tz = fma(z, x2, simd_f64(%.17e));\n", logsincoeffs[m]);
		}
		fprintf(fp, "\tY = simd_f64_2::quick_two_sum(log(abs(r)), z);\n"
				"\tY = Y + simd_f64_2::two_sum(y, logx);\n"
				"\ty = blend(y, -Y.x, neg);\n"
				"\treturn y;\n");
		fprintf(fp, "}\n");
	}
	/* asin */
	{
		static std::vector<hiprec_real> co1;
		static std::vector<hiprec_real> co2;
		static const hiprec_real z0(0.5);
		static constexpr double toler = std::numeric_limits<double>::epsilon() * 0.5;
		auto func3 = [](hiprec_real y) {
			const auto x = hiprec_real(1) - y * y;
			const auto a = asin(x);
			static const auto b = hiprec_real(2) * atan(hiprec_real(1));
			const auto c = sqrt(hiprec_real(1) - x);
			return (b - a)/c;
		};
		auto func2 = [func3](hiprec_real x) {
			const static auto z1 = sqrt(hiprec_real(1) - z0);
			return func3((z1 + x * hiprec_real(z1)) / hiprec_real(2));
		};
		co2 = ChebyCoeffs(func2, toler, 0);
		do {
			int n = co1.size();
			co1.push_back(
					factorial(hiprec_real(2 * n))
							/ (pow(hiprec_real(2), hiprec_real(2 * n)) * factorial(hiprec_real(n)) * factorial(hiprec_real(n)) * (hiprec_real(2 * n + 1))));

		} while (std::abs((double) co1.back() * pow(double(z0), 2 * co1.size() - 1)) > toler);
		auto tmp = co1;
		co1.resize(0);
		for (int i = 0; i < tmp.size(); i++) {
			co1.push_back(0.0);
			co1.push_back(tmp[i]);
		}
		int N = std::max(co1.size(), co2.size());
		for (int n = 0; n < N; n++) {
			if (co1.size() < n + 1) {
				co1.push_back(0.0);
			}
			if (co2.size() < n + 1) {
				co2.push_back(0.0);
			}
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 asin(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 y, s, z, w, x0, x1;\n");
		fprintf(fp, "\tsimd_i64 i;\n");
		fprintf(fp, "\tsize_t j;\n");
		fprintf(fp, "\tstatic const simd_f64 co[][%i] = {\n", N);
		for (int bits = 0; bits < 16; bits++) {
			fprintf(fp, "\t\t{\n");
			for (int n = 0; n < N; n++) {
				fprintf(fp, "\t\t\t{");
				for (int i = 0; i < 4; i++) {
					double c1;
					if ((bits >> i) & 0x1) {
						c1 = co2[n];
					} else {
						c1 = co1[n];
					}
					fprintf(fp, "%.17e", c1);
					if (i != 3) {
						fprintf(fp, ", ");
					}
				}
				fprintf(fp, "}");
				if (n + 1 < N) {
					fprintf(fp, ",");
				}
				fprintf(fp, "\n");
			}
			fprintf(fp, "\t\t}");
			if (bits + 1 < 16) {
				fprintf(fp, ",");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\ts = copysign(simd_f64(1), x);\n");
		fprintf(fp, "\tx = fabs(x);\n");
		fprintf(fp, "\ti = x > simd_f64(%.17e);\n", (double) z0);
		fprintf(fp, "\tz = sqrt(simd_f64(1) - x);\n");
		fprintf(fp, "\tx0 = x;\n");
		fprintf(fp, "\tx1 = simd_f64(%.17e) * z - simd_f64(1);\n", (double) (hiprec_real(2) / sqrt(hiprec_real(1) - z0)));
		fprintf(fp, "\tx = blend(x0, x1, i);\n");
		fprintf(fp, "\ti = -i;\n");
		fprintf(fp, "\tj = movemask(i);\n");
		fprintf(fp, "\ty = co[j][%i];\n", (N - 1));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, co[j][%i]);\n", n);
		}
		fprintf(fp, "\tz = simd_f64(%.17e) - y * z;\n", (double) (hiprec_real(2) * atan(hiprec_real(1))));
		fprintf(fp, "\ty = blend(y, z, -i);\n");
		fprintf(fp, "\treturn s * y;\n");
		fprintf(fp, "}\n");
	}
	/* acos */
	{

		constexpr int N = 47;
		constexpr double coeffs[] = { 0, 2., 0.33333333333333333333, 0.088888888888888888889, 0.028571428571428571429, 0.01015873015873015873,
				0.0038480038480038480038, 0.0015222872365729508587, 0.00062160062160062160062, 0.00026001594629045609438, 0.00011084890341856286129,
				0.000047986538276434139085, 0.000021037576563219314599, 9.321264692626404007e-6, 4.167443738237730892e-6, 1.8777447648151615054e-6,
				8.5179954049074866675e-7, 3.8869996856618833991e-7, 1.7830839827877528608e-7, 8.2179119548112649632e-8, 3.8034182252395726304e-8,
				1.7669771081252369944e-8, 8.2371765822751534304e-9, 3.8519743631122456622e-9, 1.8064667004311861306e-9, 8.4940801587621486629e-10,
				4.0036199843335919414e-10, 1.8912977703770147816e-10, 8.9529615234080764659e-11, 4.2462927007573696003e-11, 2.0175887917157897366e-11,
				9.6024849949455883817e-12, 4.5773750397533285887e-12, 2.1851897625675563985e-12, 1.0446319804372558902e-12, 5.0003915916582517976e-13,
				2.3965100546875424304e-13, 1.1498989377545557163e-13, 5.523549634336795704e-14, 2.6560125447826616705e-14, 1.2784161647514013927e-14,
				6.159186581157007613e-15, 2.9700495246485742391e-15, 1.4334247227031696191e-15, 6.9237259986367832437e-16, 3.3468997586419007141e-16,
				1.6190807480291086828e-16 };
		hiprec_real exact_pi = hiprec_real(4) * atan(hiprec_real(1));
		double pi1 = exact_pi;
		double pi2 = exact_pi - hiprec_real(pi1);
		fprintf(fp, "\n");

		fprintf(fp, "simd_f64 acos(simd_f64 x) {\n");

		fprintf(fp, "\tsimd_f64 y1, y2, y, s, z, x2;\n");
		fprintf(fp, "\ts = copysign(simd_f64(1), x);\n");
		fprintf(fp, "\tx = simd_f64(1) - abs(x);\n");
		fprintf(fp, "\ty = simd_f64(%.17e);\n", coeffs[N - 1]);
		for (int n = N - 2; n >= 1; n -= 1) {
			fprintf(fp, "\ty = fma(x, y, simd_f64(%.17e));\n", coeffs[n]);
		}
		fprintf(fp, "\ty = x * y;\n");
		fprintf(fp, "\ty = sqrt(y);\n");
		fprintf(fp, "\tz = (simd_f64(%.17e) - y) + simd_f64(%.17e);\n", pi1, pi2);
		fprintf(fp, "\ty = blend(y, z, s < simd_f64(0));\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	/*log*/
	{
		constexpr int N = 10;
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 log2(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 y, y2, z, x0;\n");
		fprintf(fp, "\tsimd_i64 i, j, k;\n");
		fprintf(fp, "\tx0 = x * simd_f64(M_SQRT2);\n");
		fprintf(fp, "\tj = (to_bits(x0) & simd_i64(0x7FF0000000000000ULL));\n");
		fprintf(fp, "\tk = (to_bits(x) & simd_i64(0x7FF0000000000000ULL));\n");
		fprintf(fp, "\tj >>= simd_i64(52);\n");
		fprintf(fp, "\tk >>= simd_i64(52);\n");
		fprintf(fp, "\tj -= simd_i64(1023);\n");
		fprintf(fp, "\tk -= j;\n");
		fprintf(fp, "\tk <<= simd_i64(52);\n");
		fprintf(fp, "\ti = to_bits(x);\n");
		fprintf(fp, "\ti = (i & simd_i64(0xFFFFFFFFFFFFFULL)) | k;\n");
		fprintf(fp, "\tx = from_bits(i);\n");
		fprintf(fp, "\ty = (x - simd_f64(1)) / (x + simd_f64(1));\n");
		fprintf(fp, "\ty2 = y * y;\n");
		fprintf(fp, "\tz = simd_f64(%.17e);\n", (double) (2.0L / (long double) (2 * (N - 1) + 1) / logl(2)));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\tz = fma(z, y2, simd_f64(%.17e));\n", (double) (2.0L / (long double) (2 * n + 1) / log(2)));
		}
		fprintf(fp, "\tz *= y;\n");
		fprintf(fp, "\tz += simd_f64(j);\n");
		fprintf(fp, "\treturn z;\n");
		fprintf(fp, "}\n");
	}

	/* log */
	{
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 log(simd_f64 x) {\n");
		fprintf(fp, "\treturn log2(x) * simd_f64(%.17e);\n", double(1.0L / log2l(expl(1.0L))));
		fprintf(fp, "}\n");
	}
	/* log10 */
	{
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 log10(simd_f64 x) {\n");
		fprintf(fp, "\treturn log2(x) * simd_f64(%.17e);\n", double(1.0L / log2l(10.0L)));
		fprintf(fp, "}\n");
	}
	/* cos */
	{
		constexpr int N = 11;
		double coeff[N];
		hiprec_real pi_exact = hiprec_real(4) * atan(hiprec_real(1));
		double pi1 = pi_exact;
		double pi2 = pi_exact - hiprec_real(pi1);
		hiprec_real fac(1);
		for (int n = 0; n < N; n++) {
			coeff[n] = hiprec_real(1) / fac;
			fac *= hiprec_real(2 * n + 2);
			fac *= -hiprec_real(2 * n + 3);
		}
		fprintf(fp, "\nsimd_f64 cos(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 x0, s, x2, y;\n");
		fprintf(fp, "\tx = abs(x);\n");
		fprintf(fp, "\tx0 = floor(x * simd_f64(%.17e));\n", 1.0 / M_PI);
		fprintf(fp, "\ts = simd_f64(2) * (x0 - simd_f64(2) * floor(x0 * simd_f64(0.5))) - simd_f64(1);\n");
		fprintf(fp, "\tx0 += simd_f64(0.5);\n");
		fprintf(fp, "\tx = s * fma(-x0, simd_f64(%.17e), fma(-x0, simd_f64(%.17e), x));\n", pi2, pi1);
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\ty = simd_f64(%.17e);\n", coeff[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x2, y, simd_f64(%.17e));\n", coeff[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	/* sin */
	{
		constexpr int N = 11;
		double coeff[N];
		hiprec_real pi_exact = hiprec_real(4) * atan(hiprec_real(1));
		double pi1 = pi_exact;
		double pi2 = pi_exact - hiprec_real(pi1);
		hiprec_real fac(1);
		for (int n = 0; n < N; n++) {
			coeff[n] = hiprec_real(1) / fac;
			fac *= hiprec_real(2 * n + 2);
			fac *= -hiprec_real(2 * n + 3);
		}
		fprintf(fp, "\nsimd_f64 sin(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 x0, s, x2, y;\n");
		fprintf(fp, "\tx0 = round(x * simd_f64(%.17e));\n", 1.0 / M_PI);
		fprintf(fp, "\ts = -simd_f64(2) * (x0 - simd_f64(2) * floor(x0 * simd_f64(0.5))) + simd_f64(1);\n");
		fprintf(fp, "\tx = s * fma(-x0, simd_f64(%.17e), fma(-x0, simd_f64(%.17e), x));\n", pi2, pi1);
		fprintf(fp, "\tx2 = x * x;\n");
		fprintf(fp, "\ty = simd_f64(%.17e);\n", coeff[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x2, y, simd_f64(%.17e));\n", coeff[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n");
	}
	/* log1p */
	{
		constexpr int N = 9;
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 log1p(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 y, z, z2;\n");
		fprintf(fp, "\tsimd_i64 l;\n");
		fprintf(fp, "\tl = abs(x) < simd_f64(1.0/3.0);\n");
		fprintf(fp, "\tz = x / (x + simd_f64(2));\n");
		fprintf(fp, "\tz2 = z * z;\n");
		fprintf(fp, "\ty = simd_f64(%.17e);\n", 2.0 / (2 * (N - 1) + 1));
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, z2, simd_f64(%.17e));\n", 2.0 / (2 * n + 1));
		}
		fprintf(fp, "\ty *= z;\n");
		fprintf(fp, "\ty = blend(log(x + simd_f64(1)), y, l);");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}

	{
		/* expm1 */

		constexpr int N = 14;
		int factorial[N];
		factorial[0] = 1;
		factorial[1] = 1;
		for (int n = 2; n < N; n++) {
			factorial[n] = factorial[n - 1] * n;
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 expm1(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 y;\n");
		fprintf(fp, "\tsimd_i64 l;\n");
		fprintf(fp, "\tl = abs(x) < simd_f64(1.0/3.0);\n");
		fprintf(fp, "\ty = simd_f64(%.17e);\n", 1.0 / factorial[N - 1]);
		for (int n = N - 2; n >= 1; n--) {
			fprintf(fp, "\ty = fma(y, x, simd_f64(%.17e));\n", 1.0 / factorial[n]);
		}
		fprintf(fp, "\ty *= x;\n");
		fprintf(fp, "\ty = blend(exp(x) - simd_f64(1), y, l);");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}

	{
		constexpr int N = 13;
		hiprec_real c0[N];
		int nf = 1;
		for (int n = 0; n < N; n++) {
			c0[n] = hiprec_real(1) / hiprec_real(nf);
			c0[n] *= pow(log(hiprec_real(2)), hiprec_real(n));
			nf *= (n + 1);
		}
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 exp2(simd_f64 x) {\n");
		fprintf(fp, "\tsimd_f64 x0, y;\n");
		fprintf(fp, "\tsimd_i64 i;\n");
		fprintf(fp, "\tx = fmax(simd_f64(-1000), fmin(simd_f64(1000), x));\n");
		fprintf(fp, "\tx0 = round(x);\n");
		fprintf(fp, "\tx -= x0;\n");
		fprintf(fp, "\ty = simd_f64(%.20e);\n", (double) c0[N - 1]);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(x, y, simd_f64(%.17e));\n", (double) c0[n]);
		}
		fprintf(fp, "\ti = (simd_i64(x0) + simd_i64(1023)) << (long long)(52);\n");
		fprintf(fp, "\ty *= from_bits(i);\n");
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
		fprintf(fp, "\n");
	}
	/*	{

	 constexpr int N = 15;
	 constexpr int M = 10;
	 static double coeffs[N];
	 static double epwr[M];
	 static std::once_flag once;
	 std::call_once(once, []() {
	 hiprec_real fac(1);
	 for( int n = 0; n < N; n++) {
	 coeffs[n] = (double)(hiprec_real(1) / fac);
	 fac *= hiprec_real(n+1);
	 }
	 hiprec_real ep = exp(hiprec_real(1));
	 for( int m = 0; m < M; m++) {
	 epwr[m] = ep;
	 ep *= ep;
	 }
	 });
	 fprintf(fp, "\nsimd_f64 exp(simd_f64 x) {\n");
	 fprintf(fp, "\tsimd_f64 x0, y;\n");
	 fprintf(fp, "\tsimd_i64 i, j;\n");
	 fprintf(fp, "\ti = x < simd_f64(0);\n");
	 fprintf(fp, "\tx = abs(x);\n");
	 fprintf(fp, "\tx0 = round(x);\n");
	 fprintf(fp, "\tx -= x0;\n");
	 fprintf(fp, "\ty = simd_f64(%.16e);\n", (double) coeffs[N - 1]);
	 for (int n = N - 2; n >= 0; n--) {
	 fprintf(fp, "\ty = fma(x, y, simd_f64(%.17e));\n", (double) coeffs[n]);
	 }
	 fprintf(fp, "\tj = simd_i64(x0);\n");
	 for (int m = 0; m < M; m++) {
	 fprintf(fp, "\ty = blend(y, y * simd_f64(%.17e), j & simd_i64(0x%x));\n", (double) epwr[m], 1 << m);
	 }
	 fprintf(fp, "\ty = blend(y, simd_f64(1) / y, i);\n");
	 fprintf(fp, "\treturn y;\n");
	 fprintf(fp, "}\n");
	 }*/
	{
		fprintf(fp, "\n");
		fprintf(fp, "simd_f64 exp(simd_f64 x) {\n");
		hiprec_real exact = log2(exp(hiprec_real(1)));
		double_2 log2e = double_2::quick_two_sum(double(exact), exact - hiprec_real(double(exact)));
		fprintf(fp, "\tstatic const simd_f64_2 LOG2E(%.17e, %.17e);\n", log2e.x, log2e.y);
		fprintf(fp, "\tsimd_f64 y;\n");
		fprintf(fp, "\tsimd_f64_2 XLOG2E = x * LOG2E;\n");
		fprintf(fp, "\ty = exp2(XLOG2E.x) * (simd_f64(1) + simd_f64(%.17e) * XLOG2E.y);\n", log(2));
		fprintf(fp, "\treturn y;\n");
		fprintf(fp, "}\n\n");
	}
	/* erfc */
	{

		constexpr int N = 15;
		constexpr int M = 20;
		const hiprec_real xmax = 26.6;
		static hiprec_real c0[N][M];
		hiprec_real a[M];
		for (int n = 0; n < M; n++) {
			hiprec_real xmin = pow(hiprec_real(2), hiprec_real(n) / hiprec_real(4)) - hiprec_real(1);
			hiprec_real xmax1 = pow(hiprec_real(2), hiprec_real(n + 1) / hiprec_real(4)) - hiprec_real(1);
			if (n != M - 1) {
				a[n] = (xmax1 + xmin) / hiprec_real(2);
			} else {
				a[n] = (xmax + xmin) / hiprec_real(2);
			}
		}
		const hiprec_real pi = hiprec_real(4) * atan(hiprec_real(1));
		for (int m = 0; m < M; m++) {
			c0[0][m] = exp(a[m] * a[m]) * erfc(a[m]);
			c0[1][m] = hiprec_real(2) * (hiprec_real(-1) / sqrt(pi) + a[m] * c0[0][m]);
			for (int n = 2; n < N; n++) {
				c0[n][m] = hiprec_real(2) / hiprec_real(n) * (a[m] * c0[n - 1][m] + c0[n - 2][m]);
			}
		}
		fprintf(fp, "simd_f64 erfc(simd_f64 x) {\n");
		fprintf(fp, "\tconstexpr static double x0[] = {");
		for (int m = 0; m < M; m++) {
			fprintf(fp, "%24.17e", double(a[m]));
			if (m != M - 1) {
				fprintf(fp, ", ");
			}
		}
		fprintf(fp, "};\n");
		fprintf(fp, "\tconstexpr static double coeff[%i][%i] = {\n", N, M);
		for (int n = 0; n < N; n++) {
			fprintf(fp, "\t\t{");
			for (int m = 0; m < M; m++) {
				fprintf(fp, "%24.17e", double(c0[n][m]));
				if (m != M - 1) {
					fprintf(fp, ", ");
				}
			}
			fprintf(fp, "}");
			if (n != N - 1) {
				fprintf(fp, ", ");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\t};\n");
		fprintf(fp, "\tsimd_f64 r, y, e, a, q, c, neg, rng;\n");
		fprintf(fp, "\tsimd_f64_2 Z;\n");
		fprintf(fp, "\tsimd_i64 i;\n");
		fprintf(fp, "\tneg = simd_f64(x < simd_f64(0));\n");
		fprintf(fp, "\trng = simd_f64(x < simd_f64(%.17e));\n", double(xmax));
		fprintf(fp, "\tx = fmin(fabs(x), simd_f64(%.17e));\n", double(xmax));
		fprintf(fp, "\tq = x + simd_f64(1);\n");
		fprintf(fp, "\tx = fmin(x, simd_f64(%.17e));\n", double(xmax));
		fprintf(fp, "\tq *= q;\n");
		fprintf(fp, "\tq *= q;\n");
		fprintf(fp, "\ti = ((to_bits(q) & simd_i64(0x7FF0000000000000)) >> (long long)(52)) - simd_i64(1023);\n");
		fprintf(fp, "\tZ = simd_f64_2::two_product(x, x) ;\n");
		fprintf(fp, "\te = exp(-Z.x) * (simd_f64(1) - Z.y);\n");
		fprintf(fp, "\ta.gather(x0, i);\n");
		fprintf(fp, "\tx -= a;\n");
		fprintf(fp, "\ty = c.gather(coeff[%i], i);\n", N - 1);
		for (int n = N - 2; n >= 0; n--) {
			fprintf(fp, "\ty = fma(y, x, c.gather(coeff[%i], i));\n", n);
		}
		fprintf(fp, "\ty *= e * rng;\n");
		fprintf(fp, "\treturn fma(simd_f64(1) - neg, y, neg * (simd_f64(2) - y));\n");
		fprintf(fp, "}\n");

	}

	/* pow */
	{
		fprintf(fp, "\nsimd_f64 pow(simd_f64 x, simd_f64 y) {\n");
		constexpr int M = 11;
		constexpr int N = 2;
		double coeff[M];
		double coeffx[N];
		double coeffy[N];
		double c0x;
		double c0y;
		for (int m = N; m < M; m++) {
			coeff[m] = (hiprec_real(2) / (hiprec_real(2 * m + 1))) / log(hiprec_real(2));
		}
		for (int m = 0; m < N; m++) {
			auto exact = (hiprec_real(2) / (hiprec_real(2 * m + 1))) / log(hiprec_real(2));
			coeffx[m] = exact;
			coeffy[m] = exact - hiprec_real(coeffx[m]);
		}
		fprintf(fp, "\tsimd_f64 z, x2, x0;\n");
		fprintf(fp, "\tsimd_i64 i, j, k;\n");
		fprintf(fp, "\tsimd_f64_2 X2, X, Z;\n");
		fprintf(fp, "\tx0 = x * simd_f64(M_SQRT2);\n");
		fprintf(fp, "\tj = (to_bits(x0) & simd_i64(0x7FF0000000000000ULL));\n");
		fprintf(fp, "\tk = (to_bits(x) & simd_i64(0x7FF0000000000000ULL));\n");
		fprintf(fp, "\tj >>= simd_i64(52);\n");
		fprintf(fp, "\tk >>= simd_i64(52);\n");
		fprintf(fp, "\tj -= simd_i64(1023);\n");
		fprintf(fp, "\tk -= j;\n");
		fprintf(fp, "\tk <<= simd_i64(52);\n");
		fprintf(fp, "\ti = to_bits(x);\n");
		fprintf(fp, "\ti = (i & simd_i64(0xFFFFFFFFFFFFFULL)) | k;\n");
		fprintf(fp, "\tX = from_bits(i);\n");
		fprintf(fp, "\tX = (X - simd_f64(1)) / (X + simd_f64(1));\n");
		fprintf(fp, "\tX2 = X * X;\n");
		fprintf(fp, "\tx2 = X2.x;\n");
		fprintf(fp, "\tz = simd_f64(%.17e);\n", coeff[M - 1]);
		for (int m = M - 2; m >= N; m--) {
			fprintf(fp, "\tz = fma(z, x2, simd_f64(%.17e));\n", coeff[m]);
		}
		fprintf(fp, "\tZ = simd_f64_2::two_product(z, x2) + simd_f64_2(%.17e, %.17e);\n", coeffx[N - 1], coeffy[N - 1]);
		for (int m = N - 2; m >= 0; m--) {
			fprintf(fp, "\tZ = Z * X2 + simd_f64_2(%.17e, %.17e);\n", coeffx[m], coeffy[m]);
		}
		fprintf(fp, "\tZ = y * (X * Z + simd_f64(j));\n");
		fprintf(fp, "\tz = exp2(Z.x) * (simd_f64(1) + simd_f64(M_LN2) * Z.y);\n");
		fprintf(fp, "\treturn z;\n");
		fprintf(fp, "}\n");
	}

}

/*void two_sum(simd_f64* xptr, simd_f64* yptr, simd_f64 a, simd_f64 b) {
 auto& x = *xptr;
 auto& y = *yptr;
 x = a + b;
 const simd_f64 bvirtual = x - a;
 const simd_f64 avirtual = x - bvirtual;
 const simd_f64 broundoff = b - bvirtual;
 const simd_f64 aroundoff = a - avirtual;
 y = aroundoff + broundoff;
 }

 void two_product(simd_f64* xptr, simd_f64* yptr, simd_f64 a, simd_f64 b) {
 auto& x = *xptr;
 auto& y = *yptr;
 x = a * b;
 y = fma(a, b, -x);
 }*/

int main() {
	/* The build system creates the output directory (cmake -E make_directory)
	   before running the generator. */
	const char* const out_path = "./generated_code/src/math.cpp";
	FILE* fp = fopen(out_path, "wt");
	if (!fp) {
		fprintf(stderr, "Unable to open %s for writing\n", out_path);
		return 1;
	}
	fprintf(fp, "#include \"simd.hpp\"\n");
	fprintf(fp, "#include <utility>\n");
	fprintf(fp, "\nnamespace simd {\n\n");
	float_funcs(fp);
	double_funcs(fp);
	fprintf(fp, "\n}\n");
	fclose(fp);
	return 0;
}

