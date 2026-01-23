#pragma once
#include <xmmintrin.h> // For SSE
#include <immintrin.h> // For AVX


float weno_minus_core(const float a, const float b, const float c, const float d, const float e)
{
		const float is0 = a*(a*(float)(4./3.)  - b*(float)(19./3.)  + c*(float)(11./3.)) + b*(b*(float)(25./3.)  - c*(float)(31./3.)) + c*c*(float)(10./3.);
		const float is1 = b*(b*(float)(4./3.)  - c*(float)(13./3.)  + d*(float)(5./3.))  + c*(c*(float)(13./3.)  - d*(float)(13./3.)) + d*d*(float)(4./3.);
		const float is2 = c*(c*(float)(10./3.) - d*(float)(31./3.)  + e*(float)(11./3.)) + d*(d*(float)(25./3.)  - e*(float)(19./3.)) + e*e*(float)(4./3.);

		const float is0plus = is0 + (float)WENOEPS;
		const float is1plus = is1 + (float)WENOEPS;
		const float is2plus = is2 + (float)WENOEPS;

		const float alpha0 = (float)(0.1)*((float)1/(is0plus*is0plus));
		const float alpha1 = (float)(0.6)*((float)1/(is1plus*is1plus));
		const float alpha2 = (float)(0.3)*((float)1/(is2plus*is2plus));
		const float alphasum = alpha0+alpha1+alpha2;
		const float inv_alpha = ((float)1)/alphasum;

		const float omega0 = alpha0 * inv_alpha;
		const float omega1 = alpha1 * inv_alpha;
		const float omega2 = 1-omega0-omega1;

		return omega0*((float)(1.0/3.)*a - (float)(7./6.)*b + (float)(11./6.)*c) +
					 omega1*(-(float)(1./6.)*b + (float)(5./6.)*c + (float)(1./3.)*d) +
					 omega2*((float)(1./3.)*c  + (float)(5./6.)*d - (float)(1./6.)*e);
}

void weno_minus_reference(const float * const a, const float * const b, const float * const c,
			  const float * const d, const float * const e, float * const out,
			  const int NENTRIES)
{
//#pragma omp for
		for (int i=0; i<NENTRIES; ++i)
			out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
}


// OpenMP SIMD Implementation
void weno_minus_omp(const float * const a, const float * const b, const float * const c, const float * const d, const float * const e, float * const out, const int NENTRIES)
{
    #pragma omp simd
    for (int i=0; i<NENTRIES; ++i) {
        out[i] = weno_minus_core(a[i], b[i], c[i], d[i], e[i]);
    }
}


//SSE Implementation
void weno_minus_sse(const float * const a, const float * const b, const float * const c,const float * const d, const float * const e, float * const out, const int NENTRIES)
{
    //Pre-load constants into registers
    const __m128 c4_3   = _mm_set1_ps(4.0f/3.0f);
    const __m128 c19_3  = _mm_set1_ps(19.0f/3.0f);
    const __m128 c11_3  = _mm_set1_ps(11.0f/3.0f);
    const __m128 c25_3  = _mm_set1_ps(25.0f/3.0f);
    const __m128 c31_3  = _mm_set1_ps(31.0f/3.0f);
    const __m128 c10_3  = _mm_set1_ps(10.0f/3.0f);
    const __m128 c13_3  = _mm_set1_ps(13.0f/3.0f);
    const __m128 c5_3   = _mm_set1_ps(5.0f/3.0f);
    const __m128 eps    = _mm_set1_ps((float)WENOEPS);
    const __m128 val0_1 = _mm_set1_ps(0.1f);
    const __m128 val0_6 = _mm_set1_ps(0.6f);
    const __m128 val0_3 = _mm_set1_ps(0.3f);
    const __m128 c1_0   = _mm_set1_ps(1.0f);
    const __m128 c1_3   = _mm_set1_ps(1.0f/3.0f);
    const __m128 c7_6   = _mm_set1_ps(7.0f/6.0f);
    const __m128 c11_6  = _mm_set1_ps(11.0f/6.0f);
    const __m128 c1_6   = _mm_set1_ps(1.0f/6.0f);
    const __m128 c5_6   = _mm_set1_ps(5.0f/6.0f);
    const __m128 c_neg1_6 = _mm_set1_ps(-1.0f/6.0f); // Pre-calculate -1/6

    //Loop
    for (int i = 0; i < NENTRIES; i += 4) {
        // Load 4 from each array
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vc = _mm_loadu_ps(&c[i]);
        __m128 vd = _mm_loadu_ps(&d[i]);
        __m128 ve = _mm_loadu_ps(&e[i]);

        //IS0
        __m128 term1 = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(va, c4_3), _mm_mul_ps(vb, c19_3)), _mm_mul_ps(vc, c11_3));
        __m128 term2 = _mm_sub_ps(_mm_mul_ps(vb, c25_3), _mm_mul_ps(vc, c31_3));
        __m128 term3 = _mm_mul_ps(vc, c10_3);
        __m128 is0 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(va, term1), _mm_mul_ps(vb, term2)), _mm_mul_ps(vc, term3));

        //IS1
        term1 = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(vb, c4_3), _mm_mul_ps(vc, c13_3)), _mm_mul_ps(vd, c5_3));
        term2 = _mm_sub_ps(_mm_mul_ps(vc, c13_3), _mm_mul_ps(vd, c13_3));
        term3 = _mm_mul_ps(vd, c4_3);
        __m128 is1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vb, term1), _mm_mul_ps(vc, term2)), _mm_mul_ps(vd, term3));

        //IS2
        term1 = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(vc, c10_3), _mm_mul_ps(vd, c31_3)), _mm_mul_ps(ve, c11_3));
        term2 = _mm_sub_ps(_mm_mul_ps(vd, c25_3), _mm_mul_ps(ve, c19_3));
        term3 = _mm_mul_ps(ve, c4_3);
        __m128 is2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vc, term1), _mm_mul_ps(vd, term2)), _mm_mul_ps(ve, term3));

        //ALPHA
        __m128 is0plus = _mm_add_ps(is0, eps);
        __m128 is1plus = _mm_add_ps(is1, eps);
        __m128 is2plus = _mm_add_ps(is2, eps);

        // Alpha = Const / (isX * isX)
        __m128 alpha0 = _mm_div_ps(val0_1, _mm_mul_ps(is0plus, is0plus));
        __m128 alpha1 = _mm_div_ps(val0_6, _mm_mul_ps(is1plus, is1plus));
        __m128 alpha2 = _mm_div_ps(val0_3, _mm_mul_ps(is2plus, is2plus));

        // OMEGA
        __m128 alphasum = _mm_add_ps(_mm_add_ps(alpha0, alpha1), alpha2);
        // inv_alpha = 1.0 / alphasum
        __m128 inv_alpha = _mm_div_ps(c1_0, alphasum); 

        __m128 omega0 = _mm_mul_ps(alpha0, inv_alpha);
        __m128 omega1 = _mm_mul_ps(alpha1, inv_alpha);
        __m128 omega2 = _mm_sub_ps(_mm_sub_ps(c1_0, omega0), omega1);

        //RECONSTRUCTION
        // p0 = (1/3)a - (7/6)b + (11/6)c
        __m128 part0 = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(c1_3, va), _mm_mul_ps(c7_6, vb)), _mm_mul_ps(c11_6, vc));
        
        // p1 = -(1/6)b + (5/6)c + (1/3)d
        __m128 part1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(c_neg1_6, vb), _mm_mul_ps(c5_6, vc)), _mm_mul_ps(c1_3, vd));

        // p2 = (1/3)c + (5/6)d - (1/6)e
        __m128 part2 = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(c1_3, vc), _mm_mul_ps(c5_6, vd)), _mm_mul_ps(c1_6, ve));

        // Result
        __m128 result = _mm_add_ps(_mm_add_ps(_mm_mul_ps(omega0, part0), _mm_mul_ps(omega1, part1)), _mm_mul_ps(omega2, part2));

        _mm_storeu_ps(&out[i], result);
    }
}



//AVX Implementation
void weno_minus_avx(const float * const a, const float * const b, const float * const c, const float * const d, const float * const e, float * const out, const int NENTRIES)
{
    //Constants
    const __m256 c4_3   = _mm256_set1_ps(4.0f/3.0f);
    const __m256 c19_3  = _mm256_set1_ps(19.0f/3.0f);
    const __m256 c11_3  = _mm256_set1_ps(11.0f/3.0f);
    const __m256 c25_3  = _mm256_set1_ps(25.0f/3.0f);
    const __m256 c31_3  = _mm256_set1_ps(31.0f/3.0f);
    const __m256 c10_3  = _mm256_set1_ps(10.0f/3.0f);
    const __m256 c13_3  = _mm256_set1_ps(13.0f/3.0f);
    const __m256 c5_3   = _mm256_set1_ps(5.0f/3.0f);
    const __m256 eps    = _mm256_set1_ps((float)WENOEPS);
    const __m256 val0_1 = _mm256_set1_ps(0.1f);
    const __m256 val0_6 = _mm256_set1_ps(0.6f);
    const __m256 val0_3 = _mm256_set1_ps(0.3f);
    const __m256 c1_0   = _mm256_set1_ps(1.0f);
    const __m256 c1_3   = _mm256_set1_ps(1.0f/3.0f);
    const __m256 c7_6   = _mm256_set1_ps(7.0f/6.0f);
    const __m256 c11_6  = _mm256_set1_ps(11.0f/6.0f);
    const __m256 c1_6   = _mm256_set1_ps(1.0f/6.0f);
    const __m256 c5_6   = _mm256_set1_ps(5.0f/6.0f);
    const __m256 c_neg1_6 = _mm256_set1_ps(-1.0f/6.0f);

    //Loop
    for (int i = 0; i < NENTRIES; i += 8) {
        // Load 8 floats
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        __m256 vd = _mm256_loadu_ps(&d[i]);
        __m256 ve = _mm256_loadu_ps(&e[i]);

        //IS0
        __m256 term1 = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(va, c4_3), _mm256_mul_ps(vb, c19_3)), _mm256_mul_ps(vc, c11_3));
        __m256 term2 = _mm256_sub_ps(_mm256_mul_ps(vb, c25_3), _mm256_mul_ps(vc, c31_3));
        __m256 term3 = _mm256_mul_ps(vc, c10_3);
        __m256 is0 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(va, term1), _mm256_mul_ps(vb, term2)), _mm256_mul_ps(vc, term3));

        //IS1
        term1 = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(vb, c4_3), _mm256_mul_ps(vc, c13_3)), _mm256_mul_ps(vd, c5_3));
        term2 = _mm256_sub_ps(_mm256_mul_ps(vc, c13_3), _mm256_mul_ps(vd, c13_3));
        term3 = _mm256_mul_ps(vd, c4_3);
        __m256 is1 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(vb, term1), _mm256_mul_ps(vc, term2)), _mm256_mul_ps(vd, term3));

        //IS2
        term1 = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(vc, c10_3), _mm256_mul_ps(vd, c31_3)), _mm256_mul_ps(ve, c11_3));
        term2 = _mm256_sub_ps(_mm256_mul_ps(vd, c25_3), _mm256_mul_ps(ve, c19_3));
        term3 = _mm256_mul_ps(ve, c4_3);
        __m256 is2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(vc, term1), _mm256_mul_ps(vd, term2)), _mm256_mul_ps(ve, term3));

        //ALPHA
        __m256 is0plus = _mm256_add_ps(is0, eps);
        __m256 is1plus = _mm256_add_ps(is1, eps);
        __m256 is2plus = _mm256_add_ps(is2, eps);

        __m256 alpha0 = _mm256_div_ps(val0_1, _mm256_mul_ps(is0plus, is0plus));
        __m256 alpha1 = _mm256_div_ps(val0_6, _mm256_mul_ps(is1plus, is1plus));
        __m256 alpha2 = _mm256_div_ps(val0_3, _mm256_mul_ps(is2plus, is2plus));

        //OMEGA
        __m256 alphasum = _mm256_add_ps(_mm256_add_ps(alpha0, alpha1), alpha2);
        __m256 inv_alpha = _mm256_div_ps(c1_0, alphasum);

        __m256 omega0 = _mm256_mul_ps(alpha0, inv_alpha);
        __m256 omega1 = _mm256_mul_ps(alpha1, inv_alpha);
        __m256 omega2 = _mm256_sub_ps(_mm256_sub_ps(c1_0, omega0), omega1);

        //RECONSTRUCTION
        __m256 part0 = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(c1_3, va), _mm256_mul_ps(c7_6, vb)), _mm256_mul_ps(c11_6, vc));
        __m256 part1 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(c_neg1_6, vb), _mm256_mul_ps(c5_6, vc)), _mm256_mul_ps(c1_3, vd));
        __m256 part2 = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(c1_3, vc), _mm256_mul_ps(c5_6, vd)), _mm256_mul_ps(c1_6, ve));

        __m256 result = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(omega0, part0), _mm256_mul_ps(omega1, part1)), _mm256_mul_ps(omega2, part2));

        // Store
        _mm256_storeu_ps(&out[i], result);
    }
}