#pragma once

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
