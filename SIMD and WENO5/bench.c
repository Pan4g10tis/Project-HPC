#include <stdio.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifndef WENOEPS
#define WENOEPS 1.e-6
#endif

#include "weno.h"

float * myalloc(const int NENTRIES, const int verbose )
{
	const int initialize = 1;
	enum { alignment_bytes = 32 } ;
	float * tmp = NULL;

	const int result = posix_memalign((void **)&tmp, alignment_bytes, sizeof(float) * NENTRIES);
	assert(result == 0);

	if (initialize)
	{
		for(int i=0; i<NENTRIES; ++i)
			tmp[i] = drand48();

		if (verbose)
		{
			for(int i=0; i<NENTRIES; ++i)
				printf("tmp[%d] = %f\n", i, tmp[i]);
			printf("==============\n");
		}
	}
	return tmp;
}

double get_wtime()
{
	struct timeval t;
	gettimeofday(&t,  NULL);
	return t.tv_sec + t.tv_usec*1e-6;
}

void check_error(const double tol, float ref[], float val[], const int N)
{
	static const int verbose = 0;

	for(int i=0; i<N; ++i)
	{
		assert(!isnan(ref[i]));
		assert(!isnan(val[i]));

		const double err = ref[i] - val[i];
		const double relerr = err/fmaxf(FLT_EPSILON, fmaxf(fabs(val[i]), fabs(ref[i])));

		if (verbose) printf("+%1.1e,", relerr);

		if (fabs(relerr) >= tol && fabs(err) >= tol)
			printf("\n%d: %e %e -> %e %e\n", i, ref[i], val[i], err, relerr);

		assert(fabs(relerr) < tol || fabs(err) < tol);
	}

	if (verbose) printf("\t");
}


void benchmark(int argc, char *argv[], const int NENTRIES_, const int NTIMES, const int verbose, char *benchmark_name)
{
	const int NENTRIES = 4 * (NENTRIES_ / 4);

	printf("nentries set to %e\n", (float)NENTRIES);

	float * const a = myalloc(NENTRIES, verbose);
	float * const b = myalloc(NENTRIES, verbose);
	float * const c = myalloc(NENTRIES, verbose);
	float * const d = myalloc(NENTRIES, verbose);
	float * const e = myalloc(NENTRIES, verbose);
	float * const f = myalloc(NENTRIES, verbose);
	float * const gold = myalloc(NENTRIES, verbose);
	float * const result = myalloc(NENTRIES, verbose);

	weno_minus_reference(a, b, c, d, e, gold, NENTRIES);
	weno_minus_reference(a, b, c, d, e, result, NENTRIES);

	const double tol = 1e-5;
	printf("minus: verifying accuracy with tolerance %.5e...", tol);
	check_error(tol, gold, result, NENTRIES);
	printf("passed!\n");

	free(a);
	free(b);
	free(c);
	free(d);
	free(e);
	free(gold);
	free(result);
}

int main (int argc, char *  argv[])
{
	printf("Hello, weno benchmark!\n");
	const int debug = 1;

	if (debug)
	{
		benchmark(argc, argv, 4, 1, 1, "debug");
		return 0;
	}

	/* performance on cache hits */
	{
		const double desired_kb =  16 * 4 * 0.5; /* we want to fill 50% of the dcache */
		const int nentries =  16 * (int)(pow(32 + 6, 2) * 4);//floor(desired_kb * 1024. / 7 / sizeof(float));
		const int ntimes = (int)floor(2. / (1e-7 * nentries));

		for(int i=0; i<4; ++i)
		{
			printf("*************** PEAK-LIKE BENCHMARK (RUN %d) **************************\n", i);
			benchmark(argc, argv, nentries, ntimes, 0, "cache");
		}
	}

	/* performance on data streams */
	{
		const double desired_mb =  128 * 4;
		const int nentries =  (int)floor(desired_mb * 1024. * 1024. / 7 / sizeof(float));

		for(int i=0; i<4; ++i)
		{
			printf("*************** STREAM-LIKE BENCHMARK (RUN %d) **************************\n", i);
			benchmark(argc, argv, nentries, 1, 0, "stream");
		}
	}

    return 0;
}
