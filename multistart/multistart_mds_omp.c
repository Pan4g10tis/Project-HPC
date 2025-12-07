#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#define MAXVARS		(250)	/* max # of variables	     */
#define EPSMIN		(1E-6)	/* ending value of stepsize  */

/* prototype of local optimization routine, code available in torczon.c */
extern void mds(double *startpoint, double *endpoint, int n, double *val, double eps, int maxfevals, int maxiter,
         double mu, double theta, double delta, int *ni, int *nf, double *xl, double *xr, int *term);


/* global variables */
unsigned long funevals = 0;

/* Rosenbrock classic parabolic valley ("banana") function */
double f(double *x, int n)
{
    double fv;
    int i;

    funevals++;
    fv = 0.0;
    for (i=0; i<n-1; i++)   /* rosenbrock */
        fv = fv + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);

		usleep(100);	/* do not remove, introduces some artificial work */

    return fv;
}


double get_wtime(void)
{
    struct timeval t;

    gettimeofday(&t, NULL);

    return (double)t.tv_sec + (double)t.tv_usec*1.0e-6;
}


int main(int argc, char *argv[])
{
    /* problem parameters */
    int nvars = 4;      /* number of variables (problem dimension) */
    int ntrials = 64;   /* number of trials */
    double lower[MAXVARS], upper[MAXVARS]; /* lower and upper bounds */

    /* mds parameters */
    double eps = EPSMIN;
    int maxfevals = 10000;
    int maxiter = 10000;
    double mu = 1.0;
    double theta = 0.25;
    double delta = 0.25;

    /* information about the best point found by multistart */
    double best_pt[MAXVARS];
    double best_fx = 1e10;
    int best_trial = -1;
    int best_nt = -1;
    int best_nf = -1;

    /* initialization of bounds */
    for (int i = 0; i < MAXVARS; i++) lower[i] = -2.0;
    for (int i = 0; i < MAXVARS; i++) upper[i] = +2.0;

    double t0 = get_wtime();

#pragma omp parallel
    {
        /* thread-local best */
        double t_best_pt[MAXVARS];
        double t_best_fx = 1e10;
        int    t_best_trial = -1;
        int    t_best_nt = -1;
        int    t_best_nf = -1;

        /* thread-private working arrays */
        double startpt[MAXVARS], endpt[MAXVARS];
        int nt, nf;

#pragma omp for schedule(dynamic)
        for (int trial = 0; trial < ntrials; trial++) {

            /* thread-safe random seed */
            unsigned short seed[3] = { (unsigned short)trial,
                                       (unsigned short)omp_get_thread_num(),
                                       (unsigned short)time(NULL) };

            /* random starting point */
            for (int i = 0; i < nvars; i++) {
                double r = erand48(seed);
                startpt[i] = lower[i] + (upper[i]-lower[i]) * r;
            }

            double fx = 0.0;
            int term = -1;

            /* local optimization */
            mds(startpt, endpt, nvars, &fx, eps, maxfevals, maxiter,
                mu, theta, delta, &nt, &nf, lower, upper, &term);

            /* update thread-local best */
            if (fx < t_best_fx) {
                t_best_fx    = fx;
                t_best_trial = trial;
                t_best_nt    = nt;
                t_best_nf    = nf;
                for (int i = 0; i < nvars; i++)
                    t_best_pt[i] = endpt[i];
            }
        }

        /* merge thread-local best into global best */
#pragma omp critical
        {
            if (t_best_fx < best_fx) {
                best_fx    = t_best_fx;
                best_trial = t_best_trial;
                best_nt    = t_best_nt;
                best_nf    = t_best_nf;
                for (int i = 0; i < nvars; i++)
                    best_pt[i] = t_best_pt[i];
            }
        }
    }

    double t1 = get_wtime();

    printf("\n\nFINAL RESULTS:\n");
    printf("Elapsed time = %.3lf s\n", t1 - t0);
    printf("Total number of trials = %d\n", ntrials);
    printf("Total number of function evaluations = %ld\n", funevals);
    printf("Best result at trial %d used %d iterations, %d function calls and returned\n",
           best_trial, best_nt, best_nf);

    for (int i = 0; i < nvars; i++) {
        printf("x[%3d] = %15.7le \n", i, best_pt[i]);
    }
    printf("f(x) = %15.7le\n", best_fx);

    return 0;
}

