#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>

#define MAXVARS 250
#define EPSMIN  (1E-6)

/* prototype of local optimization routine */
extern void mds(double *startpoint, double *endpoint, int n, double *val,
                double eps, int maxfevals, int maxiter,
                double mu, double theta, double delta,
                int *ni, int *nf, double *xl, double *xr, int *term);

/* global counters */
unsigned long funevals = 0;

/* objective function */
double f(double *x, int n)
{
    double fv = 0.0;

#pragma omp atomic
    funevals++;

    for (int i = 0; i < n - 1; i++)
        fv += 100.0 * pow((x[i+1] - x[i]*x[i]), 2) + pow((x[i] - 1.0), 2);

    usleep(100); /* artificial work */

    return fv;
}

double get_wtime(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}

int main(int argc, char *argv[])
{
    int nvars = 4;
    int ntrials = 64;

    double lower[MAXVARS], upper[MAXVARS];
    for (int i = 0; i < MAXVARS; i++) lower[i] = -2.0;
    for (int i = 0; i < MAXVARS; i++) upper[i] = +2.0;

    double eps = EPSMIN;
    int maxfevals = 10000;
    int maxiter = 10000;
    double mu = 1.0, theta = 0.25, delta = 0.25;

    /* global best */
    double best_pt[MAXVARS];
    double best_fx = 1e10;
    int best_trial = -1;
    int best_nt = -1;
    int best_nf = -1;

    double t0 = get_wtime();

#pragma omp parallel
    {
#pragma omp single
        {
#pragma omp taskgroup
            {
                for (int trial = 0; trial < ntrials; trial++) {

#pragma omp task firstprivate(trial) shared(best_pt, best_fx, best_trial, best_nt, best_nf)
                    {
                        double startpt[MAXVARS], endpt[MAXVARS];
                        double fx;
                        int nt, nf;

                        /* per-task random seed */
                        unsigned short seed[3] = {
                            (unsigned short)trial,
                            (unsigned short)omp_get_thread_num(),
                            (unsigned short)time(NULL)
                        };

                        for (int i = 0; i < nvars; i++)
                            startpt[i] = lower[i] + (upper[i] - lower[i]) * erand48(seed);

                        int term = -1;
                        mds(startpt, endpt, nvars, &fx, eps, maxfevals, maxiter,
                            mu, theta, delta, &nt, &nf, lower, upper, &term);
                            
                        /* update global best safely */
#pragma omp critical
                        {
                            if (fx < best_fx) {
                                best_fx = fx;
                                best_trial = trial;
                                best_nt = nt;
                                best_nf = nf;
                                for (int i = 0; i < nvars; i++)
                                    best_pt[i] = endpt[i];
                            }
                        }
                    }
                } /* end for */
            } /* end taskgroup */
        } /* end single */
    } /* end parallel */

    double t1 = get_wtime();

    /* results */
    printf("\nFINAL RESULTS (OpenMP Tasks):\n");
    printf("Elapsed time = %.3lf s\n", t1 - t0);
    printf("Total number of trials = %d\n", ntrials);
    printf("Total number of function evaluations = %ld\n", funevals);
    printf("Best result at trial %d used %d iterations and %d calls\n",
           best_trial, best_nt, best_nf);

    for (int i = 0; i < nvars; i++)
        printf("x[%3d] = %15.7le\n", i, best_pt[i]);

    printf("f(x) = %15.7le\n", best_fx);

    return 0;
}


