#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>

#define MAXVARS 250
#define EPSMIN  (1E-6)
#define TAG_RESULT 100

/* prototype of local optimization routine */
extern void mds(double *startpoint, double *endpoint, int n, double *val,
                double eps, int maxfevals, int maxiter,
                double mu, double theta, double delta,
                int *ni, int *nf, double *xl, double *xr, int *term);

/* global counters */
/* Note: In MPI, this counter is local to each process until summed up */
unsigned long funevals = 0;

/* objective function */
/* Removed the OpenMP atomic pragma as it's not applicable in MPI */
double f(double *x, int n)
{
    double fv = 0.0;

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
    int rank, size;
    double t0_total, t1_total;

    /* Initialize MPI environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    t0_total = MPI_Wtime();

    int nvars = 4;
    int ntrials = 64; // Total number of trials
    int trials_per_process;
    int start_trial;

    double lower[MAXVARS], upper[MAXVARS];
    for (int i = 0; i < MAXVARS; i++) lower[i] = -2.0;
    for (int i = 0; i < MAXVARS; i++) upper[i] = +2.0;

    double eps = EPSMIN;
    int maxfevals = 10000;
    int maxiter = 10000;
    double mu = 1.0, theta = 0.25, delta = 0.25;

    /* Local best (for each worker process) */
    /* Fix: Initialize the array to prevent 'uninitialized' warnings. */
    double local_best_pt[MAXVARS] = {0.0}; 
    double local_best_fx = 1e10;
    int local_best_trial_offset = -1; // Offset from start_trial
    int local_best_nt = -1;
    int local_best_nf = -1;

    /* --- Work Distribution --- */
    trials_per_process = ntrials / size;
    int remainder = ntrials % size;

    if (rank < remainder) {
        trials_per_process++;
        start_trial = rank * trials_per_process;
    } else {
        start_trial = rank * trials_per_process + remainder;
    }
    
    int end_trial = start_trial + trials_per_process;
    
    /* --- Parallel Loop (Each process runs its assigned trials) --- */
    for (int trial = start_trial; trial < end_trial; trial++) {
        double startpt[MAXVARS], endpt[MAXVARS];
        double fx;
        int nt, nf;

        /* per-trial random seed based on global trial index and rank */
        unsigned short seed[3] = {
            (unsigned short)trial,
            (unsigned short)rank,
            (unsigned short)time(NULL)
        };

        for (int i = 0; i < nvars; i++)
            startpt[i] = lower[i] + (upper[i] - lower[i]) * erand48(seed);

        int term = -1;
        mds(startpt, endpt, nvars, &fx, eps, maxfevals, maxiter,
            mu, theta, delta, &nt, &nf, lower, upper, &term);
            
        /* update local best */
        if (fx < local_best_fx) {
            local_best_fx = fx;
            local_best_trial_offset = trial - start_trial;
            local_best_nt = nt;
            local_best_nf = nf;
            for (int i = 0; i < nvars; i++)
                local_best_pt[i] = endpt[i];
        }
    } /* end for trials */

    
    /* --- MPI Master-Worker Aggregation (Rank 0 is the Master) --- */

    // 1. Prepare result buffer: [fx, trial_offset, nt, nf, x[0], x[1], ...]
    int result_size = 4 + nvars; 
    double *send_buffer = (double*)malloc(sizeof(double) * result_size);
    
    send_buffer[0] = local_best_fx;
    send_buffer[1] = (double)(local_best_trial_offset != -1 ? start_trial + local_best_trial_offset : -1); // Global trial index
    send_buffer[2] = (double)local_best_nt;
    send_buffer[3] = (double)local_best_nf;
    for (int i = 0; i < nvars; i++) {
        send_buffer[4 + i] = local_best_pt[i];
    }

    // 2. Aggregate total function evaluations
    unsigned long global_funevals;
    MPI_Reduce(&funevals, &global_funevals, 1, MPI_UNSIGNED_LONG, 
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* MASTER PROCESS (Rank 0) - Receive results from all, find global best */
        /* Initialize the local best point on the master process */
        double global_best_pt[MAXVARS] = {0.0}; 
        double global_best_fx = local_best_fx;
        int global_best_trial = (int)send_buffer[1];
        int global_best_nt = (int)send_buffer[2];
        int global_best_nf = (int)send_buffer[3];
        
        if (global_best_trial != -1) {
             for (int i = 0; i < nvars; i++) 
                global_best_pt[i] = send_buffer[4 + i];
        }

        double *recv_buffer = (double*)malloc(sizeof(double) * result_size);
        
        for (int i = 1; i < size; i++) {
            MPI_Recv(recv_buffer, result_size, MPI_DOUBLE, i, TAG_RESULT, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            double remote_fx = recv_buffer[0];
            int remote_trial = (int)recv_buffer[1];
            
            if (remote_trial != -1 && remote_fx < global_best_fx) {
                global_best_fx = remote_fx;
                global_best_trial = remote_trial;
                global_best_nt = (int)recv_buffer[2];
                global_best_nf = (int)recv_buffer[3];
                for (int j = 0; j < nvars; j++)
                    global_best_pt[j] = recv_buffer[4 + j];
            }
        }
        
        free(recv_buffer);

        t1_total = MPI_Wtime();

        /* results */
        printf("\nFINAL RESULTS (MPI - Master/Worker):\n");
        printf("Number of MPI processes = %d\n", size);
        printf("Elapsed time = %.3lf s\n", t1_total - t0_total);
        printf("Total number of trials = %d\n", ntrials);
        printf("Total number of function evaluations = %ld\n", global_funevals);
        
        if (global_best_trial != -1) {
            printf("Best result at global trial %d used %d iterations and %d calls\n",
                   global_best_trial, global_best_nt, global_best_nf);

            for (int i = 0; i < nvars; i++)
                printf("x[%3d] = %15.7le\n", i, global_best_pt[i]);

            printf("f(x) = %15.7le\n", global_best_fx);
        } else {
            printf("No successful optimization found across all trials.\n");
        }

    } else {
        /* WORKER PROCESSES (Rank > 0) - Send local best result to Master */
        MPI_Send(send_buffer, result_size, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
    }

    free(send_buffer);
    MPI_Finalize();

    return 0;
}
