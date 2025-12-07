CC=gcc
MPICC=mpicc
CFLAGS=-Wall -O3
CFLAGS+=-DDEBUG
LDLIBS=-lm

# Build all versions
all: multistart_mds_seq multistart_mds_omp multistart_mds_omp_tasks multistart_mds_mpi

# Sequential version
multistart_mds_seq: multistart_mds_seq.c torczon.c Makefile
	$(CC) $(CFLAGS) -o multistart_mds_seq multistart_mds_seq.c torczon.c $(LDLIBS)

# OpenMP parallel for
multistart_mds_omp: multistart_mds_omp.c torczon.c Makefile
	$(CC) $(CFLAGS) -fopenmp -o multistart_mds_omp multistart_mds_omp.c torczon.c $(LDLIBS)

# OpenMP tasks version
multistart_mds_omp_tasks: multistart_mds_omp_tasks.c torczon.c Makefile
	$(CC) $(CFLAGS) -fopenmp -o multistart_mds_omp_tasks multistart_mds_omp_tasks.c torczon.c $(LDLIBS)

# MPI version
multistart_mds_mpi: multistart_mds_mpi.c torczon.c Makefile
	$(MPICC) $(CFLAGS) -o multistart_mds_mpi multistart_mds_mpi.c torczon.c $(LDLIBS)

clean:
	rm -f multistart_mds_seq multistart_mds_omp multistart_mds_omp_tasks multistart_mds_mpi

