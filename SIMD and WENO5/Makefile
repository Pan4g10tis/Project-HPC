CC = gcc
CFLAGS = -O3 -march=native -Wall -D_GNU_SOURCE
LIBS = -lm

all: bench_ref bench_omp bench_sse bench_avx

# Reference
bench_ref: bench.c weno.h
	$(CC) $(CFLAGS) -DVERSION_REF -o bench_ref bench.c $(LIBS)

# OpenMP SIMD
bench_omp: bench.c weno.h
	$(CC) $(CFLAGS) -fopenmp -DVERSION_OMP -o bench_omp bench.c $(LIBS)

# SSE Intrinsics
bench_sse: bench.c weno.h
	$(CC) $(CFLAGS) -DVERSION_SSE -o bench_sse bench.c $(LIBS)

# AVX Intrinsics
bench_avx: bench.c weno.h
	$(CC) $(CFLAGS) -mavx -DVERSION_AVX -o bench_avx bench.c $(LIBS)

clean:
	rm -f bench_ref bench_omp bench_sse bench_avx