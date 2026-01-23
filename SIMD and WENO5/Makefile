CC=gcc
CFLAGS=-O2
LIBS=-lm

all: bench

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

clean:
	rm -f bench

