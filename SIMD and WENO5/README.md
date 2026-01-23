# HPC Exercise 2: SIMD Optimization of WENO5 Kernel

## Overview
This project focuses on optimizing the **WENO5 (Weighted Essentially Non-Oscillatory)** kernel, a critical component in Computational Fluid Dynamics (CFD). The goal was to accelerate a scalar C implementation using **SIMD (Single Instruction, Multiple Data)** techniques, moving from compiler auto-vectorization to manual assembly intrinsics.

The final optimized kernel achieves a **~12x speedup** over the reference version by utilizing AVX intrinsics.

## Implementations
The repository contains four evolutionary stages of the kernel:

1.  **Reference:** Standard scalar C code (Baseline).
2.  **OpenMP:** Compiler-assisted auto-vectorization using `#pragma omp simd`.
3.  **SSE:** Manual vectorization using 128-bit Intel SSE Intrinsics (4 floats/vector).
4.  **AVX:** Manual vectorization using 256-bit Intel AVX Intrinsics (8 floats/vector).

## Requirements
* **OS:** Linux (or WSL2 on Windows)
* **Compiler:** GCC (with OpenMP support)
* **Hardware:** CPU with support for SSE4.2 and AVX2 instruction sets.

## Build & Run

To compile all versions:  ```make```

To run the automated benchmark suite (executes all versions sequentially): ```make run_all```

To run individual binaries: 
```./bench_ref   # Reference```
```./bench_omp   # OpenMP```
```./bench_sse   # SSE```
```./bench_avx   # AVX```