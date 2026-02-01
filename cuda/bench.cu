#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// --- Ρυθμίσεις ---
#define BLOCK_SIZE 16   // Για τον Naive Kernel
#define OP_THREADS 256  // Για τα Add/Sub kernels

// Macro για έλεγχο λαθών
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// --------------------------------------------------------
// 1. NAIVE KERNEL IMPLEMENTATION
// --------------------------------------------------------
__global__ void complexMatMul_Naive(const float *A, const float *B, const float *C, const float *D, float *E, float *F, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum_ac_bd = 0.0f;
        float sum_ad_bc = 0.0f;

        for (int k = 0; k < n; k++) {
            float a = A[row * n + k];
            float b = B[row * n + k];
            float c = C[k * n + col];
            float d = D[k * n + col];

            sum_ac_bd += (a * c) - (b * d);
            sum_ad_bc += (a * d) + (b * c);
        }
        E[row * n + col] = sum_ac_bd;
        F[row * n + col] = sum_ad_bc;
    }
}

// --------------------------------------------------------
// 2. CUBLAS HELPER KERNELS (Add/Sub)
// --------------------------------------------------------
__global__ void matrixSub(float *Dest, const float *Src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) Dest[idx] -= Src[idx];
}

__global__ void matrixAdd(float *Dest, const float *Src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) Dest[idx] += Src[idx];
}

// --------------------------------------------------------
// BENCHMARK FUNCTIONS
// --------------------------------------------------------

// Συνάρτηση για μέτρηση Naive Kernel
float run_naive_benchmark(int N, float *d_A, float *d_B, float *d_C, float *d_D, float *d_E, float *d_F) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    complexMatMul_Naive<<<blocks, threads>>>(d_A, d_B, d_C, d_D, d_E, d_F, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return milliseconds;
}

// Συνάρτηση για μέτρηση cuBLAS
float run_cublas_benchmark(int N, cublasHandle_t handle, float *d_A, float *d_B, float *d_C, float *d_D, float *d_E, float *d_F, float *d_Temp) {
    float alpha = 1.0f;
    float beta = 0.0f;
    int threads = OP_THREADS;
    int blocks = (N * N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warm-up (προαιρετικό, αλλά καλό για ακρίβεια)
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_D, N, d_A, N, &beta, d_F, N);

    cudaEventRecord(start);

    // ΣΗΜΕΙΩΣΗ: Για να πετύχουμε C = A * B (Row Major) με την cuBLAS (Col Major),
    // δίνουμε τα ορίσματα ανάποδα: cublasSgemm(..., B, ..., A, ...).

    // 1. E = AC
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_C, N, d_A, N, &beta, d_E, N);
    // 2. Temp = BD
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_D, N, d_B, N, &beta, d_Temp, N);
    // 3. E = E - Temp
    matrixSub<<<blocks, threads>>>(d_E, d_Temp, N);

    // 4. F = AD
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_D, N, d_A, N, &beta, d_F, N);
    // 5. Temp = BC
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_C, N, d_B, N, &beta, d_Temp, N);
    // 6. F = F + Temp
    matrixAdd<<<blocks, threads>>>(d_F, d_Temp, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return milliseconds;
}

int main() {
    // Μεγέθη πινάκων προς έλεγχο
    // Μπορείτε να προσθέσετε και το 8192 αν θέλετε, αλλά θα αργήσει λίγο στον Naive kernel
    int sizes[] = {512, 1024, 2048, 4096}; 
    int num_sizes = 4;

    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("========================================================================================\n");
    printf("COMPLEX MATRIX MULTIPLICATION BENCHMARK (Tesla V100)\n");
    printf("========================================================================================\n");
    printf("| %-6s | %-16s | %-16s | %-16s | %-16s |\n", "N", "Naive Time (ms)", "Naive GFLOP/s", "cuBLAS Time (ms)", "cuBLAS GFLOP/s");
    printf("|--------|------------------|------------------|------------------|------------------|\n");

    for (int i = 0; i < num_sizes; i++) {
        int N = sizes[i];
        size_t bytes = N * N * sizeof(float);

        // Δέσμευση μνήμης Device
        float *d_A, *d_B, *d_C, *d_D, *d_E, *d_F, *d_Temp;
        cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes); cudaMalloc(&d_D, bytes);
        cudaMalloc(&d_E, bytes); cudaMalloc(&d_F, bytes);
        cudaMalloc(&d_Temp, bytes);

        // Δεν χρειάζεται αρχικοποίηση με τιμές για το Benchmark ταχύτητας.
        // Αν θέλαμε correctness check, θα έπρεπε να κάνουμε cudaMemcpy εδώ.

        // 1. Εκτέλεση Naive
        float time_naive = run_naive_benchmark(N, d_A, d_B, d_C, d_D, d_E, d_F);

        // 2. Εκτέλεση cuBLAS
        float time_cublas = run_cublas_benchmark(N, handle, d_A, d_B, d_C, d_D, d_E, d_F, d_Temp);

        // Υπολογισμός GFLOPs (8 * N^3 πράξεις)
        double total_flops = 8.0 * (double)N * (double)N * (double)N;
        double gflops_naive = (total_flops / (time_naive / 1000.0)) / 1e9;
        double gflops_cublas = (total_flops / (time_cublas / 1000.0)) / 1e9;

        // Εκτύπωση γραμμής πίνακα
        printf("| %-6d | %-16.2f | %-16.2f | %-16.2f | %-16.2f |\n", 
               N, time_naive, gflops_naive, time_cublas, gflops_cublas);

        // Απελευθέρωση μνήμης για την επόμενη επανάληψη
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
        cudaFree(d_E); cudaFree(d_F); cudaFree(d_Temp);
    }

    printf("========================================================================================\n");

    cublasDestroy(handle);
    return 0;
}
