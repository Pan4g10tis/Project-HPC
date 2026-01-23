#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

#define N 1024
#define BLOCK_SIZE 256 // Για τον kernel άθροισης (1D)

// Macro για έλεγχο λαθών CUDA
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Απλός Kernel για πρόσθεση/αφαίρεση πινάκων
// mode 0: E = E - Temp (Αφαίρεση για το Real part)
// mode 1: F = F + Temp (Πρόσθεση για το Imaginary part)
__global__ void matrixOp(float *Dest, const float *Src, int n, int mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        if (mode == 0)
            Dest[idx] -= Src[idx]; // E = AC - BD
        else
            Dest[idx] += Src[idx]; // F = AD + BC
    }
}

int main() {
    size_t size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_D = (float *)malloc(size);
    float *h_E = (float *)malloc(size);
    float *h_F = (float *)malloc(size);

    // Αρχικοποίηση
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
        h_C[i] = (float)rand() / RAND_MAX;
        h_D[i] = (float)rand() / RAND_MAX;
    }

    // Δέσμευση μνήμης GPU
    float *d_A, *d_B, *d_C, *d_D, *d_E, *d_F, *d_Temp;
    cudaCheckError(cudaMalloc((void **)&d_A, size));
    cudaCheckError(cudaMalloc((void **)&d_B, size));
    cudaCheckError(cudaMalloc((void **)&d_C, size));
    cudaCheckError(cudaMalloc((void **)&d_D, size));
    cudaCheckError(cudaMalloc((void **)&d_E, size));
    cudaCheckError(cudaMalloc((void **)&d_F, size));
    cudaCheckError(cudaMalloc((void **)&d_Temp, size)); // Προσωρινός πίνακας

    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_D, h_D, size, cudaMemcpyHostToDevice));

    // --- cuBLAS Setup ---
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    printf("Starting cuBLAS computation...\n");

    // ΣΗΜΑΝΤΙΚΟ: Η cuBLAS θεωρεί τους πίνακες Column-Major (Fortran style).
    // Η C είναι Row-Major. Για να υπολογίσουμε C = A * B σωστά,
    // ζητάμε από την cuBLAS να υπολογίσει B * A.
    // (Μαθηματική ιδιότητα: (AB)^T = B^T A^T)

    // 1. Υπολογισμός AC -> Αποθήκευση στο E
    // cublasSgemm(handle, OP_N, OP_N, N, N, N, &alpha, C, N, A, N, &beta, E, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_C, N, d_A, N, &beta, d_E, N);

    // 2. Υπολογισμός BD -> Αποθήκευση στο Temp
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_D, N, d_B, N, &beta, d_Temp, N);

    // 3. E = E - Temp (AC - BD)
    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;
    matrixOp<<<blocks, threads>>>(d_E, d_Temp, N, 0); // mode 0: subtraction

    // 4. Υπολογισμός AD -> Αποθήκευση στο F
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_D, N, d_A, N, &beta, d_F, N);

    // 5. Υπολογισμός BC -> Αποθήκευση στο Temp
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_C, N, d_B, N, &beta, d_Temp, N);

    // 6. F = F + Temp (AD + BC)
    matrixOp<<<blocks, threads>>>(d_F, d_Temp, N, 1); // mode 1: addition

    cudaDeviceSynchronize(); // Αναμονή να τελειώσουν όλα

    // Αντιγραφή αποτελεσμάτων πίσω
    cudaCheckError(cudaMemcpy(h_E, d_E, size, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(h_F, d_F, size, cudaMemcpyDeviceToHost));

    printf("Done. Sample E[0][0]: %f\n", h_E[0]);
    printf("Sample F[0][0]: %f\n", h_F[0]);

    // Cleanup
    cublasDestroy(handle);
    free(h_A); free(h_B); free(h_C); free(h_D); free(h_E); free(h_F);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_E); cudaFree(d_F); cudaFree(d_Temp);

    return 0;
}
