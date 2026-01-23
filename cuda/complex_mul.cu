#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Ορίζουμε το μέγεθος του πίνακα N (μπορείτε να το αλλάξετε)
#define N 1024
// Ορίζουμε το μέγεθος του Block (συνήθως 16x16 ή 32x32)
#define BLOCK_SIZE 16

// Macro για έλεγχο λαθών CUDA
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA Kernel: Υπολογίζει τα E και F
// Κάθε thread υπολογίζει ένα στοιχείο (row, col) για το E και το F ταυτόχρονα
__global__ void complexMatMul(const float *A, const float *B, const float *C, const float *D, float *E, float *F, int n) {
    
    // Υπολογισμός της γραμμής και της στήλης που αντιστοιχεί στο thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Έλεγχος ορίων (για να μην βγούμε εκτός πίνακα αν το N δεν είναι πολλαπλάσιο του BLOCK_SIZE)
    if (row < n && col < n) {
        float sum_ac_bd = 0.0f; // Για το E = AC - BD
        float sum_ad_bc = 0.0f; // Για το F = AD + BC

        // Εκτέλεση του πολλαπλασιασμού γραμμής-στήλης (dot product)
        for (int k = 0; k < n; k++) {
            float a = A[row * n + k];
            float b = B[row * n + k];
            float c = C[k * n + col];
            float d = D[k * n + col];

            // E = AC - BD
            sum_ac_bd += (a * c) - (b * d);
            
            // F = AD + BC
            sum_ad_bc += (a * d) + (b * c);
        }

        // Αποθήκευση αποτελεσμάτων στην Global Memory
        E[row * n + col] = sum_ac_bd;
        F[row * n + col] = sum_ad_bc;
    }
}

int main() {
    // 1. Δέσμευση μνήμης στον Host (CPU)
    size_t size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_D = (float *)malloc(size);
    float *h_E = (float *)malloc(size); // Αποτέλεσμα Real
    float *h_F = (float *)malloc(size); // Αποτέλεσμα Imaginary

    // 2. Αρχικοποίηση πινάκων με τυχαίες τιμές
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
        h_C[i] = (float)rand() / RAND_MAX;
        h_D[i] = (float)rand() / RAND_MAX;
    }

    printf("Initialization of %dx%d matrices complete.\n", N, N);

    // 3. Δέσμευση μνήμης στο Device (GPU)
    float *d_A, *d_B, *d_C, *d_D, *d_E, *d_F;
    cudaCheckError(cudaMalloc((void **)&d_A, size));
    cudaCheckError(cudaMalloc((void **)&d_B, size));
    cudaCheckError(cudaMalloc((void **)&d_C, size));
    cudaCheckError(cudaMalloc((void **)&d_D, size));
    cudaCheckError(cudaMalloc((void **)&d_E, size));
    cudaCheckError(cudaMalloc((void **)&d_F, size));

    // 4. Αντιγραφή δεδομένων από Host -> Device
    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_D, h_D, size, cudaMemcpyHostToDevice));

    // 5. Ρύθμιση Grid και Block διαστάσεων
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Launching CUDA kernel...\n");
    
    // Μέτρηση χρόνου (προαιρετικά, με CUDA Events)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 6. Κλήση του Kernel
    complexMatMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, d_E, d_F, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Έλεγχος για λάθη κατά την εκτέλεση του kernel
    cudaCheckError(cudaGetLastError());

    // 7. Αντιγραφή αποτελεσμάτων πίσω στον Host (Device -> Host)
    cudaCheckError(cudaMemcpy(h_E, d_E, size, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(h_F, d_F, size, cudaMemcpyDeviceToHost));

    printf("Computation finished in %.2f ms.\n", milliseconds);
    printf("Sample result E[0][0]: %f\n", h_E[0]);
    printf("Sample result F[0][0]: %f\n", h_F[0]);

    // 8. Απελευθέρωση μνήμης
    free(h_A); free(h_B); free(h_C); free(h_D); free(h_E); free(h_F);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_E); cudaFree(d_F);

    return 0;
}
