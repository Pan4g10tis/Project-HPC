COMPILATION
CUDA
nvcc -o complex_mul complex_mul.cu

cuBLAS
nvcc -o complex_cublas complex_cublas.cu -lcublas
