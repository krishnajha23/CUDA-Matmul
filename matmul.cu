#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 1024

void cpu_matmul(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void init_matrix(float* mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

bool matrices_equal(float* A, float* B, int n, float tol = 1e-3) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(A[i] - B[i]) > tol) {
            printf("Mismatch at index %d: %f vs %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int size = N * N * sizeof(float);
    
    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C_cpu = (float*)malloc(size);
    
    srand(42);
    init_matrix(A, N);
    init_matrix(B, N);
    
    clock_t start = clock();
    cpu_matmul(A, B, C_cpu, N);
    clock_t end = clock();
    
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("CPU time: %.2f ms\n", cpu_time);
    
    free(A); 
    free(B); 
    free(C_cpu);
    return 0;
}
