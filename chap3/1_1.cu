/*
    Ex 3.1: Square Matrix Addition

    v1: each thread works on 1 matrix element
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__global__ void matrixMultiply(float* A, float* B, float* C, int M) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x)
    C[i] = A[i] + B[i];
}

int main(void) {

    // parameters
    int M = 10;
    int numElements = M * M;
    size_t size = numElements * sizeof(float);

    // allocate host matrices
    float* h_A = (float*) malloc(size);
    float* h_B = (float*) malloc(size);
    float* h_C = (float*) malloc(size);

    // initialize host matrices
    int i, j, offset;
    for (i = 0; i <  M; i++) {
        for (j = 0; j < M; j++) {
            offset = i*M + j;
            h_A[offset] = 1.;
            h_B[offset] = 1.;
        }
    }

    // allocate device matrices
    cudaMalloc((void**)&d_A, size)
    cudaMalloc((void**)&d_B, size)
    cudaMalloc((void**)&d_C, size)

    // host matrices -> device matrices
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // kernel launch
    int numThreads = M;
    int numBlocks = M;
    matrixMultiply<<<numBlocks, numThreads>>>(d_A, d_B, d_C, M);

    // device matrices -> host matrices
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // print result
    for (i = 0; i <  M; i++)
        for (j = 0; j < M; j++)
            printf("%f ", h_C[i*M + j]);

    // free device and host memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
