/*
    Ex 2: Matrix Vector Multiplication
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__global__ void matrixVecMultiply(float* A, float* B, float* C, int M) {
    int i = threadIdx.x;
    int offset;
    float sum = 0;
    for (int j=0; j<M; j++) {
        offset = i*M + j;
        sum += A[offset] * B[j];
    }
    C[i] = sum;
}

int main(void) {

    // parameters
    int M = 3;
    size_t sizeMatrix = M * M * sizeof(float);
    size_t sizeVec = M * sizeof(float);

    // allocate host matrices
    float* h_A = (float*) malloc(sizeMatrix);
    float* h_B = (float*) malloc(sizeVec);
    float* h_C = (float*) malloc(sizeVec);

    // initialize host matrices
    int i, j, offset;
    int count1, count2 = 0;
    for (i = 0; i <  M; i++) {
        for (j = 0; j < M; j++) {
            offset = i*M + j;
            h_A[offset] = ++count1;
        }
        h_B[i] = ++count2;
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
    int numBlocks = 1;
    matrixVecMultiply<<<numBlocks, numThreads>>>(d_A, d_B, d_C, M);

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
