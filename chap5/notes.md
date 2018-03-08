# CUDA Memories

Simple kernels like the ones we've been writing will only achieve minimal speedup. This is because they access their data from global memory, which is typically implemented with dynamic random access memory (DRAM). DRAM tends to have long access latencies and finite access bandwidth, which contribute to traffic congestion, ultimately rendering some SMs idle. In this chapter, we'll learn about other memories we can leverage to boost the execution efficiency of our kernels.

## Memory Access Efficiency

An important metric for quantifying the efficiency of a CUDA kernel is the **Compute to Global Memory Access** ratio (CGMA). It is defined as the number of floating-point calculations performed over the number of global memory accesses performed within a region of a CUDA program. For example, in the matrix multiplication kernel we wrote in the previous chapter, the compute-heavy part of our kernel was the inner for-loop which computed the dot product of row i of A and column j of B:

```c
for (int k = 0; k < M; k++) {
    cumSum += A[row*M + k] * B[k*M + col];
}
```
In this foor loop, there are 2 FLOPS performed (1 multiplication and 1 addition) for every 2 global memory accesses (1 for A and 1 for B). Thus, `CGMA=1/1=1.0`.
