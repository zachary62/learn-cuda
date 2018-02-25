# Data-Parallel Execution Model

We'll be delving into the details of the organization, resource assignment, synchronization, and scheduling of threads in a grid.

## CUDA Thread Organization

All CUDA threads in a grid execute the same kernel function and they rely on special variables to distinguish themselves from each other and to identify the appropriate portion of the data to process. Think of the kernel function as specifying the C statements that are executed by each individual thread at runtime.

All threads in a block share the same block index, accessed via `blockIdx`. Additionally, each thread in a block has a unique index, accessed via `threadIdx`. Thus, inside this two-level hierarchy, a thread has a tuple of unique coordinates `(blockIdx, threadIdx)`. The execution configuration parameters (ECPs) in a kernel launch specify the grid size `gridDim` (i.e. the number of blocks in a grid) and the block size `blockDim` (i.e. the number of threads in a block).

In general, a grid is a 3D array of blocks, and each block is a 3D array of threads. We can choose to use fewer dimensions by setting unused dimensions to 1. At kernel launch, we specify 2 parameters enclosed within triple signs `<<<param1, param2>>>`. The first ECP specifies the dimensions of the grid in number of blocks. The second ECP specifies the dimensions of each block in number of threads. Each such parameter is of type `dim3`, which is a C struct with three unsigned integer fields: `x`, `y`, and `z`.

For 1D and 2D grids and blocks, the unused dimensions should be set to 1 for clarity. However, for convenience, CUDA C lets us use plain variables or direct mathematical expressions to specify ECPs for 1D grids. For example, suppose we would like to launch our vector addition kernel `vecAdd` with a set number of threads per block equal to 256. In this way, the number of blocks (i.e. gridDim) will vary with the size of the input vectors so that the grid will have enough threads to cover all vector elements. We can do this in 2 ways:

```c
// method 1
int threadsPerBlock = 256;
dim3 gridDim(ceil(n / (float)threadsPerBlock), 1, 1);
dim3 blockDim(threadsPerBlock, 1, 1);
vecAdd<<<gridDim, blockDim>>>(...);

// method 2
int threadsPerBlock = 256;
int blocksPerGrid = ceil(n / (float)threadsPerBlock);
vecAdd<<<blocksPerGrid, threadsPerBlock>>>(...);
```
Obviously, hardware constraints impose limits on the dimensions of the grid and the blocks. They are:

- `gridDim`: each dimension can vary between 1 and 65,536 (this number will vary with newer GPUs).
- `blockDim`: maximum of 1,024 threads, i.e the product of all dimensions cannot exceed 1,024.
    - allowed ex: `(512, 1, 1)`, `(8, 16, 4)`, `(32, 16, 2)`
    - illegel ex: `(32, 32, 2)`

## Mapping Threads to Multidimensional Data

