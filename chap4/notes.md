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

The choice of 1D, 2D, or 3D thread organizations is usually based on the nature of the data. For example, grayscale images are a 2D array of pixels `(H, W)` while RGB images are a 3D array of pixels `(H, W, C)`.

It is often convenient to use a 2D grid that consists of 2D blocks to process the pixels in a picture. For example, consider an image of size `76x62`. Assume that we decided to use a `16x16` block, with 16 threads in the x direction and 16 threads in the y direction. To process such an image, we would need `ceil(76  / 16) = 5` blocks in the x dimension and `ceil(62 / 16) = 4` blocks in the y dimension. This results in `5*4=20` blocks for a total of `20*16*16=5,120` threads. However, there are only 4,712 pixels in the image, so we would have an if statement to disable the extra threads from doing work.

Here's some code to launch a kernel to process an image of height `H` and width `W`. Note that the number of rows corresponds to the height (the y direction) and that the number of columns corresponds to the width (the x direction). In this example, we assume for simplicity that the dimensions of the blocks are fixed at `16x16`.

```c
dim3 dimBlock(ceil(W/16.0), ceil(H/16.0), 1);
dim3 dimGrid(16, 16, 1);
processImg<<<dimGrid, dimBlock>>>(...);
```
Now that we've seen how to map the 2D grids and threads to the image, let's write a kernel that scales each value in an image by 2. First, let's examine how we can compute the global coordinates of a thread in a grid.

<p align="center">
 <img src="../assets/grid-img.png" alt="Drawing", width=48%>
</p>

The row of this tread can be calculated using the y dimension and the column can be calculate using the x dimension. The pink thread is located in `row=1, col=2`.

- row: We've consumed 4 threads since we're in row 1 plus an extra 1 since we're at row 1 inside the block. Thus the index of the row is `4+1=5`. Generalizing from this example, we can write `row = (blockIdx.y * blockDim.y) + threadIdx.y`.
- col: We've consumed `4*2=8` threads since we're in col 2 plus an extra 2 since we're at col 2 inside the block. The the index of the col is `8+2=10`. Generalizing from this example, we can write `col = (blockIdx.x * blockDim.x) + threadIdx.x`.

Once the global coordinates of a thread within a grid are obtained, we linearize them to access the flattened data elements in row-major format. Thus, our image scaling kernel can now be completed:

```c
__global__ void imgScaler(float* imgIn, float* imgOut, int H, int W) {
    // compute global thread coordinates
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    // flatten coordinates
    int offset = row * W + col;

    if ((row < H) && (col < W)) {
        imgOut[offset] = 2 * imgIn[offset];
    }
}
```
How would things have changed had we used a different set of ECPs? Suppose we had fixed the number of threads in a block to be 256 in 1 dimension. Then, we would have had 1D blocks within a 2D grid `dim3 dimBlock(ceil(W/256.0), ceil(H/256.0), 1)`. The `col` global coordinate would have stayed the same, but `row` would have collapsed to `blockIdx.y`. We don't even have to instantiate a 2D grid. We could have had `dimBlock = 256` and `dimGrid = ceil((H * W) / 256.0)`. Then, the offset would be exactly as in our vector example `offset = blockIdx.x * blockDim.x + threadIdx.x`. This shows you how manipulating the ECPs can lead to different indexing in the kernel.


