/*
    - multiply 1,024 times 1,024 matrices using tiled mm
    - 32x32 thread blocks
    - 512 threads per block and up to 8 blocks per SM

    With 32x32 thread blocks, there is a total of 1,024 threads per block. This
    violates the device constraint of 512 threads per block.
*/
