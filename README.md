# learning-gpu

Currently learning some GPU CUDA programming.

## Contents

* [vecMultiply](./src/vecMultiply.cu): simple vector multiplication.
* [matMul](./src/matMul.cu): naive matrix multiplication (low CGMA).
* [matMulTiled](./src/matMulTiled.cu): tiled matrix multiplication.
* [color2gray](./src/color2gray.cu): turn an RGB image into a Grayscale image.
* [imgBlur](./src/imgBlur.cu): blur a Grayscale image by averaging pixels in a sliding window.
* [imgBlurPlus](./src/imgBlurPlus.cu): a more efficient blurring kernel that uses shared memory.
* [histogram](./src/histogram.cu): compute the histogram of a 1D sequence.
* [convolution](./src/tiledConv.cu): apply a tiled 2D convolution on an RGB image.

## References

- [CUDA By Example](https://developer.nvidia.com/cuda-example)
- [Programming Massively Parallel Processors](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-415992-1)
- [CS 179: GPU Programming](http://courses.cms.caltech.edu/cs179/)
