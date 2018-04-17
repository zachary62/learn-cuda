# learning-gpu

Currently learning some GPU CUDA programming.

## Contents

* [vecMultiply](https://github.com/kevinzakka/learning-gpu/blob/master/src/vecMultiply.cu): simple vector multiplication.
* [matMul](https://github.com/kevinzakka/learning-gpu/blob/master/src/matMul.cu): naive matrix multiplication (low CGMA).
* [matMulTiled](https://github.com/kevinzakka/learning-gpu/blob/master/src/matMulTiled.cu): tiled matrix multiplication.
* [color2gray](https://github.com/kevinzakka/learning-gpu/blob/master/src/color2gray.cu): turn an RGB image into a Grayscale image.
* [imgBlur](https://github.com/kevinzakka/learning-gpu/blob/master/src/imgBlur.cu): blur a Grayscale image by averaging pixels in a sliding window.
* [imgBlurPlus](https://github.com/kevinzakka/learning-gpu/blob/master/src/imgBlurPlus.cu): a more efficient blurring kernel that uses shared memory.
* [histogram](https://github.com/kevinzakka/learning-gpu/blob/master/src/histogram.cu): compute the histogram of a 1D sequence.

## References

- [CUDA By Example](https://developer.nvidia.com/cuda-example)
- [Programming Massively Parallel Processors](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-415992-1)
- [CS 179: GPU Programming](http://courses.cms.caltech.edu/cs179/)
