/*
    The answer depends on multiple factors.

    If the data is aligned in the x dimension (i.e. 32 consecutive threads)
    then a warp will be loaded and the __syncthreads() can be left out.

    However, the warp size (32) is hardware dependent and can change across
    devices and in the future. This means that the code, while able to run now,
    might not be able to scale to future generations.
*/
