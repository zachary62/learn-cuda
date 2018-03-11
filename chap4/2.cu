/*
    - vector length N = 2000
    - each thread calculates one output element
    - block size = 512

    grid size (in the x dimension) = ceil(N / 512) = ceil(2000 / 512) = 4.
    Dimensions y and z are set to 1.
*/
