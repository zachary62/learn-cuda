/*
    Constraints
    -----------
    - img size: (400, 900)
    - H = 400 (y), W = 900 (x)
    - 1 thread for each output pixel
    - square blocks
    - compute capability 3.0 => max threads per block = 1024

    if say 32x32:
        - ceil(900 / 32) = 29
        - ceil(400 / 32) = 13

    Thus, grid size is (29, 13, 1).
/*
