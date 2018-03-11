/*
    4 threads blocks of size 512. Thus, a total of 4*512=2,048 threads.
    Unused threads = 2,048 - N = 2,048 - 2,000 = 48.

    48 threads will have control divergence.
*/
