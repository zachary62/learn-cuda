/*
    SM max threads = 1536
    SM max blocks = 4

    - 128 threads per block: 4*128=512 <= 1536.
    - 256 threads per block: 4*256=1,024 <= 1536.
    - 512 threads per block: 4*512=2,048 > 1536 => 3*512=1,536 <= 1536.
    - 1024 threads per block: 4*1024=4,096 > 1536 => 1*1024=1,024 <= 1536.

    Configuration 3 (512 threads per block) is optimal since we totally occupy
    the SM; the other configurations underutilize the SM.
*/
