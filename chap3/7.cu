/*
    num_elements = 2000;

    threads_per_block = 512;

    blocks_per_grid = ceil(num_elements / (float)threads_per_block);

    total_num_threads = blocks_per_grid * threads_per_block

    We get 4 blocks and thus 2,048 total threads. With an if condition
    (if i < num_elements), we can disable the last 48 threads.
*/