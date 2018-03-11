/*
    We have a block of 8 threads with the following execution times:

    2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9

    after which they reach a barrier and wait. We need to calculate the
    percentage of the threads’ summed-up execution times that is spent waiting
    for the barrier.

    The threads execute in parallel. Thus, the total time spent is equal to the
    time taken by the slowest thread. In our case, thread 3 takes the longest
    (3.0 microseconds).

    Thus, if thread 1 executes in 2 seconds, then it waits 3-2=1 second at the
    barrier. Let's calculate all the waiting times:

    1.0, 0.7, 0.0, 0.2, 0.6, 1.1, 0.4, 0.1

    The sum of the waiting time is 4.1 and the sum of the execution times
    is 19.9 giving a percentage of 20.6%.

    Note: I don't like this question. I think a more important metric would
    be how much does each thread wait on average, or what percent of each
    thread's execution time is spent waiting at the barrier.

    Let me go ahead and answer that. I have the following:

    - waiting:   1.0, 0.7, 0.0, 0.2, 0.6, 1.1, 0.4, 0.1
    - execution: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9
    - ratio: 0.5, 0.304, 0.0, 0.071, 0.25, 0.578, 0.153, 0.034

    Thus, the average of the ratio is 0.236%. This means that on average,
    each thread spends 24% of its execution time waiting at the barrier.
*/
