/*
    We're looking for the formula giving the index
    of the first element to be processed by each thread.
    Given this index, the second can just be accessed by
    adding 1 to it.

    So say thread 0 takes care of elements 0 and 1, we want
    thread 1 to take care of elements 2 and 3. In terms of
    data indices, we need to access element 0, then element 2,
    then element 4. 

    So we're basically doubling the previous index. Thus:

    i = 2 * (blockIdx.x * blockDim.x + threadIdx.x)
*/