#include <stdio.h>
#include <stdlib.h>

/*
    Initializing a rectangular matrix of ints.
*/

int main(void) {

    // parameters
    int num_rows = 3;
    int num_cols = 4;
    int num_elems = num_rows * num_cols;
    size_t size = num_elems * sizeof(int);

    // allocate host matrices
    int* A = (int*) malloc(size);

    int i, j, count = 0;
    int offset;
    for (i = 0; i <  num_rows; i++) {
        for (j = 0; j < num_cols; j++) {
            offset = i*num_cols + j;
            A[offset] = ++count;
        }
    }

    for (i = 0; i < num_rows; i++)
        for (j = 0; j < num_cols; j++)
            printf("%d ", A[i*num_cols + j]);

    return 0;
}