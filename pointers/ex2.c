#include <stdio.h>

int main(void) {

    int array[] = {1, 23, 17, 4, -5, 100};
    int *ptr;

    int i;
    ptr = array;

    for (i=0; i<6; i++) {
        printf("array[%d] = %d      ", i, array[i]);
        printf("ptr + %d = %d\n", i, *(ptr++));
    }

    return 0;

    /*
        Conclusion: the name of the array is
        the address of the first element in the
        array, i.e. the name of an array is a pointer
        to the first element in that array.
    */
}