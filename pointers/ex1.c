#include <stdio.h>

int main(void)
{
    int j, k;
    int *ptr;

    j = 1;
    k = 2;
    ptr = &k;

    printf("j has the value %d and is stored at %p\n", j, &j);
    printf("k has the value %d and is stored at %p\n", k, &k);
    printf("ptr has the value %p and is stored at %p\n", ptr, &ptr);
    printf("The value of the integer pointed to by ptr is %d\n", *ptr);

    return 0;

    /*
        Conclusion:
        - we can get the address of a variable by preceding its name
          with the unary & operator.
        - we can dereference a pointer, i.e. refer to the value that
          it is pointing to by using the unary operator *.
    */
}