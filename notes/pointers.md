# Pointers in C

A pointer is a variable which contains the address in memory of another variable. We can have a pointer to any variable type.

## Address-Of and Dereference Operators

The address of a variable can be obtained by preceding its name with an ampersand sign (&). & is known as the "address-of" operator. For example, `foo = &my_var` assigns the address of variable `my_var` to foo. Let's write some code to see what it would print:

```c
#include <stdio.h>

int main() {

    int var = 20;
    int *foo = &var;

    printf("foo: %p\n", foo); // prints 0x7fff52735a68

    return 0;
}
```
The actual address of a variable in memory cannot be known before runtime. Running this code twice will print out different values for foo.

Pointers are said to "point to" the variable whose address they store. An interesting property of pointers is they they can be used to access the variable they point to directly. This is done by preceding the pointer name with the *dereference oeprator (*)* which can be read as "value pointed to by".

```c
baz = foo;   // baz equal to foo 0x7fff52735a68
baz = *foo;  // baz equal to value pointed to by foo (20)
```
The reference and dereference operators are complementary:

* `&` is the **address-of** operator and can be read as "address of".
* `*` is the **dereference** operator and can be read as "value pointed to by".

To declare a pointer to a variable do:

```c
int *pointer;
```
We must associate a pointer to a particular type. Why? Well, as we just saw, a pointer has the ability to directly refer to the value that it points to (using the dereference operator). As such, a pointer will have different properties if it points to a char, an int or a float. The pointer needs to know how many bytes the data is stored in. Thus, if we increment a pointer by 1, we implicitly increase the pointer by one **block** of memory.

When a pointer is declared it does not point anywhere. We must set it to point somewhere before we use it. Common practice is to assign it the value of `NULL` at instantiation. 

```c
int *pointer = NULL;
```

A pointer to any variable type is an address in memory, which is an integer address. However, a pointer is definitely NOT an integer.

## Pointers and Functions

## Pointers and Arrays

## References

- [C Pointer](https://users.cs.cf.ac.uk/Dave.Marshall/C/node10.html)