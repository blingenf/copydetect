#include <stdio.h>

int add_num(int a, int b) {
    return a + b;
}

int main(int argc, char* argv[]) {
    int a = 2;
    int b = 2;
    int result = add_num(a, b);
    printf("%d + %d = 5\n", a, b);

    return 0;
}
