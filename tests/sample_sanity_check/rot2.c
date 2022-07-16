// comment

#include <stdio.h>

int* rot_left(int* a, int a_len, int r) {
    // comment
    int loop_start = 0;
    int n_idx = a_len - r;
    int curr = a[0];
    int next = 0;
    for (int i = 0; i < a_len; i++) {
        next = a[n_idx];
        a[n_idx] = curr;
        curr = next;
        n_idx = (a_len + n_idx - r) % a_len;

        if ((n_idx + r) % a_len == loop_start) {
            loop_start += 1;
            n_idx = (a_len + loop_start - r) % a_len;
            curr = a[loop_start];
        }
    }

    return a;
}

int main() {
    int arr[] = {0, 1, 2, 3, 4, 5};
    rot_left(arr, 6, 2);
    for (int i = 0; i < 6; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}