#include <stdio.h>

int* rot_left(int* a, int len_a, int rot_count) {
    int curr_val = a[0];
    int next_val = 0;
    int next_idx = len_a - rot_count;

    int loop_start = 0;
    for (int i = 0; i < len_a; i++) {
        next_val = a[next_idx];
        a[next_idx] = curr_val;
        next_idx = (len_a + next_idx - rot_count) % len_a;
        curr_val = next_val;

        if ((next_idx + rot_count) % len_a == loop_start) {
            loop_start += 1;
            next_idx = (len_a + loop_start - rot_count) % len_a;
            curr_val = a[loop_start];
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