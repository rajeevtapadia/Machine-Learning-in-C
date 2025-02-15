#include <assert.h>
#include <stdio.h>
#define NN_IMPLEMENTATION
#include "nn.h"

// XOR
float xor_df[] = {0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0};

void print_predictions(NN nn) {
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            nn.as[0].es[0] = i;
            nn.as[0].es[1] = j;
            nn_forward(nn);
            printf("%d XOR %d %f\n", i, j, NN_OUTPUT(nn).es[0]);
        }
    }
}

int main() {
    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARR_LEN(arch));
    NN diff = nn_alloc(arch, ARR_LEN(arch));

    nn_rand(nn, 0, 1);

    size_t stride = 3;
    size_t row_count = ARR_LEN(xor_df) / 3;
    Matrix ti = {.rows = row_count, .cols = 2, .stride = stride, .es = xor_df};
    Matrix to = {.rows = row_count, .cols = 1, .stride = stride, .es = xor_df + 2};
    float eps = 1e-1, rate = 1;
    // NN_PRINT(nn);
    printf("err: %f\n", nn_rms_error(nn, ti, to));
    for (int i = 0; i < 100*100; i++) {
        nn_train(nn, diff, eps, ti, to);
        nn_apply_diff(nn, diff, rate);
    }
    printf("err: %f\n", nn_rms_error(nn, ti, to));
    print_predictions(nn);
    return 0;
}
