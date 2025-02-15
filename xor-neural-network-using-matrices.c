#include <stdlib.h>
#define NN_IMPLEMENTATION
#include "nn.h"

typedef struct {
    Matrix a0;
    Matrix w1, b1, a1;
    Matrix w2, b2, a2;
} model;

// XOR
float xor_df[] = {0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0};

model xor_alloc() {
    model m;
    m.a0 = mat_alloc(1, 2);

    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);
    m.a1 = mat_alloc(1, 2);

    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);
    m.a2 = mat_alloc(1, 1);
    return m;
}

void forward(model m) {
    // pass x from first layer
    mat_mult(m.a1, m.a0, m.w1);
    mat_add(m.a1, m.b1);
    mat_sig(m.a1);

    // pass x from second layer
    mat_mult(m.a2, m.a1, m.w2);
    mat_add(m.a2, m.b2);
    mat_sig(m.a2);
    // return MAT_AT(m.a2, 0, 0);
}

void print_predictions(model m) {
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MAT_AT(m.a0, 0, 0) = i;
            MAT_AT(m.a0, 0, 1) = j;
            forward(m);
            float y = *m.a2.es;
            printf("%zu XOR %zu = %f\n", i, j, y);
        }
    }
}

float rms_error(model m, Matrix ti, Matrix to) {
    assert(ti.rows == to.rows);
    assert(m.a2.cols == to.cols);
    size_t input_row_count = ti.rows;
    size_t output_col_count = to.cols;

    float error = 0.f;

    for (size_t i = 0; i < input_row_count; i++) {
        Matrix x = mat_get_row(ti, i);
        Matrix y = mat_get_row(to, i);

        m.a0 = x;
        forward(m);
        for (size_t j = 0; j < output_col_count; j++) {
            float diff = MAT_AT(m.a2, i, j) - MAT_AT(y, i, j);
            error += diff * diff;
        }
    }
    return error / input_row_count;
}

void apply_diff(model m, model diff, float rate) {
    for (int i = 0; i < m.w1.rows; i++) {
        for (int j = 0; j < m.w1.cols; j++) {
            MAT_AT(m.w1, i, j) -= rate * MAT_AT(diff.w1, i, j);
        }
    }

    for (int i = 0; i < m.w2.rows; i++) {
        for (int j = 0; j < m.w2.cols; j++) {
            MAT_AT(m.w2, i, j) -= rate * MAT_AT(diff.w2, i, j);
        }
    }

    for (int i = 0; i < m.b1.rows; i++) {
        for (int j = 0; j < m.b1.cols; j++) {
            MAT_AT(m.b1, i, j) -= rate * MAT_AT(diff.b1, i, j);
        }
    }
    for (int i = 0; i < m.b2.rows; i++) {
        for (int j = 0; j < m.b2.cols; j++) {
            MAT_AT(m.b2, i, j) -= rate * MAT_AT(diff.b2, i, j);
        }
    }
}

void train(model m, Matrix ti, Matrix to) {
    model diff = xor_alloc();
    float eps = 1e-2;
    float rate = 1e-2;
    for (int itr = 0; itr < 100*1000; itr++) {
        float init_cost = rms_error(m, ti, to);
        float saved = 0;
        // for w1
        for (int i = 0; i < m.w1.rows; i++) {
            for (int j = 0; j < m.w1.cols; j++) {
                // save
                saved = MAT_AT(m.w1, i, j);
                // wiggle
                MAT_AT(m.w1, i, j) += eps;
                // set in diff
                MAT_AT(diff.w1, i, j) = (rms_error(m, ti, to) - init_cost) / eps;
                // restore
                MAT_AT(m.w1, i, j) = saved;
            }
        }

        // for w2
        for (int i = 0; i < m.w2.rows; i++) {
            for (int j = 0; j < m.w2.cols; j++) {
                // save
                saved = MAT_AT(m.w2, i, j);
                // wiggle
                MAT_AT(m.w2, i, j) += eps;
                // set in diff
                MAT_AT(diff.w2, i, j) = (rms_error(m, ti, to) - init_cost) / eps;
                // restore
                MAT_AT(m.w2, i, j) = saved;
            }
        }

        for (int i = 0; i < m.b1.rows; i++) {
            for (int j = 0; j < m.b1.cols; j++) {
                // save
                saved = MAT_AT(m.b1, i, j);
                // wiggle
                MAT_AT(m.b1, i, j) += eps;
                // set in diff
                MAT_AT(diff.b1, i, j) = (rms_error(m, ti, to) - init_cost) / eps;
                // restore
                MAT_AT(m.b1, i, j) = saved;
            }
        }

        for (int i = 0; i < m.b2.rows; i++) {
            for (int j = 0; j < m.b2.cols; j++) {
                // save
                saved = MAT_AT(m.b2, i, j);
                // wiggle
                MAT_AT(m.b2, i, j) += eps;
                // set in diff
                MAT_AT(diff.b2, i, j) = (rms_error(m, ti, to) - init_cost) / eps;
                // restore
                MAT_AT(m.b2, i, j) = saved;
            }
        }
        apply_diff(m, diff, rate);
        // printf("%f\n", rms_error(m, ti, to));
    }
}

int main() {
    // model m = {.first_layer = mat_alloc(2, 2)}
    // row = no of inputs
    // col = no of outputs
    model m = xor_alloc();

    mat_rand(m.b1, 0, 1);
    mat_rand(m.w2, 0, 1);
    mat_rand(m.w1, 0, 1);
    mat_rand(m.b2, 0, 1);


    size_t stride = 3;
    size_t row_count = sizeof(xor_df) / sizeof(xor_df[0]) / 3;
    printf("%zu\n", row_count);
    Matrix ti = {.rows = row_count, .cols = 2, .stride = stride, .es = xor_df};
    Matrix to = {.rows = row_count, .cols = 1, .stride = stride, .es = xor_df + 2};
    MAT_PRINT(ti);
    MAT_PRINT(to);

    printf("inital cost: %f\n", rms_error(m, ti, to));
    train(m, ti, to);
    printf("after cost: %f\n", rms_error(m, ti, to));
    print_predictions(m);

    return 0;
}
