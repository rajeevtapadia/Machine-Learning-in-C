#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <sys/types.h>
// matrix
typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Matrix;

Matrix mat_alloc(size_t rows, size_t cols);
float get_rand();
float sigmoidf(float x);
void mat_rand(Matrix mat, float low, float high);
void mat_fill(Matrix mat, float num);
Matrix mat_get_row(Matrix mat, size_t row);
void mat_copy(Matrix dest, Matrix src);
void mat_mult(Matrix dest, Matrix a, Matrix b);
void mat_add(Matrix dest, Matrix a);
void mat_print(Matrix m, char *name, size_t padding);
void mat_sig(Matrix mat);

// neural network
typedef struct {
    size_t layer_count;
    Matrix *ws;
    Matrix *bs;
    Matrix *as; // no of activation matrices = no of layers + 1
} NN;

#define NN_PRINT(nn) nn_print(nn, #nn)
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).layer_count]

NN nn_alloc(size_t *layer_arch, size_t layer_count);
void nn_rand(NN nn, float low, float high);
void nn_print(NN nn, char *name);
void nn_forward(NN nn);
float nn_rms_error(NN nn, Matrix ti, Matrix to);
void nn_train(NN nn, NN diff, float eps, Matrix ti, Matrix to);
void nn_apply_diff(NN nn, NN diff, float rate);

#endif

#ifdef NN_IMPLEMENTATION
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// macro to access matrix at i th row and j th column
#define MAT_AT(m, i, j) m.es[(i) * (m).stride + (j)]
#define MAT_PRINT(matrix_name) mat_print(matrix_name, #matrix_name, 0)
#define ARR_LEN(arr) sizeof(arr) / sizeof((arr)[0])

Matrix mat_alloc(size_t rows, size_t cols) {
    Matrix m = {.rows = rows, .cols = cols, .stride = cols};
    m.es = (float *)malloc(sizeof(*m.es) * rows * cols);
    assert(m.es != NULL);
    return m;
}

float get_rand() { return (float)rand() / RAND_MAX; }

// float sigmoidf(float x) { return 1.f / 1.f + expf(-x); }
float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

void mat_rand(Matrix mat, float low, float high) {
    srand(time(0));
    for (size_t i = 0; i < mat.rows; i++) {
        for (size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = get_rand() * (high - low) + low;
        }
    }
}

void mat_fill(Matrix mat, float num) {
    for (size_t i = 0; i < mat.rows; i++) {
        for (size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = num;
        }
    }
}

Matrix mat_get_row(Matrix mat, size_t row) {
    Matrix new_mat = {.es = &MAT_AT(mat, row, 0), .rows = 1, .cols = mat.cols, .stride = mat.cols};
    return new_mat;
}

void mat_copy(Matrix dest, Matrix src) {
    assert(dest.rows == src.rows);
    assert(dest.cols == src.cols);
    for (size_t i = 0; i < dest.rows; i++) {
        for (size_t j = 0; j < dest.cols; j++) {
            MAT_AT(dest, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_mult(Matrix dest, Matrix a, Matrix b) {
    assert(a.cols == b.rows);
    assert(dest.rows == a.rows);
    assert(dest.cols == b.cols);

    for (size_t i = 0; i < dest.rows; i++) {
        for (size_t j = 0; j < dest.cols; j++) {
            MAT_AT(dest, i, j) = 0;
            for (size_t k = 0; k < a.cols; k++) {
                MAT_AT(dest, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_add(Matrix dest, Matrix a) {
    assert(dest.rows == a.rows);
    assert(dest.cols == a.cols);
    for (size_t i = 0; i < dest.rows; i++) {
        for (size_t j = 0; j < dest.cols; j++) {
            MAT_AT(dest, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_print(Matrix mat, char *name, size_t padding) {
    printf("%*s%s [\n", padding, "", name);
    for (size_t i = 0; i < mat.rows; i++) {
        printf("%*s    ", padding, "");
        for (size_t j = 0; j < mat.cols; j++) {
            printf("%f ", MAT_AT(mat, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", padding, "");
}

void mat_sig(Matrix mat) {
    for (size_t i = 0; i < mat.rows; i++) {
        for (size_t j = 0; j < mat.cols; j++) {
            MAT_AT(mat, i, j) = sigmoidf(MAT_AT(mat, i, j));
        }
    }
}

NN nn_alloc(size_t *layer_arch, size_t layer_count) {
    assert(layer_count > 0);
    NN nn;
    nn.layer_count = layer_count - 1;
    /*
     * eg arch {2, 2, 1}
     *
     * for a matrix
     * row = no of inputs to the layer
     * col = no of outputs from the layer
     */

    // allocate arrays for weigts, bias, activation
    nn.ws = malloc(sizeof(*nn.ws) * nn.layer_count);
    assert(nn.ws != NULL);
    nn.bs = malloc(sizeof(*nn.bs) * nn.layer_count);
    assert(nn.bs != NULL);
    nn.as = malloc(sizeof(*nn.as) * (nn.layer_count + 1));
    assert(nn.as != NULL);
    /*
     * for a weight matrix
     * row = no of neurons in previous layer
     * col = no of neurons in current layer
     *
     * for a activation matrix
     * row = 1
     * col = no of neurons in current layer
     *
     * for bias matrix
     * row = 1
     * col = no of neurons in current layer
     */
    nn.as[0] = mat_alloc(1, layer_arch[0]);
    for (int i = 1; i < layer_count; i++) {
        nn.ws[i - 1] = mat_alloc(layer_arch[i - 1], layer_arch[i]);
        nn.bs[i - 1] = mat_alloc(1, layer_arch[i]);
        nn.as[i] = mat_alloc(1, layer_arch[i]);
    }
    return nn;
}

void nn_rand(NN nn, float low, float high) {
    for (int i = 0; i < nn.layer_count; i++) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_print(NN nn, char *name) {
    char buff[256];
    printf("%s: [\n", name);
    for (int i = 0; i < nn.layer_count; i++) {
        snprintf(buff, sizeof(buff), "ws%zu", i);
        mat_print(nn.ws[i], buff, 4);
        snprintf(buff, sizeof(buff), "bs%zu", i);
        mat_print(nn.bs[i], buff, 4);
    }
    printf("]\n");
}

void nn_forward(NN nn) {
    for (int i = 0; i < nn.layer_count; i++) {
        mat_mult(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_add(nn.as[i + 1], nn.bs[i]);
        mat_sig(nn.as[i + 1]);
    }
}

float nn_rms_error(NN nn, Matrix ti, Matrix to) {
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);

    size_t row_count = ti.rows;
    size_t output_col_count = to.cols;
    float error = 0.f;
    for (size_t i = 0; i < row_count; i++) {
        mat_copy(NN_INPUT(nn), mat_get_row(ti, i));
        nn_forward(nn);

        Matrix x = mat_get_row(ti, i);
        Matrix y_actual = mat_get_row(to, i);
        Matrix y_pred = NN_OUTPUT(nn);

        for (size_t j = 0; j < output_col_count; j++) {
            float diff = MAT_AT(y_pred, 0, j) - MAT_AT(y_actual, 0, j);
            error += diff * diff;
        }
    }
    return error / row_count;
}

void nn_train(NN nn, NN diff, float eps, Matrix ti, Matrix to) {
    float saved;
    float init_cost = nn_rms_error(nn, ti, to);
    for (size_t i = 0; i < nn.layer_count; i++) {
        // for weights
        for (size_t j = 0; j < nn.ws[i].rows; j++) {
            for (size_t k = 0; k < nn.ws[i].cols; k++) {
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) -= eps;
                float new_cost = nn_rms_error(nn, ti, to);
                MAT_AT(diff.ws[i], j, k) = (new_cost - init_cost) / eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        // for bias
        for (size_t j = 0; j < nn.bs[i].rows; j++) {
            for (size_t k = 0; k < nn.bs[i].cols; k++) {
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) -= eps;
                float new_cost = nn_rms_error(nn, ti, to);
                MAT_AT(diff.bs[i], j, k) = (new_cost - init_cost) / eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void nn_apply_diff(NN nn, NN diff, float rate) {
    for (size_t i = 0; i < nn.layer_count; i++) {
        // for weights
        for (size_t j = 0; j < nn.ws[i].rows; j++) {
            for (size_t k = 0; k < nn.ws[i].cols; k++) {
                MAT_AT(nn.ws[i], j, k) += rate * MAT_AT(diff.ws[i], j, k);
            }
        }

        // for bias
        for (size_t j = 0; j < nn.bs[i].rows; j++) {
            for (size_t k = 0; k < nn.bs[i].cols; k++) {
                MAT_AT(nn.bs[i], j, k) += rate * MAT_AT(diff.bs[i], j, k);
            }
        }
    }
}
#endif
