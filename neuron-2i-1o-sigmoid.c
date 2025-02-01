/*
 * training a neruon with two two inputs, single output, no bias
 * and sigmod as activation function
 * 
 * result: loss = 0.062632 after 400k iterations,
           a bit better than no activation function
           can use bias to optimise further
 * output:
 * intitial weights (randomly selected)
 * w1: 0.840188 w2: 0.394383
 * updated weights - 
 * w1: 4.105291 w2: 4.103314
 *    x1        x2       y       pred
 * 0.000000 0.000000 0.000000 0.500000  - this case can be solved by bias
 * 1.000000 0.000000 1.000000 0.983782
 * 0.000000 1.000000 1.000000 0.983751
 * 1.000000 1.000000 1.000000 0.999728
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SIZE 4

typedef struct {
    float w1;
    float w2;
} weight_t;

int df[][3] = {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

float get_rand() {
    // srand(time(0));
    return (float)rand() / RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f/ (1.f + expf(-x));
}

void print_predictions(float w1, float w2) {
    printf("   x1        x2       y       pred\n");
    for (int i = 0; i < SIZE; i++) {
        float x1 = df[i][0];
        float x2 = df[i][1];
        float y = df[i][2];
        float pred = sigmoidf(w1 * x1 + w2 * x2);
        float diff = pred - y;
        printf("%f %f %f %f\n", x1, x2, y, pred);
    }
}

float rms_error(float w1, float w2) {
    float result = 0;
    for (int i = 0; i < SIZE; i++) {
        float x1 = df[i][0];
        float x2 = df[i][1];
        float y = df[i][2];
        float pred = sigmoidf(w1 * x1 + w2 * x2);
        float diff = pred - y;
        result += diff * diff;
    }
    result /= SIZE;
    return result;
}

weight_t train(float w1, float w2) {
    float rate = 1e-2;
    float eps = 1e-2;
    weight_t weights;

    for (int i = 0; i < 400000; i++) {
        float vim = rms_error(w1, w2);
        // printf("loss func value: %f\n", vim);
        float slope_w1 = (rms_error(w1 - eps, w2) - vim) / eps;
        float slope_w2 = (rms_error(w1, w2 - eps) - vim) / eps;

        // printf("slope_w1: %f slope_w2: %f\n", slope_w1, slope_w2);
        w1 += slope_w1 * rate;
        w2 += slope_w2 * rate;
        // printf("updated weights - w1: %f w2: %f\n", w1, w2);
        // printf("updated loss func value: %f\n", rms_error(w1, w2));
        weights.w1 = w1;
        weights.w2 = w2;
    }
    return weights;
}

int main() {
    float w1 = get_rand();
    float w2 = get_rand();
    printf("intitial weights\n%f, %f\n", w1, w2);
    weight_t w = train(w1, w2);
    printf("updated weights - w1: %f w2: %f\n", w.w1, w.w2);
    print_predictions(w.w1, w.w2);
    return 0;
}