/*
 * training a neruon with two two inputs, single output, no bias
 * and sigmod as activation function
 * 
 * result: loss = 0.0768 after 40k iterations,
           a bit better than no activation function
           can use bias to optimise further
 * output:
 * intitial weights and bias
 * w1: 9.878065 w2: 9.878065 b: -9.878065
 * updated weights and bias
 * w1: 11.863961 w2: 11.863961 b: -5.827868
 *    x1        x2       y      bias     pred
 * 0.000000 0.000000 0.000000 -5.827868 0.002936
 * 1.000000 0.000000 1.000000 -5.827868 0.997615
 * 0.000000 1.000000 1.000000 -5.827868 0.997615
 * 1.000000 1.000000 1.000000 -5.827868 1.000000
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SIZE 4

typedef struct {
    float w1;
    float w2;
    float bias;
} weight_t;

// OR
int df[][3] = {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
// AND
// int df[][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1}};

float get_rand() {
    srand(time(0));
    return (float)rand() / RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f/ (1.f + expf(-x));
}

void print_predictions(float w1, float w2, float bias) {
    printf("   x1        x2       y      bias     pred\n");
    for (int i = 0; i < SIZE; i++) {
        float x1 = df[i][0];
        float x2 = df[i][1];
        float y = df[i][2];
        float pred = sigmoidf(w1 * x1 + w2 * x2 + bias);
        float diff = pred - y;
        printf("%f %f %f %f %f\n", x1, x2, y, bias, pred);
    }
}

float rms_error(float w1, float w2, float bias) {
    float result = 0;
    for (int i = 0; i < SIZE; i++) {
        float x1 = df[i][0];
        float x2 = df[i][1];
        float y = df[i][2];
        float pred = sigmoidf(w1 * x1 + w2 * x2 + bias);
        float diff = pred - y;
        result += diff * diff;
    }
    result /= SIZE;
    return result;
}

weight_t train(float w1, float w2, float bias) {
    float rate = 1e-1;
    float eps = 1e-1;
    weight_t weights;

    int iterations = 2000000;
    for (int i = 0; i < iterations; i++) {
        float curr_error = rms_error(w1, w2, bias);
        // printf("loss func value: %f\n", curr_error);
        float slope_w1 = (rms_error(w1 - eps, w2, bias) - curr_error) / eps;
        float slope_w2 = (rms_error(w1, w2 - eps, bias) - curr_error) / eps;
        float slope_bias = (rms_error(w1, w2, bias - eps) - curr_error) / eps;

        // printf("slope_w1: %f slope_w2: %f\n", slope_w1, slope_w2);
        w1 += slope_w1 * rate;
        w2 += slope_w2 * rate;
        bias += slope_bias * rate;
        // printf("updated weights - w1: %f w2: %f\n", w1, w2);
        if(i == 0 || i == iterations -1)
            printf("updated loss func value: %f\n", rms_error(w1, w2, bias));
        weights.w1 = w1;
        weights.w2 = w2;
        weights.bias = bias;
    }
    return weights;
}

int main() {
    float w1 = get_rand()*10;
    float w2 = get_rand()* 10;
    float bias = get_rand()* (-10);
    printf("intitial weights and bias\n%f, %f, %f\n", w1, w2, bias);
    weight_t w = train(w1, w2, bias);
    printf("updated weights and bias\n%f, %f, %f\n", w.w1, w.w2, w.bias);
    print_predictions(w.w1, w.w2, w.bias);
    return 0;
}