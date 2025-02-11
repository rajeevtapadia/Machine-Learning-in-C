#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 4
#define PARAMS 3 // input params + bias

enum { w1, w2, bias };

typedef struct {
    float or_gate[PARAMS];
    float and_gate[PARAMS];
    float nand_gate[PARAMS];
} model;

// OR
int or_df[][3] = {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
// AND
int and_df[][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1}};
// NAND
int nand_df[][3] = {{0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 0}};
// XOR
int xor_df[][3] = {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 0}};

float get_rand() {
    return (float)rand() / RAND_MAX;
}

// return a model with random weights
model rand_xor() {
    srand(time(0));
    model m;
    float *ptr = (float *)&m;
    for (int i = 0; i < PARAMS * 3; i++, ptr++) {
        *ptr = get_rand();
    }
    return m;
}

float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// takes input x and y to model and predicts the ouput
float predict(model *m, float x1, float x2) {
    float a = sigmoidf(m->or_gate[w1] * x1 + m->or_gate[w2] * x2 + m->or_gate[bias]);
    float b = sigmoidf(m->nand_gate[w1] * x1 + m->nand_gate[w2] * x2 + m->nand_gate[bias]);
    return sigmoidf(m->and_gate[w1] * a + m->and_gate[w2] * b + m->and_gate[bias]);
}

void print_predictions(model *m) {
    printf("   x1       x2    actual_y   pred\n");
    for (int i = 0; i < SIZE; i++) {
        float x1 = xor_df[i][0];
        float x2 = xor_df[i][1];
        float acutal_y = xor_df[i][2];
        float pred = predict(m, x1, x2);
        float diff = pred - x2;
        printf("%f %f %f %f\n", x1, x2, acutal_y, pred);
    }
}

void print_weights(model m) {
    printf("------------------\n");
    printf("or\n");
    for (int i = 0; i < 3; i++)
        printf("%f\n", m.or_gate[i]);
    printf("nand\n");
    for (int i = 0; i < 3; i++)
        printf("%f\n", m.nand_gate[i]);
    printf("and\n");
    for (int i = 0; i < 3; i++)
        printf("%f\n", m.and_gate[i]);
}

/*
 * Note that even if our design is supposed to have
 * the nerurons working as individual gates it is not 
 * the case. The neural network in training adjustes the
 * neurons on its own. These individual neruons might 
 * even have continuous ouput.
*/
void print_individual_gate_predictions(model m) {
    printf("\n-------------------\n");
    printf("'AND' neuron\n");
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            printf("%d & %d = %f\n", i, j, sigmoidf(m.and_gate[w1] * i + m.and_gate[w2] * j + m.and_gate[bias]));
        }
    }
    
    printf("'OR' neuron\n");
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            printf("%d | %d = %f\n", i, j, sigmoidf(m.or_gate[w1] * i + m.or_gate[w2] * j + m.or_gate[bias]));
        }
    }
    
    printf("'NAND' neuron\n");
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            printf("~(%d & %d) = %f\n", i, j, sigmoidf(m.nand_gate[w1] * i + m.nand_gate[w2] * j + m.nand_gate[bias]));
        }
    }
}

float rms_error(model *m) {
    float result = 0;
    for (int i = 0; i < SIZE; i++) {
        float x1 = xor_df[i][0];
        float x2 = xor_df[i][1];
        float y = xor_df[i][2];
        float pred = predict(m, x1, x2);
        float diff = pred - y;
        result += diff * diff;
    }
    result /= SIZE;
    return result;
}

void apply_diff(model *m, model *diff) {
    float *ptr = (float *)m;
    float *ptr_diff = (float *)diff;
    for (int i = 0; i < PARAMS * 3; i++) {
        *(ptr + i) -= *(ptr_diff + i);
    }
}

void train(model *m) {
    float eps = 1e-1;
    model diff;
    float *ptr = (float *)m;
    float *ptr_diff = (float *)&diff;
    float saved = 0.f;
    // iterate over all attr of neruons for all 3 neurons
    for (int itr = 0; itr < 100*1000; itr++) {
        float prev_error = rms_error(m);
        for (int i = 0; i < PARAMS * 3; i++) {
            // save the weight
            saved = *(ptr + i);
            // wiggle the weight
            *(ptr + i) += eps;
            // calculate the new error
            *(ptr_diff + i) = (rms_error(m) - prev_error) / eps;
            // restore weight for wiggling different parameter
            *(ptr + i) = saved;
        }
        apply_diff(m, &diff);
        // printf("cost: %f\n", rms_error(m));
    }
}


int main() {
    model m = rand_xor();
    print_weights(m);
    printf("initial cost: %f\n", rms_error(&m));
    train(&m);
    print_predictions(&m);
    print_weights(m);
    printf("updated cost: %f\n", rms_error(&m));
    print_individual_gate_predictions(m);
    return 0;
}
