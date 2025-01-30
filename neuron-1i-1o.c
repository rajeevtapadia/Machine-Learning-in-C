/*
 * training a neruon with single input, single output and no bias
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 4

int df[][2] = {{1, 2}, {2, 4}, {3, 6}, {4, 8}};

float get_rand() {
  // srand(time(0));
  return (float)rand() / RAND_MAX;
}

void print_predictions(float weight) {
  for (int i = 0; i < SIZE; i++) {
    float x = df[i][0];
    float y = df[i][1];
    float pred = weight * df[i][0];
    float diff = pred - y;
    printf("%f %f %f %f\n", x, y, pred, diff);
  }
}

float rms_error(float weight) {
  float result = 0;
  for (int i = 0; i < SIZE; i++) {
    float x = df[i][0];
    float y = df[i][1];
    float pred = weight * df[i][0];
    float diff = pred - y;
    result += diff * diff;
  }
  result /= SIZE;
  return result;
}

float train(float weight) {
  float rate = 1e-3;
  float eps = 1e-3;
  for(int i = 0; i < 500; i++) {
      float derror = (rms_error(weight + eps) - rms_error(weight)) / eps;
      // printf("derror %f\n", derror * rate);
      weight -= derror * rate;
      printf("update weight: %f\n", weight);
  }
  return weight;
}

int main() {
  // float weight = get_rand() * 10;
  float weight = 1.0f;
  printf("%f\n", weight);
  print_predictions(weight);
  float new_weight = train(weight);
  print_predictions(new_weight);
  printf("root mean square error: %f\n", rms_error(new_weight));
  // rms_error(weight - 0.01);
  return 0;
}
