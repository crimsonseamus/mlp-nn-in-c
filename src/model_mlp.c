#include <stdint.h>
#include <math.h>

#include "../include/model_mlp.h"

static float frand_uniform(unsigned int *state) {
    *state = (*state) * 1664525u + 1013904223u;
    uint32_t x = *state;
    return (float)(x / 4294967296.0);
}

static void relu_forward(const float *z, float *a, int n) {
    for (int i = 0; i < n; i++) {
        float v = z[i];
        a[i] = (v > 0.0f) ? v : 0.0f;
    }
}

static void relu_backward(const float *z, const float *da, float *dz, int n) {
    for (int i = 0; i < n; i++) {
        dz[i] = (z[i] > 0.0f) ? da[i] : 0.0f;
    }
}

void mlp_init(
    float *W1,
    float *b1,
    int input_dim,
    int hidden_dim,
    float *W2,
    float *b2,
    int num_classes,
    unsigned int seed
) {
    unsigned int st = seed ? seed : 1u;

    float s1 = sqrtf(2.0f / (float)input_dim);
    for (int i = 0; i < input_dim * hidden_dim; i++) {
        float u = frand_uniform(&st);
        float v = (u * 2.0f - 1.0f) * s1;
        W1[i] = v;
    }
    for (int j = 0; j < hidden_dim; j++) b1[j] = 0.0f;

    float s2 = sqrtf(2.0f / (float)hidden_dim);
    for (int i = 0; i < hidden_dim * num_classes; i++) {
        float u = frand_uniform(&st);
        float v = (u * 2.0f - 1.0f) * s2;
        W2[i] = v;
    }
    for (int j = 0; j < num_classes; j++) b2[j] = 0.0f;
}

void mlp_zero_grads(
    float *dW1,
    float *db1,
    int input_dim,
    int hidden_dim,
    float *dW2,
    float *db2,
    int num_classes
) {
    int n1 = input_dim * hidden_dim;
    int n2 = hidden_dim * num_classes;

    for (int i = 0; i < n1; i++) dW1[i] = 0.0f;
    for (int i = 0; i < hidden_dim; i++) db1[i] = 0.0f;

    for (int i = 0; i < n2; i++) dW2[i] = 0.0f;
    for (int i = 0; i < num_classes; i++) db2[i] = 0.0f;
}

void mlp_forward(
    const float *X,
    int batch_size,
    int input_dim,
    const float *W1,
    const float *b1,
    int hidden_dim,
    const float *W2,
    const float *b2,
    int num_classes,
    float *z1_out,
    float *a1_out,
    float *logits_out
) {
    for (int i = 0; i < batch_size; i++) {
        for (int h = 0; h < hidden_dim; h++) {
            float sum = b1[h];
            for (int k = 0; k < input_dim; k++) {
                sum += X[i * input_dim + k] * W1[k * hidden_dim + h];
            }
            z1_out[i * hidden_dim + h] = sum;
        }
    }

    relu_forward(z1_out, a1_out, batch_size * hidden_dim);

    for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < num_classes; c++) {
            float sum = b2[c];
            for (int h = 0; h < hidden_dim; h++) {
                sum += a1_out[i * hidden_dim + h] * W2[h * num_classes + c];
            }
            logits_out[i * num_classes + c] = sum;
        }
    }
}

void mlp_backward(
    const float *X,
    int batch_size,
    int input_dim,
    const float *W1,
    const float *b1,
    int hidden_dim,
    const float *W2,
    const float *b2,
    int num_classes,
    const float *z1,
    const float *a1,
    const float *dlogits,
    float *dW1_out,
    float *db1_out,
    float *dW2_out,
    float *db2_out,
    float *da1_tmp,
    float *dz1_tmp
){
    (void)W1;
    (void)b1;
    (void)b2;

    mlp_zero_grads(dW1_out, db1_out, input_dim, hidden_dim, dW2_out, db2_out, num_classes);

    for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < num_classes; c++) {
            db2_out[c] += dlogits[i * num_classes + c];
        }
    }

    for (int h = 0; h < hidden_dim; h++) {
        for (int c = 0; c < num_classes; c++) {
            float sum = 0.0f;
            for (int i = 0; i < batch_size; i++) {
                sum += a1[i * hidden_dim + h] * dlogits[i * num_classes + c];
            }
            dW2_out[h * num_classes + c] = sum;
        }
    }

    for (int i = 0; i < batch_size; i++) {
        for (int h = 0; h < hidden_dim; h++) {
            float sum = 0.0f;
            for (int c = 0; c < num_classes; c++) {
                sum += dlogits[i * num_classes + c] * W2[h * num_classes + c];
            }
            da1_tmp[i * hidden_dim + h] = sum;
        }
    }

    relu_backward(z1, da1_tmp, dz1_tmp, batch_size * hidden_dim);

    for (int h = 0; h < hidden_dim; h++) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += dz1_tmp[i * hidden_dim + h];
        }
        db1_out[h] = sum;
    }

    for (int k = 0; k < input_dim; k++) {
        for (int h = 0; h < hidden_dim; h++) {
            float sum = 0.0f;
            for (int i = 0; i < batch_size; i++) {
                sum += X[i * input_dim + k] * dz1_tmp[i * hidden_dim + h];
            }
            dW1_out[k * hidden_dim + h] = sum;
        }
    }
}
