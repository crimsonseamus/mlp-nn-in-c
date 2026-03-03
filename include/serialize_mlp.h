#ifndef SERIALIZE_MLP_H
#define SERIALIZE_MLP_H

#include <stdint.h>

int mlp_save(
    const char* path,
    const float* W1, const float* b1, int input_dim, int hidden_dim,
    const float* W2, const float* b2, int num_classes
);

int mlp_load(
    const char* path,
    float* W1, float* b1, int input_dim, int hidden_dim,
    float* W2, float* b2, int num_classes
);

#endif
