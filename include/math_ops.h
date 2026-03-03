#ifndef MATH_OPS_H
#define MATH_OPS_H

#include <stdint.h>

void softmax_rowwise_inplace(
    float* logits,
    int batch_size,
    int num_classes
);

float cross_entropy_loss(
    const float* probs,
    const uint8_t* labels,
    int batch_size,
    int num_classes
);

void softmax_cross_entropy_backward(
    const float* probs,
    const uint8_t* labels,
    int batch_size,
    int num_classes,
    float* dlogits_out
);

void argmax_rowwise(
    const float* probs,
    int batch_size,
    int num_classes,
    uint8_t* pred_out
);

float accuracy(
    const uint8_t* pred,
    const uint8_t* labels,
    int batch_size
);

#endif
