#ifndef UTILS_H
#define UTILS_H

#define MIN(A, B) ((A) < (B) ? (A) : (B))

void predict_one(
    const float* image,
    const float* W,
    const float* b,
    int input_dim,
    int num_classes
);

#endif
