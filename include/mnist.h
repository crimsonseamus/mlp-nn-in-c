#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>

float* mnist_load_images (
    const char* file_path,
    int* out_count,
    int* out_rows,
    int* out_cols
);

uint8_t* mnist_load_labels (
    const char* file_path,
    int* out_count
);


#endif
