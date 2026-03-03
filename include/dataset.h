#ifndef DATASET_H
#define DATASET_H

#include <stdint.h>

typedef struct {
	float* X;
	uint8_t* y;
	int n;
	int dim;
	int* indices;
} Dataset;

Dataset dataset_create (
	float* X,
	uint8_t* y,
	int n,
	int dim
);

void dataset_print_info (Dataset* ds);

void dataset_init_indices(Dataset* ds);

void dataset_shuffle(Dataset* ds, unsigned int seed);

void dataset_get_batch(
	const Dataset* ds,
	int start,
	int batch_size,
	float* out_X,
	uint8_t* out_y
);

void dataset_free(Dataset* ds);

#endif
