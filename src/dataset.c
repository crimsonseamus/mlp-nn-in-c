#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../include/dataset.h"

Dataset dataset_create(
	float* X,
	uint8_t* y,
	int n,
	int dim
) {

	Dataset new_dataset;

	new_dataset.X = X;
	new_dataset.y = y;
	new_dataset.n = n;
	new_dataset.dim = dim;

	return new_dataset;
}

void dataset_print_info(Dataset* ds) {
	printf("n: %d\n", ds->n);
	printf("dim: %d\n", ds->dim);
	printf("first pixel: %f\n", ds->X[0]);
	printf("first label: %u\n", ds->y[0]);
}

void dataset_init_indices(Dataset *ds) {
	if(ds->indices) free(ds->indices);

	ds->indices = malloc(ds->n * sizeof(int));

	if(!ds->indices) {
		printf("Error allocating memory for indices\n");
		exit(1);
	}

	for(int i=0; i< ds->n; i++) {
		ds->indices[i] = i;
	}
}

void dataset_shuffle(Dataset *ds, unsigned int seed) {
	srand(seed);

	for(int i = ds->n-1; i > 0; i--) {
		unsigned int rand_idx =  rand() % (i + 1);
		
		int temp = ds->indices[i];
		ds->indices[i] = ds->indices[rand_idx];
		ds->indices[rand_idx] = temp;
	}
}

void dataset_get_batch(
	const Dataset *ds,
	int start,
	int batch_size,
	float *out_X,
	uint8_t *out_y
) {
	if(start + batch_size > ds->n) {
		printf("Wrong start index or batch size!\n");
		exit(1);
	}


	for(int b = 0; b < batch_size; b++) {
		unsigned int idx = ds->indices[start + b];
		
		for(int j = 0; j < ds->dim; j++) {
			out_X[b * ds->dim + j] = ds->X[idx*ds->dim + j];
		}
		out_y[b] = ds->y[idx];
	}
}

void dataset_free(Dataset* ds) {
    if (!ds) return;
    free(ds->indices);
    ds->indices = NULL;
}

