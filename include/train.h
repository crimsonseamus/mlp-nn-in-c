#ifndef TRAIN_H
#define TRAIN_H

#include "dataset.h"
#include "optimizer.h"
#include <stdint.h>

#define INPUT_DIM 784
#define NUM_CLASSES 10
#define HIDDEN_DIM 128

float evaluate_mlp(
	const Dataset* ds,
	int batch_size,
	const float* W1, const float* b1, int hidden_dim,
	const float* W2, const float* b2,
	float* z1, float* a1, float* logits,
	uint8_t* batch_labels, float* batch_X, uint8_t* preds,
	float* out_accuracy
);

float train_one_epoch_mlp(
	Dataset* train_ds,
	int batch_size,
	float* W1, float* b1, int hidden_dim,
	float* W2, float* b2,
	float* dW1, float* db1,
	float* dW2, float* db2,
	float* z1, float* a1,
	float* logits, float* dlogits,
	float* da1_tmp, float* dz1_tmp,
	uint8_t* batch_labels, float* batch_X,
	const SGD* opt,
	unsigned int shuffle_seed,
	uint8_t* preds,
	float* out_accuracy
);

void train_mlp(
	Dataset* train_ds,
	Dataset* test_ds,
	int epochs,
	int batch_size,
	float lr,
	int hidden_dim,
	unsigned int seed
);

#endif
