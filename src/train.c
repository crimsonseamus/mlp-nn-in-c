#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../include/train.h"
#include "../include/dataset.h"
#include "../include/optimizer.h"
#include "../include/math_ops.h"
#include "../include/model_mlp.h"
#include "../include/serialize_mlp.h"

#ifndef MIN
#define MIN(A,B) ((A) < (B) ? (A) : (B))
#endif

float evaluate_mlp(
	const Dataset* ds,
	int batch_size,
	const float* W1, const float* b1, int hidden_dim,
	const float* W2, const float* b2,
	float* z1, float* a1, float* logits,
	uint8_t* batch_labels, float* batch_X, uint8_t* preds,
	float* out_accuracy
) {
	float total_loss = 0.0f;
	int total_correct = 0;
	int total_seen = 0;

	for (int i = 0; i < ds->n; i += batch_size) {
		int current_batch = MIN(batch_size, ds->n - i);

		dataset_get_batch(ds, i, current_batch, batch_X, batch_labels);

		mlp_forward(
			batch_X,
			current_batch,
			INPUT_DIM,
			W1,
			b1,
			hidden_dim,
			W2,
			b2,
			NUM_CLASSES,
			z1,
			a1,
			logits
		);

		softmax_rowwise_inplace(logits, current_batch, NUM_CLASSES);

		float batch_loss = cross_entropy_loss(logits, batch_labels, current_batch, NUM_CLASSES);
		total_loss += batch_loss * (float)current_batch;

		argmax_rowwise(logits, current_batch, NUM_CLASSES, preds);

		for (int t = 0; t < current_batch; t++) {
			if (preds[t] == batch_labels[t]) total_correct++;
		}

		total_seen += current_batch;
	}

	float avg_loss = total_loss / (float)total_seen;
	*out_accuracy = (float)total_correct / (float)total_seen;
	return avg_loss;
}

float train_one_epoch_mlp(
	Dataset* train_ds,
	int batch_size,
	float* W1,
	float* b1,
	int hidden_dim,
	float* W2,
	float* b2,
	float* dW1,
	float* db1,
	float* dW2,
	float* db2,
	float* z1,
	float* a1,
	float* logits,
	float* dlogits,
	float* da1_tmp,
	float* dz1_tmp,
	uint8_t* batch_labels,
	float* batch_X,
	const SGD* opt,
	unsigned int shuffle_seed,
	uint8_t* preds,
	float* out_accuracy
) {
	float total_loss = 0.0f;
	int total_correct = 0;
	int total_seen = 0;

	dataset_shuffle(train_ds, shuffle_seed);

	for (int i = 0; i < train_ds->n; i += batch_size) {
		int current_batch = MIN(batch_size, train_ds->n - i);

		dataset_get_batch(train_ds, i, current_batch, batch_X, batch_labels);

		mlp_forward(
			batch_X, current_batch, INPUT_DIM,
			W1, b1, hidden_dim,
			W2, b2, NUM_CLASSES,
			z1, a1, logits
		);

		softmax_rowwise_inplace(logits, current_batch, NUM_CLASSES);

		float batch_loss = cross_entropy_loss(logits, batch_labels, current_batch, NUM_CLASSES);
		total_loss += batch_loss * (float)current_batch;

		softmax_cross_entropy_backward(logits, batch_labels, current_batch, NUM_CLASSES, dlogits);

		mlp_backward(
			batch_X, current_batch, INPUT_DIM,
			W1, b1, hidden_dim,
			W2, b2, NUM_CLASSES,
			z1, a1,
			dlogits,
			dW1, db1,
			dW2, db2,
			da1_tmp, dz1_tmp
		);

		sgd_step(opt, W1, b1, dW1, db1, INPUT_DIM, hidden_dim);
		sgd_step(opt, W2, b2, dW2, db2, hidden_dim, NUM_CLASSES);

		argmax_rowwise(logits, current_batch, NUM_CLASSES, preds);
		for (int t = 0; t < current_batch; t++) {
			if (preds[t] == batch_labels[t]) total_correct++;
		}

		total_seen += current_batch;
	}

	float avg_loss = total_loss / (float)total_seen;
	*out_accuracy = (float)total_correct / (float)total_seen;
	return avg_loss;
}

void train_mlp(
	Dataset* train_ds,
	Dataset* test_ds,
	int epochs,
	int batch_size,
	float lr,
	int hidden_dim,
	unsigned int seed
) {
	const int input_dim = INPUT_DIM;
	const int num_classes = NUM_CLASSES;

	float* W1  = malloc(sizeof(float) * input_dim * hidden_dim);
	float* b1  = malloc(sizeof(float) * hidden_dim);
	float* W2  = malloc(sizeof(float) * hidden_dim * num_classes);
	float* b2  = malloc(sizeof(float) * num_classes);

	float* dW1 = malloc(sizeof(float) * input_dim * hidden_dim);
	float* db1 = malloc(sizeof(float) * hidden_dim);
	float* dW2 = malloc(sizeof(float) * hidden_dim * num_classes);
	float* db2 = malloc(sizeof(float) * num_classes);

	float* batch_X      = malloc(sizeof(float) * batch_size * input_dim);
	uint8_t* batch_lbls = malloc(sizeof(uint8_t) * batch_size);

	float* z1      = malloc(sizeof(float) * batch_size * hidden_dim);
	float* a1      = malloc(sizeof(float) * batch_size * hidden_dim);
	float* da1_tmp = malloc(sizeof(float) * batch_size * hidden_dim);
	float* dz1_tmp = malloc(sizeof(float) * batch_size * hidden_dim);

	float* logits  = malloc(sizeof(float) * batch_size * num_classes);
	float* dlogits = malloc(sizeof(float) * batch_size * num_classes);

	uint8_t* preds = malloc(sizeof(uint8_t) * batch_size);

	if (!W1 || !b1 || !W2 || !b2 || !dW1 || !db1 || !dW2 || !db2 ||
		!batch_X || !batch_lbls || !z1 || !a1 || !da1_tmp || !dz1_tmp ||
		!logits || !dlogits || !preds) {
		printf("train_mlp: malloc failed\n");
			exit(1);
		}

		dataset_init_indices(train_ds);
		dataset_init_indices(test_ds);

		mlp_init(W1, b1, input_dim, hidden_dim, W2, b2, num_classes, seed);

		SGD opt = sgd_create(lr);

		for (int e = 0; e < epochs; e++) {
			float train_acc = 0.0f;
			float test_acc  = 0.0f;

			float train_loss = train_one_epoch_mlp(
				train_ds, batch_size,
				W1, b1, hidden_dim,
				W2, b2,
				dW1, db1,
				dW2, db2,
				z1, a1,
				logits, dlogits,
				da1_tmp, dz1_tmp,
				batch_lbls, batch_X,
				&opt,
				seed + (unsigned int)e,
				preds,
				&train_acc
			);

			float test_loss = evaluate_mlp(
				test_ds, batch_size,
				W1, b1, hidden_dim,
				W2, b2,
				z1, a1, logits,
				batch_lbls, batch_X, preds,
				&test_acc
			);

			printf("Epoch %d/%d | train loss %.4f acc %.4f | test loss %.4f acc %.4f\n",
				   e + 1, epochs, train_loss, train_acc, test_loss, test_acc);
		}

		if (!mlp_save("model_mlp.bin", W1, b1, input_dim, hidden_dim, W2, b2, num_classes)) {
			printf("Failed to save model_mlp.bin\n");
		} else {
			printf("Saved model to model_mlp.bin\n");
		}

		free(W1);
		free(b1);
		free(W2);
		free(b2);
		free(dW1);
		free(db1);
		free(dW2);
		free(db2);

		free(batch_X);
		free(batch_lbls);

		free(z1);
		free(a1);
		free(da1_tmp);
		free(dz1_tmp);

		free(logits);
		free(dlogits);

		free(preds);
}
