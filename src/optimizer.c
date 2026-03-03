#include "../include/optimizer.h"

SGD sgd_create(float lr) {
	SGD new_sgd;
	new_sgd.lr = lr;

	return new_sgd;
}

void sgd_step(
	const SGD *opt, 
	float *W, 
	float *b, 
	const float *dW, 
	const float *db, 
	int input_dim, 
	int num_classes
) {
	for(int i = 0; i < input_dim * num_classes; i++) {
		W[i] -= opt->lr * dW[i];
	}

	for(int j = 0; j < num_classes; j++) {
		b[j] -= opt->lr * db[j];
	}
}
