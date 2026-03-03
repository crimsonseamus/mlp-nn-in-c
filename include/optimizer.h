#ifndef OPTIMIZER_H
#define OPTIMIZER_H

typedef struct {
	float lr;
} SGD;

SGD sgd_create(float lr);

void sgd_step(
	const SGD* opt, 
	float* W,
	float* b,
	const float* dW,
	const float* db,
	int input_dim,
	int num_classes
);

#endif
