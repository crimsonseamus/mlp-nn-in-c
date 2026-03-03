#include <math.h>
#include <stdint.h>
#include "../include/math_ops.h"


void softmax_rowwise_inplace(
	float *logits,
	int batch_size,
	int num_classes
) {
	for(int i = 0; i < batch_size; i++) {
		float max = -INFINITY;
		float ex[num_classes];

		float sum = 0;	

		for(int j = 0; j < num_classes; j++) {
			if(max < logits[i * num_classes + j]) {
				max = logits[i * num_classes + j];
			}
		}

		for(int j = 0; j < num_classes; j++) {
			ex[j] = exp(logits[i * num_classes + j] - max);
		}

		for(int j = 0; j < num_classes; j++) {
			sum += ex[j];
		}

		for(int j = 0; j < num_classes; j++) {
			logits[i * num_classes + j] = ex[j] / sum;
		}
	}
}

float cross_entropy_loss(
	const float* probs,
	const uint8_t* labels,
	int batch_size,
	int num_classes
) {
	float total_loss = 0;

	for (int i = 0; i < batch_size; i++) {
		uint8_t correct_label = labels[i];
		float p = probs[i * num_classes + correct_label];

		if(p < 1e-7) {
			p = 1e-7;
		}
		total_loss += -log(p);
	}
	
	return total_loss / batch_size;
}

void softmax_cross_entropy_backward(
	const float *probs,
	const uint8_t *labels,
	int batch_size,
	int num_classes,
	float *dlogits_out
) {
	for(int i = 0; i < batch_size; i++) {
		uint8_t correct_label = labels[i];
		
		for(int j = 0; j < num_classes; j++) {
			float d = probs[i * num_classes +j];
			if(j == correct_label) d = d - 1;

			dlogits_out[i * num_classes + j] = d / batch_size;
		}
	}
}

void argmax_rowwise(
	const float *probs,
	int batch_size,
	int num_classes,
	uint8_t *pred_out
) {
	for(int i = 0; i < batch_size; i++) {
		float best_val = probs[i * num_classes];
		int best_idx = 0;

		for(int j = 1; j < num_classes; j++) {
			if(probs[i * num_classes + j] > best_val) {
				best_val = probs[i * num_classes + j];
				best_idx = j;
			}
		}

		pred_out[i] = best_idx;
	}
}

float accuracy(
	const uint8_t *pred,
	const uint8_t *labels,
	int batch_size
) {
	float correct = 0;

	for(int i = 0; i < batch_size; i++) {
		if(pred[i] == labels[i]) correct++;
	}
	
	return correct/(float)batch_size;
}
