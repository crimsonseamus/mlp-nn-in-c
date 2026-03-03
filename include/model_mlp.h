#ifndef MODEL_MLP_H
#define MODEL_MLP_H

void mlp_init(
    float *W1, float *b1, int input_dim, int hidden_dim,
    float *W2, float *b2, int num_classes,
    unsigned int seed
);

void mlp_zero_grads(
    float *dW1, float *db1, int input_dim, int hidden_dim,
    float *dW2, float *db2, int num_classes
);

void mlp_forward(
    const float *X, int batch_size, int input_dim,
    const float *W1, const float *b1, int hidden_dim,
    const float *W2, const float *b2, int num_classes,
    float *z1_out,
    float *a1_out,
    float *logits_out
);

void mlp_backward(
    const float *X, int batch_size, int input_dim,
    const float *W1, const float *b1, int hidden_dim,
    const float *W2, const float *b2, int num_classes,
    const float *z1,
    const float *a1,
    const float *dlogits,
    float *dW1_out, float *db1_out,
    float *dW2_out, float *db2_out,
    float *da1_tmp,
    float *dz1_tmp
);

#endif
