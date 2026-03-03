#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "../include/mnist.h"
#include "../include/dataset.h"
#include "../include/train.h"
#include "../include/gui.h"

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "gui") == 0) {
        return gui_run("model_mlp.bin");
    }

    int n_train = 0, rows = 0, cols = 0;
    int n_train_labels = 0;

    float* X_train = mnist_load_images("data/mnist/train-images.idx3-ubyte", &n_train, &rows, &cols);
    uint8_t* y_train = mnist_load_labels("data/mnist/train-labels.idx1-ubyte", &n_train_labels);

    if (n_train != n_train_labels) {
        printf("Train images/labels count mismatch: %d vs %d\n", n_train, n_train_labels);
        return 1;
    }

    int n_test = 0, rows_t = 0, cols_t = 0;
    int n_test_labels = 0;

    float* X_test = mnist_load_images("data/mnist/t10k-images.idx3-ubyte", &n_test, &rows_t, &cols_t);
    uint8_t* y_test = mnist_load_labels("data/mnist/t10k-labels.idx1-ubyte", &n_test_labels);

    if (n_test != n_test_labels) {
        printf("Test images/labels count mismatch: %d vs %d\n", n_test, n_test_labels);
        return 1;
    }

    if (rows != rows_t || cols != cols_t) {
        printf("Train/Test image shape mismatch: train %dx%d vs test %dx%d\n", rows, cols, rows_t, cols_t);
        return 1;
    }

    const int dim = rows * cols;

    Dataset train_ds = dataset_create(X_train, y_train, n_train, dim);
    Dataset test_ds  = dataset_create(X_test,  y_test,  n_test,  dim);

    int epochs = 30;
    int batch_size = 64;
    float lr = 0.01f;
    unsigned int seed = 1234;

    train_mlp(&train_ds, &test_ds, epochs, batch_size, lr, HIDDEN_DIM, seed);

    free(train_ds.X);
    free(train_ds.y);
    free(test_ds.X);
    free(test_ds.y);

    printf("Training done. Run: ./mnist_mlp gui\n");
    return 0;
}

