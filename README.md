# Creating an MLP Neural Network in C (From Scratch)

This project is a from-scratch implementation of a neural network (MLP) written in pure C, without using any machine learning libraries.

The goal of this project is deep understanding of neural networks, not just high accuracy.

Everything is implemented manually:
- Dataset loading (MNIST)
- Forward propagation
- Backpropagation
- Gradient descent (SGD)
- Model evaluation
- Model saving/loading
- GUI-based digit prediction

--------------------------------------------------

PROJECT OVERVIEW

Model: Multi-Layer Perceptron (MLP)

Architecture:
- Input layer: 784 neurons (28x28 image)
- Hidden layer: 128 neurons + ReLU
- Output layer: 10 neurons + Softmax

Loss Function: Cross-Entropy Loss\
Optimizer: Stochastic Gradient Descent (SGD)\
Dataset: MNIST handwritten digits\
Language: C\
External Library: SDL2 (GUI only)

--------------------------------------------------

PROJECT STRUCTURE

src/\
  main.c\
  mnist.c\
  dataset.c\
  math_ops.c\
  model_mlp.c\
  optimizer.c\
  train.c\
  serialize_mlp.c\
  gui.c

include/\
  header files

data/\
  mnist dataset (not committed)

--------------------------------------------------

PROJECT FLOW

1. Load MNIST images and labels
2. Normalize pixel values
3. Create dataset structure
4. Initialize model weights and biases
5. Training loop:
   - Create mini-batches
   - Forward pass
   - Compute loss
   - Backpropagation
   - Update weights using SGD
6. Evaluate on test dataset
7. Save trained model
8. Load model and predict using GUI

--------------------------------------------------

TRAINING RESULTS

After 30 epochs:
- Train accuracy ~96%
- Test accuracy ~95.8%

--------------------------------------------------

GUI FEATURES

- Draw digits with mouse
- Convert drawing to MNIST format
- Predict digit in real-time
- Show top-3 predictions with confidence

Controls:\
C      Clear canvas\
Enter  Predict\
[ ]    Change brush size\
R      Erase mode\
I      Invert colors\
ESC    Exit

--------------------------------------------------

BUILD INSTRUCTIONS

Requirements:
- GCC or Clang
- CMake
- SDL2

Build:\
cmake -S . -B build\
cmake --build build -j\
./build/mnist_mlp


Run training:\
./mnist_mlp

Run GUI:\
./mnist_mlp gui

--------------------------------------------------

DATASET

Download MNIST from:\
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Place files in:\
data/mnist/

--------------------------------------------------

LEARNING OBJECTIVES

- Neural network math from scratch
- Forward and backward propagation
- Gradient-based optimization
- Mini-batch training
- Model serialization
- Real-time inference

--------------------------------------------------

AUTHOR NOTE

This project was built to understand neural networks from first principles.

