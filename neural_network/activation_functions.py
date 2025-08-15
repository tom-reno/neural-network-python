import numpy as np

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax_activation(x, node_values):
    return np.exp(x) / np.sum(np.exp(x))

def softmax_derivative(x, node_values):
    activation = softmax_activation(x, node_values)
    return activation * (1 - activation)

def relu_activation(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return 1 - np.maximum(0, x)

def tanh_activation(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
