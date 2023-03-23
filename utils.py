import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    
    cache = (A, W, b)
    
    return Z, cache


def visualize_convergence(costs, learning_rate, epochs):
    plt.plot(costs)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Learning rate: {}, Epochs: {}'.format(learning_rate, epochs))
    plt.show()


