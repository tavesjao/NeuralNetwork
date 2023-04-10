import numpy as np
import ipdb
import matplotlib.pyplot as plt
from utils import visualize_convergence, relu, sigmoid, tanh, relu_backward, sigmoid_backward, tanh_backward, load_data

class NeuralNetwork:
    def __init__(self, sizes, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sizes = sizes


    def initialize_params(self, sizes):
        params = {}
        L = len(sizes)

        for i in range(1, L):
            params['W' + str(i)] = np.random.randn(sizes[i], sizes[i-1]) / np.sqrt(sizes[i-1]) #* 0.01
            params['b' + str(i)] = np.zeros((sizes[i], 1))

        return params
    
    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation='relu'):
        if activation == 'relu':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)

        elif activation == 'sigmoid':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        elif activation == 'tanh':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = tanh(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_propagation(self, X, params):
        caches = []

        A = X
        L = len(params) // 2

        for i in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, params['W' + str(i)], params['b' + str(i)], activation='relu')
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, params['W' + str(L)], params['b' + str(L)], activation='sigmoid')
        caches.append(cache)

        return AL, caches

    def compute_cost(self, AL, y):
        m = y.shape[0]

        cost = -1/m * np.sum(np.multiply(np.log(AL),y) + (1-y) * np.log(1-AL)) 
        
        cost = np.squeeze(cost)

        return cost

    
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        
        linear_cache, activation_cache = cache

        if activation == 'relu':
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        elif activation == 'sigmoid':
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            
        elif activation == 'tanh':
            dZ = tanh_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    
    def backward_propagation(self, AL, y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        y = y.reshape(AL.shape)
        
        dAL = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))

        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, caches[L-1], 'sigmoid')
        grads["dA" + str(L-1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA_prev_temp, current_cache, 'relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l+1)] = dW_temp
            grads["db" + str(l+1)] = db_temp

        return grads
  
    def update_params(self, grads, params):

        L = len(params) // 2

        for i in range(L):
            params['W' + str(i+1)] = params['W' + str(i+1)] - self.learning_rate * grads['dW' + str(i+1)]
            params['b' + str(i+1)] = params['b' + str(i+1)] - self.learning_rate * grads['db' + str(i+1)]

        return params

    def fit(self, X, y):
        costs = []
        parameters = self.initialize_params(self.sizes)
        for i in range(self.epochs):
            AL, caches = self.forward_propagation(X, parameters)
            cost = self.compute_cost(AL, y)
            grads = self.backward_propagation(AL, y, caches)
            parameters = self.update_params(grads, parameters)

            if i % 100 == 0:
                costs.append(cost)
                print('Epoch: {}, Cost: {}'.format(i, cost))

        visualize_convergence(costs, self.learning_rate, self.epochs)

        return parameters

    def predict(self, X, y, parameters):
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.forward_propagation(X, parameters)

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        #print results
        accuracy = str(np.sum((p == y)/m))
            
        return accuracy



