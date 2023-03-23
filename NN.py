import numpy as np
import ipdb
import matplotlib.pyplot as plt
from utils import visualize_convergence, relu, sigmoid, tanh

class NeuralNetwork:
    def __init__(self, sizes, activation, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sizes = sizes

        if activation == 'sigmoid':
            self.activation_function = sigmoid
        elif activation == 'tanh':
            self.activation_function = tanh
        elif activation == 'relu':
            self.activation_function = relu


        self.params = self.initialize_params(sizes)

        self.cache = {}

    def initialize_params(self, sizes):
        params = {}
        L = len(sizes)

        for i in range(1, L):
            params['W' + str(i)] = np.random.randn(sizes[i], sizes[i-1]) * 0.01
            params['b' + str(i)] = np.zeros((sizes[i], 1))

        return params
    
    def forward_propagation(self, X):
        params = self.params
        cache = self.cache
        L = len(params) // 2

        cache['A0'] = X
        #ipdb.set_trace()
        for i in range(1, L+1):
            cache['Z' + str(i)] = np.dot(params['W' + str(i)], cache['A' + str(i-1)]) + params['b' + str(i)]
            cache['A' + str(i)] = self.activation_function(cache['Z' + str(i)])

        return cache['A' + str(i)]
    
    def compute_cost(self, y_hat, y):
        m = y.shape[1]
        cost = -1/m * np.sum(y.T * np.log(y_hat) + (1 - y.T) * np.log(1 - y_hat))
        return cost
    
    def backward_propagation(self, y_hat, y):
        params = self.params
        cache = self.cache
        grads = {}

        L = len(params) // 2

        m = y.shape[1]

        grads['dZ' + str(L)] = y_hat - y
        for i in reversed(range(1, L+1)):
            if i == L:
                dZ = grads['dZ' + str(i)]
                A_prev = cache['A' + str(i-1)].T
            else:
                dZ = np.dot(dA, params['W' + str(i+1)].T) * (cache['A' + str(i)] > 0)
                A_prev = cache['A' + str(i-1)].T
            grads['dW' + str(i)] = np.dot(A_prev, dZ) / m
            grads['db' + str(i)] = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, params['W' + str(i)].T)

        return grads
    
    def update_params(self, grads):
        params = self.params

        for i in range(1, len(params) // 2 + 1):
            params['W' + str(i)] = params['W' + str(i)] - self.learning_rate * grads['dW' + str(i)]
            params['b' + str(i)] = params['b' + str(i)] - self.learning_rate * grads['db' + str(i)]

    def fit(self, X, y):
        costs = []

        for i in range(self.epochs):
            y_hat = self.forward_propagation(X)
            cost = self.compute_cost(y_hat, y)
            grads = self.backward_propagation(y_hat, y.T)
            self.update_params(grads)

            if i % 100 == 0:
                costs.append(cost)
                print('Epoch: {}, Cost: {}'.format(i, cost))
            
        visualize_convergence(costs, self.learning_rate, self.epochs)

    def predict(self, X):
        y_hat = self.forward_propagation(X)
        return np.round(y_hat)
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y_hat == y)
    
    def get_params(self):
        return self.params
        
def main():
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    data = load_diabetes()
    X = data.data
    y = data.target.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    nn = NeuralNetwork([353, 10, 5, 2, 1], activation='relu', learning_rate=0.01, epochs=1000)

    #ipdb.set_trace()
    params = nn.get_params()
    nn.fit(X_train, y_train.T)
    print(nn.score(X_test, y_test.T))

if __name__ == '__main__':
    main()

        





