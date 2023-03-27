import numpy as np
import ipdb
import matplotlib.pyplot as plt
from utils import visualize_convergence, relu, sigmoid, tanh, relu_backward, sigmoid_backward, tanh_backward

class NeuralNetwork:
    def __init__(self, sizes, activation, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sizes = sizes

        self.params = self.initialize_params(sizes)

        self.cache = {}

    def initialize_params(self, sizes):
        params = {}
        L = len(sizes)

        for i in range(1, L):
            params['W' + str(i)] = np.random.randn(sizes[i], sizes[i-1]) * 0.01
            params['b' + str(i)] = np.zeros((sizes[i], 1))

        return params
    
    def linear_forward(self, A, W, b):
        Z = (np.dot(W, A) + b)
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

    def forward_propagation(self, X):
        params = self.params
        cache = self.cache
        caches = []

        A = X
        L = len(params) // 2

        for i in range(1, L):
            A, cache = self.linear_activation_forward(A, params['W' + str(i)], params['b' + str(i)], activation='relu')
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, params['W' + str(L)], params['b' + str(L)], activation='sigmoid')
        caches.append(cache)
        return AL, caches

    def compute_cost(self, y_hat, y):
        m = y.shape[0]

        cost = -1/m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat), axis=1)
        return cost

    
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation='relu'):
        
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
        grads[f"dA{L-1}"] = dA_prev_temp
        grads[f"dW{L}"] = dW_temp
        grads[f"db{L}"] = db_temp

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA_prev_temp, current_cache, 'relu')
            grads[f"dA{l}"] = dA_prev_temp
            grads[f"dW{l+1}"] = dW_temp
            grads[f"db{l+1}"] = db_temp

        return grads
  
    def update_params(self, grads):
        params = self.params

        L = len(params) // 2

        for i in range(1, len(params) // 2 + 1):
            params['W' + str(i)] = params['W' + str(i)] - self.learning_rate * grads['dW' + str(i)]
            params['b' + str(i)] = params['b' + str(i)] - self.learning_rate * grads['db' + str(i)]

        return params

    def fit(self, X, y):
        costs = []

        for i in range(self.epochs):
            y_hat, caches = self.forward_propagation(X)
            cost = self.compute_cost(y_hat, y)
            grads = self.backward_propagation(y_hat, y.T, caches)
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
    #from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    data = pd.read_csv('diabetes.csv')
    #ipdb.set_trace()
    X = data.drop('Outcome', axis=1).values
    y = data['Outcome'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    nn = NeuralNetwork([614, 8, 5, 2, 1], activation='relu', learning_rate=0.01, epochs=1000)

    #ipdb.set_trace()
    params = nn.get_params()
    nn.fit(X_train, y_train)
    print(nn.score(X_test, y_test))

if __name__ == '__main__':
    main()

        





