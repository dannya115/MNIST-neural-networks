import numpy as np
import random
from collections import deque

from import_data import training_data, validation_data, test_data

# define activation function and derivative
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))

class Network(object):

    # randomly generated weights and biases upon instantiating Network class - these
    # are normally distributed with the standard normal distribution 
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.rand(x, 1) for x in sizes[1:]]
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data is not None:
            n_test = len(test_data)
        
        n_training = len(training_data)

        # for each epoch, randomly shuffle (i.e. stochastic) training data
        # and split into mini batches
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n_training, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data is not None:
                print(f"Epoch {i}: {self.evaluate(test_data)} / {n_test}")
            
            else:
                print(f"Epoch {i} complete")

    # recursive feedforward algorithm used for generating the output layer for test data
    def feedforward(self, a):
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a

    def update_mini_batch(self, mini_batch, eta):
        
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        # for each data, label pair in a mini batch, calculate the gradient of the cost function with respect
        # to weights/biases. Add this to the total for a mini batch, and at the end of the mini batch, update
        # the weights/biases according to the learning rule
        for x, y in mini_batch:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        # construct numpy arrays of the change in weights and biases for each layer
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.weights]
        
        activation = x
        activations = [activation]
        zs = []
        delta = deque()

        # feedforward algorithm which populates zs and activations with vectors for l>1
        for w, b in zip(self.weights, self.biases):
            z = w @ activation + b 
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # computing the output error, and populating nabla_b/w for output layer
        delta.append(self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]))
        nabla_w[-1] = delta[-1] @ activations[-2].T
        nabla_b[-1] = delta[-1]

        # backpropagate the error for each layer
        for layer in range(2, self.num_layers):
            delta.appendleft((self.weights[-layer+1].T @ delta[0]) * sigmoid_prime(zs[-layer]))
            nabla_w[-layer] = delta[-layer] @ activations[-layer-1].T
            nabla_b[-layer] = delta[-layer]
        
        return (nabla_w, nabla_b)
        
    # cost derivative for a quadratic cost function
    def cost_derivative(self, a_L, y):
        return (a_L - y)

    # determines how many outputs correspond correctly to the test data labels
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.where(y == 1.)[0][0]) for x, y in test_data]
        return sum(int(x == y) for (x, y) in test_results)

# initialise neural network
net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
