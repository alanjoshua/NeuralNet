import numpy as np


class DenseLayer:

    def relu(x):
        np.maximum(0, x)
    
    def softmax(X):
        exp_scores = np.exp(X)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
    

    def __init__(self, size, activation=relu):
        self.size = size
        self.activation = activation

    def forwardpass(self, x, w, b):
        return self.activation(np.matmul(w, x) + b)
    
    
    def backprop(self):
        print('Not implemented yet')


class NeuralNet:

    def __init__(self, layers=[]):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.structure = []

    def addLayer(self, layer):
        self.layers.append(layer)


    def initParams(self, inputDim):

        prevDim = inputDim

        for i in range(len(self.layers)):
            self.weights.append(np.random.uniform(low=-1, high=1, size=(self.layers[i].size, prevDim)))
            self.biases.append(np.zeros(self.layers[i].size))
            self.structure.append(self.weights[i].shape)

            prevDim = self.layers[i].size


    def forward(self, X):
        a = X
        for layer, weight, bias in zip(self.layers, self.weights, self.biases):
            a = layer.forwardpass(X, weight, bias)

