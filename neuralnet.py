import numpy as np


class DenseLayer:

    def relu(x):
        return np.maximum(0, x)
    
    def reluDerivate(x):
        der = np.ones_like(x)
        der[x <= 0] = 0
        return der
    

    def softmax(X):
        exp_scores = np.exp(X)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs


    # The cross entropy func already takes care of it
    def softmaxDer(x):
        return x
    

    def __init__(self, size, activation="relu"):
        self.setActivation(activation)
        self.size = size
    

    def setActivation(self, act='relu'):
        if act=='relu':
            self.activation = DenseLayer.relu
            self.activationDer = DenseLayer.reluDerivate
        
        elif act=='softmax':
            self.activation = DenseLayer.softmax
            self.activationDer = DenseLayer.softmaxDer


    def forwardpass(self, x, w, b):
        z = np.matmul(w, x)
        z = (z.T + b.flatten()).T # (num neurons x samples)
        a = self.activation(z) # (num neurons x samples)
        return a, z
    
    
    # dcda =  (samples x num of neurons)
    # a = (num of neurons x samples)
    def backprop(self, dcDa, aPrev, z, w):
        
        dadz = self.activationDer(z.T) # (samples x num neurons)
        dcdz = dadz * dcDa # (samples x num neurons)
        dw = np.matmul(aPrev, dcdz).T # (next layer neurons x cur layer neurons)
        
        db = np.sum(dcdz, axis=0, keepdims=True).T # (num neurons x samples)
        # print(f'dcdz size: {dcdz.shape}')
        # print(f'db size: {db.shape}')
        prevDcDa = np.matmul(dcdz, w) # ()
        return dw, db, prevDcDa


class NeuralNet:
    

    # Takes in data as [samples x feature] size
    def catCrossEntropyLossDcDa(actual, predicted):
        num_samples = len(actual)

        ## compute the gradient on predictions
        dscores = predicted
        dscores[range(num_samples),actual] -= 1
        dscores /= num_samples

        return dscores
    
    # Takes in data as [samples x feature] size
    # We do 1/2(y - y^) loss func
    def rmsDcDa(actual, predicted):
        return actual - predicted
    

    def _get_accuracy(self, predicted, actual):
        """
        Calculate accuracy after each iteration
        """
        return np.mean(np.argmax(predicted, axis=1)==actual)
    

    def _calculate_loss(self, predicted, actual):
        """
        Calculate cross-entropy loss after each iteration
        """
        samples = len(actual)

        correct_logprobs = -np.log(predicted[range(samples),actual])
        data_loss = np.sum(correct_logprobs)/samples

        return data_loss


    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []
        self.structure = []
        self.lossFunc = NeuralNet.catCrossEntropyLossDcDa
        
        self.prevAs = []
        self.prevZz = []
        self.dws = []
        self.dbs = []
        self.learningRate = 0.1


    def addLayer(self, layer):
        self.layers.append(layer)
        

    def initParams(self, inputDim):

        prevDim = inputDim

        for i in range(len(self.layers)):
            self.weights.append(np.random.uniform(low=-1, high=1, size=(self.layers[i].size, prevDim)))
            self.biases.append(np.zeros((self.layers[i].size, 1)))
            self.structure.append(self.weights[i].shape)

            prevDim = self.layers[i].size


    def forward(self, X):
        a = X
        self.prevAs = []
        self.prevZz = []
        self.dws = []
        self.dbs = []
        
        for layer, weight, bias in zip(self.layers, self.weights, self.biases):
            self.prevAs.append(a)
            a, z = layer.forwardpass(a, weight, bias)
            self.prevZz.append(z)
        return a
    

    def backprop(self, ypred, y):
        
        curDcDa = self.lossFunc(y, ypred) # (samples x num of neurons)

        # print(curDcDa.shape)
        

        for layerId, layer in reversed(list(enumerate(self.layers))):

            aPrev = self.prevAs[layerId]
            zCur = self.prevZz[layerId]
            w = self.weights[layerId]

            dw, db, curDcDa = layer.backprop(curDcDa, aPrev, zCur, w)
            self.weights[layerId] = w - (dw * self.learningRate)
            # print(f'bias multiplication shape: {(db * self.learningRate).shape}')

            # print(f'before update bias: {(self.biases[layerId]).shape}')
            self.biases[layerId] = np.subtract(self.biases[layerId], (db * self.learningRate))

            # print(f'updated bias: {(self.biases[layerId]).shape}')

            self.dws.append(dw)
            self.dbs.append(db)
