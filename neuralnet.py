import numpy as np


class DenseLayer:

    def relu(x):
        return np.maximum(0, x)
    
    def reluDerivate(x):
        der = np.ones_like(x)
        der[x <= 0] = 0
        return der
    

    def softmax(x):
        exp_scores = np.exp(x)
        probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
        return probs
    
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    def sigmoidDerivate(x):
        sig = DenseLayer.sigmoid(x)
        return sig * (1 - sig)



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
        
        elif act=='sigmoid':
            self.activation = DenseLayer.sigmoid
            self.activationDer = DenseLayer.sigmoidDerivate


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
    # We do 1/2(y - y^) loss func
    def rmsDcDa(predicted, actual):
        return (np.subtract(predicted, actual))
    

    def _get_accuracy(self, predicted, actual):
        count = 0
        for r1, r2 in zip(predicted, actual):
            if np.isclose(r1, r2).all():
                count+=1
        return count * 100/len(predicted)
    

    def rms_cost(self, ypred, y):
        return 0.5 * np.sum(np.power((y.T - ypred.T).T, 2))


    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []
        self.structure = []

        self.setLossFunc('crossentropy')

        self.prevAs = []
        self.prevZz = []
        self.dws = []
        self.dbs = []
        self.learningRate = 0.1

    def setLossFunc(self, lossFunc='rms'):
        if lossFunc == 'rms':
            self.lossFunc = NeuralNet.rmsDcDa
            self.costFunc = NeuralNet.rms_cost
            

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
        
        curDcDa = self.lossFunc(ypred, y) # (samples x num of neurons)

        for layerId, layer in reversed(list(enumerate(self.layers))):

            aPrev = self.prevAs[layerId]
            zCur = self.prevZz[layerId]
            w = self.weights[layerId]

            dw, db, curDcDa = layer.backprop(curDcDa, aPrev, zCur, w)

            self.weights[layerId] = np.subtract(w, (dw * self.learningRate))
            self.biases[layerId] = np.subtract(self.biases[layerId], (db * self.learningRate))

            self.dws.append(dw)
            self.dbs.append(db)
