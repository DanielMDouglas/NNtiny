class neuron:
    def __init__(self):
        self.upStreamConnections = []
    def output(self):
        return 1/(1 + pow(2, -sum(thisConnection.w*thisConnection.n1.output() for thisConnection in self.upStreamConnections))) # weighted sum of inputs, through a sigmoid activation function
class connection: 
    def __init__(self, n1, n2):
        self.n1, self.w = n1, 0 # save the upstream node and connection weight
        n2.upStreamConnections.append(self) # hook myself up to the downstream node
class network:
    def __init__(self, layerMap):
        self.neuronMap = [[neuron()for i in range(layer)] for layer in layerMap]
        self.connections = sum([[connection(n1, n2) for n1 in thisLayer for n2 in nextLayer] for thisLayer, nextLayer in zip(self.neuronMap[:-1], self.neuronMap[1:])], start = [])
    def output(self, x):
        for i, xi in enumerate(x):
            self.neuronMap[0][i].output = lambda : xi # output for input neurons is just the input
        return self.neuronMap[-1][-1].output() # output of the network is the output of the last neuron
    def update_weights(self, w):
        for i, wi in enumerate(w):
            self.connections[i].w = wi # update the connection weights from a list of floats
NN = network([2, 2, 1]) # a fully connected network with 2 inputs, one output, and one hidden layer
