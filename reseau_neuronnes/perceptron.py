import numpy


class Perceptron:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        self.layers =layers
        for i in range(len(layers) - 1):
            self.weights.append(numpy.zeros([]))

    def forward_propagation(self, vector):
        pass

    def backward_propagation(self, vector, expected):
        pass

    def training(self, vectors, expecteds):
        pass


        
