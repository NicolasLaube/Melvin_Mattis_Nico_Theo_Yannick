import numpy


class Perceptron:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        self.layers =layers
        for i in range(len(layers) - 1):
            self.weights.append(numpy.zeros([]))

    def forward_propagation(self, vector):
        """
        Executes the forward propagation algorithm.
        :param vector: Numpy vector
        :return: Numpy vector
        """
        for loop in range(len(self.layers)-1):
            vector = self.weights[loop]*vector
        return vector

    def backward_propagation(self, vector, expected):
        pass

    def training(self, vectors, expecteds):
        pass


        
