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
        vect = numpy.copy(vector)
        for loop in range(len(self.layers)-1):
            vect = numpy.dot(self.weights[loop],vect) + self.biases[loop]
        return vect

    def backward_propagation(self, vector, expected):
        pass

    def training(self, vectors, expecteds):
        pass


        
