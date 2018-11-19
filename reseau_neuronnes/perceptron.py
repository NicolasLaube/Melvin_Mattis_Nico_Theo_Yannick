import numpy
import random


class Perceptron:
    def __init__(self, layers):
        """
        Initializes an empty network with the specified sizes of layers
        :param layers: a list with the size (number of neurones) of each layers
        """

        self.weights = []
        self.biases = []

        self.layers = layers

        for k in range(len(layers) - 1):
            self.weights.append(numpy.zeros((layers[k+1], layers[k])))
            self.biases.append(numpy.zeros((layers[k+1], 1)))

    def set_weights_and_biases(self, weights, biases):
        """
        Sets the weights and the biases of the network
        :param weights:
        :param biases:
        :return:
        """

        self.weights = weights
        self.biases = biases

    def randomize(self, low, high):
        """
        Randomizes the weights and the biases of the network
        :param low: the lowest value possible for a weight / bias
        :param high: the highest value possible for a weight / bias
        """

        for k in range(len(self.layers) - 1):
            for i in range(self.layers[k+1]):
                for j in range(self.layers[k]):
                    self.weights[k][i, j] = random.uniform(low, high)

                self.biases[k][i, 0] = random.uniform(low, high)

    def forward_propagation(self, vector):
        pass

    def backward_propagation(self, vector, expected):
        pass

    def training(self, vectors, expecteds):
        pass



