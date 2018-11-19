import numpy
from random import randint

def fonction_cout(vectors, expecteds, hypothese):
    """

    :param vectors:
    :param expecteds:
    :param hypothese:
    :return:
    """
    nb_vectors = len(vectors)
    len_vector = len(vectors[0])
    cost_function = 0
    for i in range(nb_vectors):
        for k in range(len_vector):
            y_i_k = expecteds[i, k]
            h_i_k = hypothese[i, k]
            cost_i_k = y_i_k * numpy.log(h_i_k) + (1 - y_i_k) * numpy.log(1 - h_i_k)
            cost_function += cost_i_k
    return cost_function

class Perceptron:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        self.layers = layers
        for i in range(len(layers) - 1):
            self.weights.append(numpy.zeros([]))

    def forward_propagation(self, vector):
        pass

    def backward_propagation(self, vector, expected):
        pass

    def training(self, learning_rate, vectors, expecteds):
        """
        :param learning_rate:
        :param vectors: list of vectors
        :param expecteds: the list of 0 or 1 if there is a match or not
        :return: This function computes the weights and biases for the next step determination
        of the minimum of cost function
        """
        delta_weights = numpy.zeros([])
        delta_biaises = numpy.zeros([])
        nb_iteration = 100
        for i in range(nb_iteration):
            chosen_sample = randint(0, len(vectors) - 1)
            delta_weights_i , delta_biaises_i = self.backward_propagation(vectors[chosen_sample],
                                                                          expecteds[chosen_sample])
            delta_biaises += delta_biaises_i
            delta_weights += delta_biaises_i
        self.weights += learning_rate * delta_weights / nb_iteration
        self.biases += learning_rate * delta_biaises / nb_iteration






