import numpy
from random import randint


def cost_function(expecteds, hypothese):
    """
    This function will enable us to see if the cost function has the good behavior.
    :param vectors: list of vectors
    :param expecteds: list of expected results to get after learning
    :param hypothese: list of vectors obtained with the Perceptron
    :return: returns the value of the cost function
    """
    cost_function = 0
    for k in range(len(expecteds)):
        y_k = expecteds[k]
        h_k = hypothese[k]
        cost_k = y_k * numpy.log(h_k) + (1 - y_k) * numpy.log(1 - h_k)
        cost_function += cost_k
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

    def training(self, learning_rate, vectors, expecteds, nb_iteration, epoch):
        """
        :param learning_rate:
        :param vectors: list of vectors
        :param expecteds: the list of 0 or 1 if there is a match or not
        :return: This function computes the weights and biases for the next step determination
        of the minimum of cost function
        """
        cost_list = numpy.zeros([epoch])
        for k in range(epoch):
            #loop till we obtained the minimum
            delta_weights = numpy.zeros([])
            delta_biaises = numpy.zeros([])
            cost = 0
            for i in range(nb_iteration):
                # loop to avoid the zigzags

                chosen_sample = randint(0, len(vectors) - 1)
                cost_i = cost_function(expecteds[chosen_sample], self.forward_propagation(vectors[chosen_sample]))
                delta_weights_i , delta_biaises_i = self.backward_propagation(vectors[chosen_sample],
                                                                              expecteds[chosen_sample])
                delta_biaises += delta_biaises_i
                delta_weights += delta_biaises_i
                cost += cost_i
            cost *= 1 / nb_iteration
            cost_list[k] = cost
            self.weights += learning_rate * delta_weights / nb_iteration
            self.biases += learning_rate * delta_biaises / nb_iteration
        return cost_list






