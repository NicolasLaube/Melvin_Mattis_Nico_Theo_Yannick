import numpy


class Perceptron:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        self.layers = layers
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
            vect = sigmoid(vect)
        return vect

    def backward_propagation(self, vector, expected):
        """
        Performs a backward propagation on a vector, given an expected value.
        :param vector: A numpy columns (n, 1) representing the input x.
        :param expected: A numpy columns (m, 1) representing the expected prediction y.
        :return: Two lists of numpy matrix representing errors on weights and biases.
        """
        k = len(self.layers)
        activations = []
        # Forward propagation with steps saved in activations
        for num_layer in range(k-1):
            # The weights part of the vector
            vector = sigmoid(numpy.dot(self.weights[num_layer], vector) + self.biases[num_layer])
            # Save the step
            activations.append(vector)
        error_weights = [numpy.zeros(self.layers[i+1], self.layers[i]) for i in range(k)]
        error_biases = [numpy.zeros(self.layers[i]) for i in range(k)]
        error_biases[-1] = activations[-1] - expected
        error_weights[-1] = numpy.dot(error_biases[-1], numpy.transpose(activations[-1]))
        # Backward propagation with errors saved
        for layer in range(k-2, 0, -1):
            # The biases part of the vector (small delta)
            error_biases[layer] = numpy.dot(numpy.transpose(self.weights[layer]), error_biases[layer+1])*sigmoid_prime(activations[layer])
            # The weights part of the vector (capital delta)
            error_weights[layer] = numpy.dot(error_biases[layer+1], numpy.transpose(activations[layer]))
        return error_weights, error_biases


    def training(self, vectors, expecteds):
        pass


def sigmoid(z):
    """
    Returns the sigmoïd value of z
    :param z: an number or an array
    :return: an number or an array
    """
    return 1/(1+numpy.exp(-z))


def sigmoid_prime(z):
    """
    Returns the derivative of the sigmoïd function
    :param z: an number or an array
    :return: an number or an array
    """
    return z*(1-z)


