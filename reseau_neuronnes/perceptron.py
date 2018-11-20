import numpy
import random


class MultiPerceptron:
    def __init__(self, layers):
        """
        Initializes an empty network with the specified sizes of layers
        :param layers: a list with the size (number of neurones) of each layers
        """

        # Initializes local attributes
        self.weights = []
        self.biases = []
        self.layers = layers

        for k in range(len(layers) - 1):
            self.weights.append(numpy.zeros((layers[k+1], layers[k])))
            self.biases.append(numpy.zeros((layers[k+1], 1)))

    def set_weights_and_biases(self, weights, biases):
        """
        Sets the weights and the biases of the network
        :param weights: the list of weights matrices
        :param biases: the list of biases vectors
        """

        self.weights = numpy.copy(weights)
        self.biases = numpy.copy(biases)

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
        """
        Executes the forward propagation algorithm.
        :param vector: Numpy vector
        :return: Numpy vector
        """

        output = numpy.copy(vector)

        for layer in range(len(self.layers)-1):
            output = sigmoid(numpy.dot(self.weights[layer], output) + self.biases[layer])

        return output

    def backward_propagation(self, vector, expected):
        """
        Performs a backward propagation on a vector, given an expected value.
        :param vector: A numpy columns (n, 1) representing the input x.
        :param expected: A numpy columns (m, 1) representing the expected prediction y.
        :return: Two lists of numpy matrix representing errors on weights and biases.
        """

        network_length = len(self.layers)
        layers_activations = [vector]

        # Forward propagation with steps saved in activations
        for num_layer in range(network_length-1):
            # The weights part of the vector
            vector = sigmoid(numpy.dot(self.weights[num_layer], vector) + self.biases[num_layer])

            # Save the step
            layers_activations.append(vector)

        error_weights = [numpy.zeros((self.layers[i+1], self.layers[i])) for i in range(network_length-1)]
        error_biases = [numpy.zeros((self.layers[i+1], 1)) for i in range(network_length-1)]

        error_biases[-1] = (layers_activations[-1] - expected) * sigmoid_prime(layers_activations[-1])
        error_weights[-1] = numpy.dot(error_biases[-1], numpy.transpose(layers_activations[-2]))

        # Backward propagation with errors saved
        for layer in range(2, network_length):
            # The biases part of the vector (small delta)
            error_biases[-layer] = numpy.dot(numpy.transpose(self.weights[-layer+1]), error_biases[-layer+1])*sigmoid_prime(layers_activations[-layer])

            # The weights part of the vector (capital delta)
            error_weights[-layer] = numpy.dot(error_biases[-layer], numpy.transpose(layers_activations[-layer-1]))

        return error_weights, error_biases

    def training(self, vectors, expected, epochs, batch_size, learning_rate):
        """
        :param learning_rate:
        :param vectors: list of vectors
        :param expected: the list of 0 or 1 if there is a match or not
        :param batch_size: number samples used simultaneously
        :param epochs: number of steps made
        :return: This function computes the weights and biases for the next step determination
        of the minimum of cost function
        """

        cost_list = numpy.zeros([epochs])

        for k in range(epochs):
            # loop till we obtained the minimum
            delta_weights = [numpy.zeros((self.layers[i+1], self.layers[i])) for i in range(len(self.layers)-1)]
            delta_biases = [numpy.zeros((self.layers[i+1], 1)) for i in range(len(self.layers)-1)]
            cost = 0

            for i in range(batch_size):
                # loop to avoid the zigzags
                chosen_sample = random.randint(0, len(vectors) - 1)

                cost_i = cost_function(expected[chosen_sample], self.forward_propagation(vectors[chosen_sample]))
                delta_weights_i, delta_biases_i = self.backward_propagation(vectors[chosen_sample], expected[chosen_sample])

                for j in range(len(self.layers) - 1):
                    delta_biases[j] += delta_biases_i[j]
                    delta_weights[j] += delta_weights_i[j]

                cost += cost_i

            cost *= 1 / batch_size
            cost_list[k] = cost

            for j in range(len(self.layers) - 1):
                self.weights[j] -= learning_rate * delta_weights[j] / batch_size
                self.biases[j] -= learning_rate * delta_biases[j] / batch_size

            print("Epoch {}/{} complete; average cost of the network over this epoch : {}".format(k+1, epochs, cost))

        return cost_list


def sigmoid(z):
    """
    Returns the sigmoid value of z
    :param z: an number or an array
    :return: an number or an array
    """

    return 1 / (1 + numpy.exp(-z))


def sigmoid_prime(z):
    """
    Returns the derivative of the sigmoïd function
    :param z: an number or an array
    :return: an number or an array
    """

    return z * (1 - z)


def cost_function(expected, hypothesis):
    """
    This function will enable us to see if the cost function has the good behavior.
    :param expected: list of expected results to get after learning
    :param hypothesis: list of vectors obtained with the forward propagation
    :return: returns the value of the cost function
    """

    cost = 0
    for k in range(len(expected)):
        y_k = expected[k]
        h_k = hypothesis[k]

        cost -= y_k * numpy.log(h_k) + (1 - y_k) * numpy.log(1 - h_k)

    return cost


def save_network(network, path, separators=";,"):
    """
    Saves a neural network to a file
    :param network: the network to save
    :param path: the path to the file
    :param separators: the list of separators used in the file
    """

    layers = network.layers
    weights = network.weights
    biases = network.biases

    file = open(path, "w")

    for k in range(len(layers)):
        if k != 0:
            file.write(separators[0])

        file.write(str(layers[k]))

    file.write("\n")

    for k in range(len(weights)):
        for i in range(weights[k].shape[0]):
            if i != 0:
                file.write(separators[0])

            for j in range(weights[k].shape[1]):
                if j != 0:
                    file.write(separators[1])

                file.write(str(weights[k][i, j]))

        file.write("\n")

    for k in range(len(biases)):
        for i in range(biases[k].shape[0]):
            if i != 0:
                file.write(separators[0])

            file.write(str(biases[k][i, 0]))

        file.write("\n")

    file.flush()
    file.close()


def load_network(path, separator=";,"):
    """
    Loads a neural network from a file
    :param path: the path to file which contains the network
    :param separator: the list of separators used in the file
    :return: an instance of the loaded neural network
    """

    layers = []
    weights = []
    biases = []

    file = open(path, "r")

    layers_line = file.readline()

    for size in layers_line.split(separator[0]):
        layers.append(int(size))

    for k in range(len(layers) - 1):
        weight = numpy.zeros((layers[k+1], layers[k]), dtype=float)
        line = file.readline()
        nodes = line.split(separator[0])

        for i in range(len(nodes)):
            node = nodes[i].split(separator[1])

            for j in range(len(node)):
                weight[i, j] = float(node[j])

        weights.append(weight)

    for k in range(len(layers) - 1):
        bias = numpy.zeros((layers[k+1], 1), dtype=float)
        line = file.readline()
        nodes = line.split(separator[0])

        for i in range(len(nodes)):
            bias[i, 0] = float(nodes[i])

        biases.append(bias)

    network = MultiPerceptron(layers)
    network.set_weights_and_biases(weights, biases)

    return network


