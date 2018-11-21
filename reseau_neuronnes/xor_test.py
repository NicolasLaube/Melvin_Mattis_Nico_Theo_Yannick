from perceptron import *
from perceptron import MultiPerceptron

import matplotlib.pyplot as plt

vectors = [
    numpy.array([[0], [0]]),
    numpy.array([[0], [1]]),
    numpy.array([[1], [0]]),
    numpy.array([[1], [1]])
]

expects = [
    numpy.array([[0]]),
    numpy.array([[1]]),
    numpy.array([[1]]),
    numpy.array([[0]])
]

samples = [
    [numpy.array([[0], [0]]), numpy.array(([0]))],
    [numpy.array([[0], [1]]), numpy.array(([1]))],
    [numpy.array([[1], [0]]), numpy.array(([1]))],
    [numpy.array([[1], [1]]), numpy.array(([0]))]
]

layers = [2, 2, 1]

network = MultiPerceptron(layers)
network.randomize(-1.0, 1.0)

cost_list = network.training(samples, 2000, 100, 0.9)

for k in range(len(vectors)):
    print(network.forward_propagation(vectors[k]))

plt.plot(range(len(cost_list)), cost_list)
plt.grid()

plt.show()
