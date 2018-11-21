from perceptron import *
from perceptron import MultiPerceptron

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

layers = [2, 2, 1]

network = MultiPerceptron(layers)
network.randomize(-1.0, 1.0)

network.training(0.9, vectors, expects, 100, 10000)

for k in range(len(vectors)):
    print(network.forward_propagation(vectors[k]))
