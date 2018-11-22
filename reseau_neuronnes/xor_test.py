from perceptron import *
from perceptron import MultiPerceptron
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


samples = [
    [numpy.array([[0], [0]]), numpy.array(([0]))],
    [numpy.array([[0], [1]]), numpy.array(([1]))],
    [numpy.array([[1], [0]]), numpy.array(([1]))],
    [numpy.array([[1], [1]]), numpy.array(([0]))]
]

layers = [2, 128, 128, 1]

network = MultiPerceptron(layers)
network.randomize(-1.0, 1.0)

save_network(network, "../database/temp_xor.nn")

momentum = [0, 0.1]
networks = [load_network("../database/temp_xor.nn") for _ in range(len(momentum))]

for k in range(len(networks)):
    print(id(networks[k]))

costs_lists = [networks[k].training(samples, 1000, 100, 1, momentum[k]) for k in range(len(momentum))]

for k in range(len(momentum)):
    plt.plot(range(len(costs_lists[k])), costs_lists[k], label="eta = " + str(momentum[k]))

plt.legend()
plt.grid()
plt.show()
