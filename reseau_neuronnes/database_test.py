import os
import numpy as np
import cv2
from perceptron import *
from perceptron import MultiPerceptron


def proximity_cut(x, ref, thd):
    """Returns a float between 0 and 1 characterising the proximity between a number x
    and a reference ref, being 0 if over a certain threshold thd
    :param x: float
    :param ref: float
    :param thd: float
    :return: float between 0 and 1
    """

    if abs(x-ref) >= thd:
        return 0
    else:
        return 1-abs(x-ref)/thd


# Build the HoG representation of an image
def convert_image_to_hog(image):
    """
    Builds the HoG representation the input image
    :param image: a cv2 image
    :return: HoG a (16, 16, 8) histogram of gradients representation
    """

    img = cv2.resize(image, (128, 128))

    # Compute coordinates of the gradient along x et y
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # Compute angles and magnitudes of the gradient
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Compute the local histogram in a 8x8 pixels square
    hog_data = np.zeros([16, 16, 8])



    for p in range(16):
        for q in range(16):
            for i in range(8):
                for j in range(8):
                    for ang_i in range(8):
                        hog_data[p, q, ang_i] += mag[p*8+i, q*8+j] * proximity_cut(angle[p*8+i, q*8+j], ang_i*20, 20)

    return hog_data, mag


def convert_image_to_vector(image):
    """
    Converts the image to a 2048-d vector
    :param image: cv2 image
    :return: a 2048-d vector
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_data, hog_img = convert_image_to_hog(image)
    vector = np.zeros((2048, 1))

    ind = 0
    for i in range(16):
        for j in range(16):
            for k in range(8):
                vector[ind, 0] = hog_data[i, j, k]
                ind += 1

    return vector


def generate_vectors(image_folder, vector_folder):
    """
    Generates the vectors representations of each image
    :param image_folder: path to the image folder
    :param vector_folder: path to the vector folder
    """

    images = os.listdir(image_folder)

    for name in images:
        print(name)

        image = cv2.imread(image_folder + name)
        vector = convert_image_to_vector(image)

        to_write = ""

        for i in range(2048):
            to_write += str(vector[i, 0]) + "\n"

        file = open(vector_folder + name.split(".")[0] + ".vec", "w")
        file.write(to_write)


def load_vectors(vector_folder):
    """
    Loads the vectors and put labels on them
    :param vector_folder: path to the vector folder
    :return: an array of 2048-d vector
    """

    files = os.listdir(vector_folder)
    vectors = []
    expects = []

    labels = []

    for name in files:
        label = name.split("_")[0]

        if label not in labels:
            labels.append(label)

    for name in files:
        file = open(vector_folder + name, "r")

        vector = np.zeros((2048, 1))

        for i in range(2048):
            vector[i, 0] = float(file.readline())

        vectors.append(vector)

        label = name.split("_")[0]

        expected = np.zeros((len(labels), 1))

        for i in range(len(labels)):
            if labels[i] == label:
                expected[i, 0] = 1

        expects.append(expected)

    return vectors, expects


should_generated_vectors = False
should_train = True
should_test = False

if should_generated_vectors:
    generate_vectors("P:/coding_weeks/machine_learning/repo/database/images/", "P:/coding_weeks/machine_learning/repo/database/vectors/")

if should_train:
    ins, outs = load_vectors("P:/coding_weeks/machine_learning/repo/database/vectors/")

    layers = [2048, 16, 16, outs[0].shape[0]]

    network = MultiPerceptron(layers)
    network.randomize(-1.0, 1.0)

    train = 0
    alpha = 1

    while True:
        costs = network.training(ins, outs, 1000, 100, alpha)

        if costs[-1] > costs[0]:
            alpha /= 2

        save_network(network, "P:/coding_weeks/machine_learning/repo/database/networks/trained_" + str(train) + "_a" + str(alpha) + ".nn")

        train += 1
