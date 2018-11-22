import image_processing
import cv2
import os
import numpy

from matplotlib import pyplot
from skimage.feature import hog


def plot_cost_function(network_folder):
    """
    Loads all the traces of the cost function over each iteration and plots it
    :param network_folder: the path to the network folder
    """

    file_names = os.listdir(network_folder)

    max_cost = 0

    for file_name in file_names:
        if file_name.startswith("costs_"):
            index = int(file_name.split("_")[1].split(".")[0])

            if index > max_cost:
                max_cost = index

    costs = []

    for index in range(max_cost):
        file = open(network_folder + "costs_" + str(index) + ".txt", "r")
        vector_serialized = file.readline()

        file.close()

        for cost in vector_serialized.split(";"):
            if cost != "":
                costs.append(numpy.log10(float(cost)))

    pyplot.plot(range(len(costs)), costs, label="Cost function")
    pyplot.legend()
    pyplot.grid()
    pyplot.show()


def create_vector_database(database_path, image_folder):
    """

    :return:
    """

    


def load_images(image_folder):
    """
    Loads the images of the database and puts labels on them
    :param image_folder: the path to the image folder
    :return: a dict of labels with the assigned image list
    """

    file_names = os.listdir(image_folder)

    images = {}

    for file_name in file_names:
        label = file_name.split("_0")[0]

        if label not in images:
            images[label] = []

        image = cv2.imread(image_folder + file_name, cv2.IMREAD_GRAYSCALE)

        images[label].append(image)

    return images


images = load_images("../database/images/")

count = 0
tries = 0

for label in images:
    for image in images[label]:
        guesses = image_processing.process(image)

        if label in guesses:
            count += 1

        if label == "THEO_COMBEY" and image_processing.LABEL_UNKNOWN in guesses:
            count += 1

        tries += 1

        print("ACCURACY {}".format(str(100.0 * count / tries)[:5]))
