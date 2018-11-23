import cv2
import os
import numpy

from matplotlib import pyplot
from skimage.feature import hog

from perceptron import *

import json
import urllib.request


########################################################################################################################
#  PLOT FUNCTION                                                                                                       #
########################################################################################################################

def plot_cost_function(network_folder):
    """
    Loads all the traces of the cost function over each iteration and plots it
    :param network_folder: the path to the network folder
    """

    file_names = os.listdir(network_folder + "costs_lists/")

    max_cost = 0

    for file_name in file_names:
        if file_name.startswith("costs_"):
            index = int(file_name.split("_")[1].split(".")[0])

            if index > max_cost:
                max_cost = index

    costs = []

    for index in range(max_cost):
        file = open(network_folder + "costs_lists/costs_" + str(index) + ".vec", "r")
        vector_serialized = file.readline()

        file.close()

        for cost in vector_serialized.split(";"):
            if cost != "":
                costs.append(numpy.log10(float(cost)))

    pyplot.plot(range(len(costs)), costs, label="Cost function")
    pyplot.legend()
    pyplot.grid()
    pyplot.show()


########################################################################################################################
#  DATABASE FUNCTIONS                                                                                                  #
########################################################################################################################


def create_vector_database(database_path, image_folder, xml_path, separator="#;,"):
    """
    Creates the vector database using the image folder
    :param database_path: the path to the database file
    :param image_folder: the path to the image folder
    :param xml_path: the path to the XML folder
    :param separator: the list of separators used in the file
    """

    file_names = os.listdir(image_folder)

    images = {}

    for file_name in file_names:
        label = file_name.split("_0")[0]

        if label not in images:
            images[label] = []

        image = cv2.imread(image_folder + file_name, cv2.IMREAD_GRAYSCALE)

        to_convert = face_detection(image, xml_path)

        for face in to_convert:
            images[label].append(face)

            cv2.imwrite("P:/coding_weeks/machine_learning/repo/database/images/pred/" + label + str(random.uniform(0, 1000)) + ".png", face)

    write = ""

    for label in images:
        if len(images[label]) > 0:
            write += label + separator[0]
            line = ""

            for image in images[label]:
                serialized_vector = ""
                image = cv2.resize(image, (128, 128))
                vector = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))

                for k in range(len(vector)):
                    serialized_vector += str(vector[k]) + separator[2]

                line += serialized_vector[:-1] + separator[1]

            write += line[:-1] + "\n"

    file = open(database_path, "w")
    file.write(write)
    file.flush()
    file.close()


def load_vector_database(database_path, separator="#;,"):
    """
    Loads the vectors from the database
    :param database_path:
    :param separator:
    :return:
    """

    file = open(database_path, "r")
    line = file.readline()

    images = {}

    while line != "":
        label, serialized_vectors = line.split(separator[0])

        vectors = []

        for serialized_vector in serialized_vectors.split(separator[1]):
            coordinates = serialized_vector.split(separator[2])

            vector = numpy.zeros((len(coordinates), 1))

            for index in range(len(coordinates)):
                vector[index, 0] = float(coordinates[index])

            vectors.append(vector)

        images[label] = vectors

        line = file.readline()

    return images


########################################################################################################################
#  IMAGE PROCESSING FUNCTIONS                                                                                          #
########################################################################################################################


def face_detection(image, xml_path):
    """
    Shows the faces on an image and recognize the eyes and the nose
    :param image: image to scale and work with
    :param xml_path: the path to the XML folder
    :return: Noting to return
    """

    face_cascade = cv2.CascadeClassifier(xml_path)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    images = []

    for face in faces:
        x_beginning, y_beginning, face_width, face_height = face
        roi_img = image[y_beginning:y_beginning + face_height, x_beginning:x_beginning + face_width]

        images.append(roi_img)

    return images


########################################################################################################################
#  NETWORK TRAINING FUNCTIONS                                                                                          #
########################################################################################################################


def train_network(network_folder, layers, samples, epochs_per_saves=1000, batch_size=100, learning_rate=1.0, momentum=0.1, adaptive_factor=0.7):
    """
    Creates and trains the network over the given samples
    :param network_folder: the path to the network folder
    :param layers: the structure of the network
    :param epochs_per_saves: the number of epochs done before saving the version of the network
    :param batch_size: the size of each batch
    :param learning_rate: the stating learning rate
    :param momentum: the momentum used in the backprop algorithm
    :param adaptive_factor: decreases the learning rate if the last epochs where unsuccessful
    """

    network = MultiPerceptron(layers)
    network.randomize(-1.0, 1.0)

    train = 0
    alpha = learning_rate

    while True:
        costs = network.training(samples, epochs_per_saves, batch_size, alpha, momentum)

        if costs[-1] > costs[0]:
            alpha *= adaptive_factor

        file = open(network_folder + "costs_lists/costs_" + str(train) + ".vec", "w")

        to_write = ""

        for cost in costs:
            to_write += str(cost) + ";"

        file.write(to_write)
        file.flush()
        file.close()

        save_network(network, network_folder + "network_versions/network_" + str(train) + "_a" + str(alpha) + "_c" + str(costs[-1]) + ".nn")

        train += 1

        test_network_vector(network)


def create_sample_data(database_path):
    """
    Generates the samples list stored in the specified database
    :param database_path: the path to the database
    :return: the samples in a list
    """

    data = load_vector_database(database_path)

    samples = []
    index = 0

    for label in data:
        expected = numpy.zeros((len(data), 1))
        expected[index, 0] = 1.0

        for input in data[label]:
            samples.append([input, expected])

        index += 1

    return samples


########################################################################################################################
#  NETWORK TESTING FUNCTIONS                                                                                           #
########################################################################################################################


def test_network_vector(network):
    test_images = load_vector_database("P:/coding_weeks/machine_learning/repo/database/test_database.vdb")
    known_images = load_vector_database("P:/coding_weeks/machine_learning/repo/database/test_database.vdb")

    known_labels = []

    for label in known_images:
        known_labels.append(label)

    count = 0
    tries = 0

    for label in test_images:
        for image in test_images[label]:
            guess = network.forward_propagation(image)

            max_index = 0
            max_value = guess[0, 0]

            for k in range(len(known_labels)):
                if guess[k, 0] > max_value:
                    max_index = k
                    max_value = guess[k, 0]

            tries += 1

            if max_value < 0.5:
                if label not in known_labels:
                    count += 1

                    print("GUESSED UNKNOWN \t EXPECTED UNKNOWN \t ACCURACY {} \t TRUSTED {}".format(str(100.0 * count / tries)[:5], str(100.0 * max_value)[:5]))
                else:
                    print("GUESSED UNKNOWN \t EXPECTED {} \t ACCURACY {} \t TRUSTED {}".format(label, str(100.0 * count / tries)[:5], str(100.0 * max_value)[:5]))

            else:
                if label == known_labels[max_index]:
                    count += 1

                print("GUESSED {} \t EXPECTED {} \t ACCURACY {} \t TRUSTED {}".format(known_labels[max_index], label, str(100.0 * count / tries)[:5], str(100.0 * max_value)[:5]))

    return count / tries


def pick_best_network(network_folder):
    """
    Tests all the networks of the folder
    :param network_folder: the path to the network folder
    :return: the name of the best network
    """

    file_names = os.listdir(network_folder + "network_versions/")

    max_accuracy = 0
    best_network = ""

    for file_name in file_names:
        network = load_network(network_folder + "network_versions/" + file_name)
        accuracy = test_network_vector(network)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_network = file_name

    return best_network


should_plot_cost_function = False
should_create_database_test = False
should_create_database_train = False
should_train_network = False
should_test_network_vector = False
should_pick_network = False
should_json = False

if should_plot_cost_function:
    plot_cost_function("P:/coding_weeks/machine_learning/repo/database/networks/network_128_16_4/")

if should_create_database_test:
    create_vector_database("P:/coding_weeks/machine_learning/repo/database/linkcs_database.vdb", "P:/coding_weeks/machine_learning/repo/database/images_linkCS_killer/", "P:/coding_weeks/machine_learning/repo/database/xml/haarcascade_frontalface_default.xml")

if should_create_database_train:
    create_vector_database("P:/coding_weeks/machine_learning/repo/database/training_database.vdb", "P:/coding_weeks/machine_learning/repo/database/images/training/", "P:/coding_weeks/machine_learning/repo/database/xml/haarcascade_frontalface_default.xml")
    create_vector_database("P:/coding_weeks/machine_learning/repo/database/training_database_double.vdb", "P:/coding_weeks/machine_learning/repo/database/images/training_double/", "P:/coding_weeks/machine_learning/repo/database/xml/haarcascade_frontalface_default.xml")
    create_vector_database("P:/coding_weeks/machine_learning/repo/database/training_database_triple.vdb", "P:/coding_weeks/machine_learning/repo/database/images/training_triple/", "P:/coding_weeks/machine_learning/repo/database/xml/haarcascade_frontalface_default.xml")

if should_train_network:
    train_network("P:/coding_weeks/machine_learning/repo/database/networks/network_128_128_4/", [2048, 128, 128, 5], create_sample_data("P:/coding_weeks/machine_learning/repo/database/test_database.vdb"))

if should_test_network_vector:
    test_network_vector(load_network("P:/coding_weeks/machine_learning/repo/database/trained_networks/network_2048_128_16_4_3.nn"))

if should_pick_network:
    print(pick_best_network("P:/coding_weeks/machine_learning/repo/database/networks/network_128_16_4_tpl/"))

if should_json:
    with open("P:/coding_weeks/machine_learning/repo/database/LinkCS.json", "r") as file:
        users = json.load(file)
    print(len(users))

    for user in users:
        # Check that the user has a profile picture
        if user["ctiPhotoURI"] is None:
            continue
        # Check if the user plays the killer
        killer = False
        memberships = user["memberships"]
        for asso in memberships:
            if asso["association"]["name"] == "Killer Primal":
                killer = True
        name = user["firstName"].upper() + "_" + user["lastName"].upper()
        name = str(name.encode("ASCII", "ignore"))[2:-1]
        print(name + "  killer: {0}".format(killer))
        # Add the image to the database
        if killer:
            resp = urllib.request.urlopen(user["ctiPhotoURI"])
            image = numpy.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cv2.imwrite("P:/coding_weeks/machine_learning/repo/database/images_linkCS_killer/" + name + ".png", image)
