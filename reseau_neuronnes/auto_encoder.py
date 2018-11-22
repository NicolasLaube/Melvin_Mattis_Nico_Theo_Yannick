import os
import cv2

from skimage.feature import hog
from perceptron import *

import database_train

PATH_IMAGES = "../database/lfw_funneled/"
PATH_DATABASE = "../database/vector_database/"
PATH_XML = "../database/xml/haarcascade_frontalface_default.xml"

DATABASE_SEPARATOR = ";"


def convert_samples():
    """
    Converts all the images of the folder into HoG vectors and saves them into the database
    """

    file = open(PATH_DATABASE + "database_part_0.vecl", "w")

    names = os.listdir(PATH_IMAGES)
    index = 0
    count = 0

    for name in names:
        image_files = os.listdir(PATH_IMAGES + "/" + name + "/")

        for image_file in image_files:
            print("Converting {}".format(image_file))

            image = cv2.imread(PATH_IMAGES + "/" + name + "/" + image_file)
            vectors = convert_image_to_vectors(image)

            for vector in vectors:
                vector_str = ""

                for k in range(len(vector)):
                    vector_str += str(vector[k]) + DATABASE_SEPARATOR

                if count == 1000:
                    file.flush()
                    file.close()

                    file = open(PATH_DATABASE + "database_part_" + str(index) + ".vecl", "w")

                    count = 0
                    index += 1

                file.write(vector_str[0:-1] + "\n")

                count += 1


def face_detection(image):
    """
    Applies the OpenCV face detection algorithm to the image and returns the cropped faces
    :param image: an OpenCV image (numpy array)
    :return: the list of faces
    """

    face_cascade = cv2.CascadeClassifier(PATH_XML)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    images = []

    for face in faces:
        x_beginning, y_beginning, face_width, face_height = face
        roi_img = image[y_beginning:y_beginning + face_height, x_beginning:x_beginning + face_width]

        images.append(cv2.resize(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY), (128, 128)))

    return images


def convert_image_to_vectors(image):
    """
    Converts the faces of the image into a 2048-d vector
    :param image: an OpenCV image (numpy array)
    :return: the HoG vector representation of the face
    """

    faces = face_detection(image)
    vectors = []

    for face in faces:
        vectors.append(hog(face, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False))

    return vectors


def load_database():
    """
    Loads the vector HoG representation of each image from the database
    :return: the list of samples, ready to be used in the network
    """

    database_parts = os.listdir(PATH_DATABASE)

    samples = []

    for database_part in database_parts:
        print("Loading {}".format(database_part))

        file = open(PATH_DATABASE + database_part, "r")
        line = file.readline()

        while line != "":
            vector_str = line.split(DATABASE_SEPARATOR)
            vector = numpy.zeros((len(vector_str), 1))

            for k in range(len(vector_str)):
                vector[k, 0] = float(vector_str[k])

            samples.append([vector, vector])

            line = file.readline()

    return samples




SHOULD_CONVERT = False
SHOULD_TRAIN = False
SHOULD_TEST = False
SHOULD_SPLIT = False

if SHOULD_CONVERT:
    convert_samples()

if SHOULD_TRAIN:
    samples = load_database()

    print("Loaded database with {} samples".format(len(samples)))

    layers = [2048, 512, 128, 512, 2048]

    network = MultiPerceptron(layers)
    network.randomize(-1.0, 1.0)

    alpha = 0.5
    epoch = 0

    while True:
        costs = network.training(samples, 100, 100, alpha, 0)

        if costs[0] < costs[-1]:
            alpha *= 0.9

        save_network(network, "../database/networks/autoencoder_" + str(epoch) + ".nn")

        epoch += 1

if SHOULD_TEST:
    samples = load_database()

    print("Loaded database with {} samples".format(len(samples)))

    network = load_network("../database/networks/autoencoder_29.nn")

    cost = 0

    guess = network.forward_propagation(samples[0][0])

    for k in range(2048):
        print(samples[0][0][k, 0], "\t", guess[k, 0])

    for sample in samples:
        guess = network.forward_propagation(sample[0])

        cost += cost_function(guess, sample[1], 0)

    print("Global network cost : {}".format(str(cost / len(samples))))

if SHOULD_SPLIT:
    network = load_network("../database/autoencoder.nn")

    sub_network = network.sub_network(0, 3)

    save_network(sub_network, "../database/encoder.nn")
