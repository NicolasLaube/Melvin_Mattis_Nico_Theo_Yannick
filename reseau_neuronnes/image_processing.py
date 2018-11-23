import numpy
import cv2
import os

from perceptron import *

from skimage.feature import hog


PATH_NETWORK = "../database/multi_net/"
PATH_DATABASE = "../database/training_database.vdb"
PATH_XML = "../database/xml/haarcascade_frontalface_default.xml"

LABEL_UNKNOWN = "UNKNOWN"


def load_labels(separator="#"):
    """
    Loads the list of used labels
    :return: the label list
    """

    file = open(PATH_DATABASE, "r")
    line = file.readline()

    labels = []

    while line != "":
        labels.append(line.split(separator))

        line = file.readline()

    return line


def face_detection(image):
    """
    Shows the faces on an image and recognize the eyes and the nose
    :param image: image to scale and work with
    :return: Noting to return
    """

    face_cascade = cv2.CascadeClassifier(PATH_XML)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    images = []

    for face in faces:
        x_beginning, y_beginning, face_width, face_height = face
        roi_img = image[y_beginning:y_beginning + face_height, x_beginning:x_beginning + face_width]

        images.append(roi_img)

    return faces, images


def process(source):
    """
    Applies the hog transformation over the image and feed forward the vector through the network
    :param source: the cv2 image as a numpy array
    :return: the guessed label
    """

    faces, images = face_detection(source)

    if len(images) == 0:
        return []

    labels = []

    for image in images:
        image = cv2.resize(image, (128, 128))
        vector_list = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))

        vector = numpy.zeros((len(vector_list), 1))

        for k in range(len(vector_list)):
            vector[k, 0] = vector_list[k]

        names = []

        for network in networks:
            guess = network.forward_propagation(vector)

            max_index = 0
            max_value = guess[0, 0]

            for k in range(len(loaded_labels)):
                if guess[k, 0] > max_value:
                    max_index = k
                    max_value = guess[k, 0]

            names.append(loaded_labels[max_index])

        counter = {}

        for name in names:
            if name not in counter:
                counter[name] = 1
            else:
                counter[name] += 1

        max = 0
        label = ""

        for name in counter:
            if counter[name] > max:
                max = counter[name]
                label = name

        if max >= 0.8 * len(files):
            print(label + ":" + max)
            labels.append(label)
        else:
            labels.append(LABEL_UNKNOWN)

    return labels


files = os.listdir(PATH_NETWORK)

networks = [load_network(PATH_NETWORK + files[k]) for k in range(len(files))]
loaded_labels = load_labels()
