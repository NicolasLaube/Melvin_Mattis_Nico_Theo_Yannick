import numpy
import cv2
import os

from perceptron import *

from skimage.feature import hog


PATH_NETWORK = "../database/networks/trained_0_a1.nn"
PATH_VECTORS = "../database/vectors/"
PATH_XML = "../database/xml/haarcascade_frontalface_default.xml"

LABEL_UNKNOWN = "UNKNOWN"


def load_labels():
    """
    Loads the list of used labels
    :return: the label list
    """

    labels_non_sanitized = os.listdir(PATH_VECTORS)
    labels_sanitized = []

    for label in labels_non_sanitized:
        labels_sanitized.append(label.split(".")[0])

    return labels_sanitized


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


def process(image):
    """
    Applies the hog transformation over the image and feed forward the vector through the network
    :param image: the cv2 image as a numpy array
    :return: the guessed label
    """

    faces, images = face_detection(image)

    if len(images) == 0:
        return []

    labels = []

    for image in images:
        image = cv2.resize(image, (128, 128))
        vector = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=False)

        guess = network.forward_propagation(vector)

        max_index = 0
        max_value = guess[0, 0]

        for k in range(len(labels)):
            if guess[k, 0] > max_value:
                max_index = k
                max_value = guess[k, 0]

        if max_value < 0.5:
            labels.append(LABEL_UNKNOWN)
        else:
            labels.append(loaded_labels[max_index])

    return labels


network = load_network(PATH_NETWORK)
loaded_labels = load_labels()
