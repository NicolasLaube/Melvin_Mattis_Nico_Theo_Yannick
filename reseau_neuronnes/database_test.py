import os
import numpy as np
import cv2
from perceptron import *
from perceptron import MultiPerceptron
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt


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

    return hog_data


def face_delimitation(image):
    """
    Shows the faces on an image and recognize the eyes and the nose
    :param image: image to scale and work with
    :return: Noting to return
    """
    face_cascade = cv2.CascadeClassifier('../database/xml/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    images = []

    for face in faces:
        (x_beginning, y_beginning, face_width, face_height) = face

        roi_img = image[y_beginning:y_beginning+face_height, x_beginning:x_beginning+face_width]

        images.append(roi_img)

    return images


def generate_vectors(image_folder, vector_folder, hog_folder):
    """
    Generates the vectors representations of each image
    :param image_folder: path to the image folder
    :param vector_folder: path to the vector folder
    """

    images = os.listdir(image_folder)
    for name in images:
        print(name)
        if name == "Thumbs.db":
            continue
        image = cv2.imread(image_folder + name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(face_delimitation(image)) == 0:
            continue
        image = face_delimitation(image)[0]
        image = cv2.resize(image, (128, 128))
        vector, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        to_write = ""

        for i in range(2048):
            to_write += str(vector[i]) + "\n"

        plt.imsave(hog_folder + name, hog_image_rescaled, cmap=plt.cm.gray)
        file = open(vector_folder + name.split(".")[0] + ".vec", "w")
        file.write(to_write)
        file.flush()
        file.close()


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
        label = name.split("_")[0] + "_" + name.split("_")[1]

        if label not in labels:
            labels.append(label)

    for name in files:
        file = open(vector_folder + name, "r")

        vector = np.zeros((2048, 1))

        for i in range(2048):
            vector[i, 0] = float(file.readline())

        vectors.append(vector)

        label = name.split("_")[0] + "_" + name.split("_")[1]

        expected = np.zeros((len(labels), 1))

        for i in range(len(labels)):
            if labels[i] == label:
                expected[i, 0] = 1

        expects.append(expected)

    return vectors, expects


should_generated_vectors = True
should_train = False
should_test = False

if should_generated_vectors:
    generate_vectors("../database/images/", "../database/vectors/", "../database/images_hog/")

if should_train:
    ins, outs = load_vectors("../database/vectors/")
    training_examples = ins[:int(0.6*len(ins))]
    validation_examples = ins[int(0.6*len(ins)):int(0.8*len(ins))]
    test_examples = ins[int(0.8*len(ins)):]
    layers = [2048, 16, 16, outs[0].shape[0]]

    network = MultiPerceptron(layers)
    network.randomize(-1.0, 1.0)

    train = 0
    alpha = 1

    while True:
        costs = network.training(ins, outs, 1000, 100, alpha)

        if costs[-1] > costs[0]:
            alpha /= 2

        save_network(network, "../database/networks/trained_" + str(train) + "_a" + str(alpha) + ".nn")

        train += 1
