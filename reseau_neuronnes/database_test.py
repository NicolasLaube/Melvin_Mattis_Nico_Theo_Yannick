import os
import numpy as np
import cv2
from perceptron import *
from perceptron import MultiPerceptron
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
import urllib.request
import json


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
        x_beginning, y_beginning, face_width, face_height = face

        roi_img = image[y_beginning:y_beginning + face_height, x_beginning:x_beginning + face_width]

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
        vector, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        to_write = ""

        for i in range(len(vector)):
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
    samples = []

    labels = []

    for name in files:
        label = name.split("_")[0] + "_" + name.split("_")[1]

        if label not in labels:
            labels.append(label)

    for name in files:
        sample = []
        file = open(vector_folder + name, "r")

        vector = np.zeros((2048, 1))

        for i in range(2048):
            vector[i, 0] = float(file.readline())

        sample.append(vector)

        label = name.split("_")[0] + "_" + name.split("_")[1]

        expected = np.zeros((len(labels), 1))

        for i in range(len(labels)):
            if labels[i] == label:
                expected[i, 0] = 1

        sample.append(expected)
        samples.append(sample)

    return samples, labels


def load_vectors_first_only(vector_folder):
    files = os.listdir(vector_folder)
    samples = []

    labels = []
    processed = []

    for name in files:
        label = name.split("_")[0] + "_" + name.split("_")[1]

        if label not in labels:
            labels.append(label)

    for name in files:
        sample = []
        file = open(vector_folder + name, "r")

        vector = np.zeros((2048, 1))

        for i in range(2048):
            vector[i, 0] = float(file.readline())

        sample.append(vector)

        label = name.split("_")[0] + "_" + name.split("_")[1]

        if label in processed:
            continue

        expected = np.zeros((len(labels), 1))

        for i in range(len(labels)):
            if labels[i] == label:
                expected[i, 0] = 1

        sample.append(expected)
        samples.append(sample)
        processed.append(label)

    return samples, labels


def reduce_sample_space(encoder, samples):
    reduced_samples = []

    for sample in samples:
        reduced_samples.append([encoder.forward_propagation(sample[0]), sample[1]])

    return reduced_samples


should_generated_vectors = False
should_train = False
should_train_v2 = False
should_test = True

if should_generated_vectors:
    with open("../database/LinkCS.json", "r") as file:
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
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cv2.imwrite("../database/images_linkCS_killer/" + name + ".png", image)
    generate_vectors("../database/images_linkCS_killer/", "../database/vectors_linkCS_killer/", "../database/images_hog_killer/")

if should_train_v2:
    samples, labels = load_vectors_first_only("../database/vectors_linkCS_killer_128/")

    print(len(samples), len(labels))

    samples = reduce_sample_space(load_network("../database/encoder.nn"), samples)

    layers = [128, 128, 128, len(labels)]

    print(layers)

    sum = 0

    for k in range(len(layers)-1):
        sum += layers[k] * layers[k+1]

    print(sum)

    network = MultiPerceptron(layers)
    network.randomize(-1.0, 1.0)

    train = 0
    alpha = 1

    while True:
        costs = network.training(samples, 1000, 100, alpha, 0)

        if costs[-1] > costs[0]:
            alpha /= 2

        save_network(network, "../database/networks/trained_" + str(train) + "_a" + str(alpha) + ".nn")

        train += 1

if should_train:
    samples, labels = load_vectors_first_only("../database/vectors_linkCS_killer/")

    print(len(samples), len(labels))

    layers = [2048, 128, 128, len(labels)]

    print(layers)

    sum = 0

    for k in range(len(layers)-1):
        sum += layers[k] * layers[k+1]

    print(sum)

    network = MultiPerceptron(layers)
    network.randomize(-1.0, 1.0)

    train = 0
    alpha = 1

    while True:
        costs = network.training(samples, 1000, 100, alpha, 0)

        if costs[-1] > costs[0]:
            alpha /= 2

        save_network(network, "../database/networks/trained_" + str(train) + "_a" + str(alpha) + ".nn")

        train += 1

if should_test:
    samples, labels = load_vectors("../database/vectors_linkCS_killer/")

    print(labels)

    network = load_network("../database/networks/trained_1_a1.nn")
    acc = 0
    tries = 1

    for sample in samples:
        guess = network.forward_propagation(sample[0])
        label = ""

        max_index = 0
        max_value = guess[0, 0]

        for k in range(len(labels)):
            if guess[k, 0] > max_value:
                max_index = k
                max_value = guess[k, 0]

            if sample[1][k, 0] == 1:
                label = labels[k]

        if max_value < 0.5:
            print("GUESSED UNKNOWN \t EXPECTED {} \t ACCURACY {} \t SKIP".format(label, str(100.0 * acc / tries)[:5]))

            continue

        if sample[1][max_index, 0] == 1:
            acc += 1

        tries += 1

        print("GUESSED {} \t EXPECTED {} \t ACCURACY {} \t TRUSTED {}".format(labels[max_index], label, str(100.0 * acc / tries)[:5], str(100.0 * max_value)[:5]))
