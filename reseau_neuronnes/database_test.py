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
def img_to_hog(image):
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


def generate_vectors(path):
    """

    :param path:
    :return:
    """

    inputs = []
    outputs = []
    labels = []

    list_dir = os.listdir(path)

    for image_path in list_dir:
        label = image_path.split("_")[0]

        if label not in labels:
            labels.append(label)

    for image_path in list_dir:
        print(image_path)
        print(labels)

        image = cv2.imread(path + image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_data, hog_img = img_to_hog(image)
        vector = np.zeros((16*16*8, 1))

        ind = 0
        for k in range(16):
            for j in range(16):
                for i in range(8):
                    vector[ind, 0] = hog_data[k, j, i]
                    ind += 1

        label = image_path.split("_")[0]

        inputs.append(vector)

        out = np.zeros((len(labels), 1))
        for k in range(len(labels)):
            if labels[k] == label:
                out[k, 0] = 1

        outputs.append(out)

    return inputs, outputs, len(labels)


ins, outs, end = generate_vectors("P:/coding_weeks/machine_learning/repo/database/images/")

print(ins)
print(outs)
print(end)

layers = [2048, 16, 16, end]

network = MultiPerceptron(layers)
network.randomize(-1.0, 1.0)

network.training(0.9, ins, outs, 100, 10000)

for k in range(len(ins)):
    print(network.forward_propagation(ins[k]))
