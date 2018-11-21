# Created by Mathis Fédérico the 15/11/2018 at 21:40

# File to convert Data in HoG representations

import os
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import cv2
import time
from skimage.feature import hog
from skimage import data, exposure


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
             mag a (128, 128) image of the gradients
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


path_images = 'P:/coding_weeks/repo_commun/facerecognition/data/database/'

list_dir = os.listdir(path_images)

for image_path in list_dir:
    print(image_path)

    image = cv2.imread(path_images + image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_data, hog_img = img_to_hog(image)
    vector = np.zeros((16*16*8, 1))

    ind = 0
    for k in range(16):
        for j in range(16):
            for i in range(8):
                vector[ind, 0] = hog_data[k, j, i]
                ind += 1

    
