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


path_images = 'Data/lfw/'
path_hog_data = 'Data/HoG_Data_Vec/'
path_hog_img = 'Data/HoG_Data/'

list_dir = os.listdir(path_images)
n = len(list_dir)
t0 = time.time()
progress = 0
for i in range(n):
    list_img = os.listdir(path_images+list_dir[i])
    if not(os.path.isdir(path_hog_img+list_dir[i])):
        os.mkdir(path_hog_img+list_dir[i])
    if not(os.path.isdir(path_hog_data + list_dir[i])):
        os.mkdir(path_hog_data + list_dir[i])
    if int(1000*i/n)/10 != progress:
        progress = int(1000*i/n)/10
        t = time.time()
        print('Transforming ({0}%) \t Time Left : {1} min'.format(progress, round((t - t0)/(i+1)*(n-i)/60)))
    for j in range(len(list_img)):
        if list_img[j] == 'Thumbs.db':
            continue
        img = cv2.imread(path_images + list_dir[i] + '/' + list_img[j], 0)
        if os.path.isfile(path_hog_img + list_dir[i] + '/' + list_img[j]):
            os.remove(path_hog_img + list_dir[i] + '/' + list_img[j])
        hog_data, hog_img = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True)
        if not(os.path.isfile(path_hog_img + list_dir[i] + '/HoG_' + list_img[j])):
            hog_image_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
            plt.imsave(path_hog_img + list_dir[i] + '/HoG_' + list_img[j], hog_image_rescaled, cmap=plt.cm.gray)
        np.save(path_hog_data + list_dir[i] + '/HoG_' + list_img[j][:-4], hog_data)
t = time.time()
print('Done in {0}'.format(t-t0))
