import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def face_delimitation(face_path, padding):
    """
    Shows the faces on an image and recognize the eyes and the nose
    :param face_path: path to the image to scale and work with
    :param padding: readjusting parameter
    :return: Noting to return
    """
    face_cascade = cv.CascadeClassifier('../database/haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('../database/haarcascade_eye.xml')
    nose_cascade = cv.CascadeClassifier('../database/haarcascade_nose.xml')
    img = cv.imread(face_path)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for face in faces:
        (x_beginning, y_beginning, face_width, face_height) = face
        roi_img = img[y_beginning:y_beginning+face_height, x_beginning:x_beginning+face_width]
        eyes = eye_cascade.detectMultiScale(roi_img)
        noses = nose_cascade.detectMultiScale(roi_img)
        #points = []
        for (ex_beginning, ey_beginning, eye_width, eye_height) in eyes:
            #points.append((ex_beginning, ey_beginning))
            #points.append((ex_beginning+eye_width, ey_beginning + eye_height))
            cv.rectangle(roi_img, (ex_beginning, ey_beginning), (ex_beginning+eye_width, ey_beginning+eye_height), (0, 255, 0), 2)
        for (nx_beginning, ny_beginning, nose_width, nose_height) in noses:
            #points.append((nx_beginning, ny_beginning))
            #points.append((nx_beginning + nose_width, ny_beginning + nose_height))
            cv.rectangle(roi_img, (nx_beginning, ny_beginning), (nx_beginning+nose_width, ny_beginning+nose_height), (0, 0, 255), 2)
        #rows, cols = len(img), len(img[0])
        #pts1 = np.float32(points)
        #pts2 = np.float32([[0,0], [0, 0], [0, 0], [0, 0]])
        #M = cv.getAffineTransform(pts1, pts2)
        #dst = cv.warpAffine(img, M, (cols, rows))

        #plt.subplot(121),plt.imshow(img), plt.title('Input')
        #plt.subplot(122),plt.imshow(dst),plt.title('Output')
        #plt.show()
        cv.imshow('img', roi_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
face_delimitation('../database/images/MERKEL_003.png', 50)
