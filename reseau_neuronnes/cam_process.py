from perceptron import *
from skimage.feature import hog
from database_train import load_vector_database

import cv2
import os


def detection_cam(network_path, xml_path):
    """
    Applies the face detection algorithm oer the webcam stream
    :param network_path:
    """

    files = os.listdir(network_path)

    networks = [load_network(network_path + files[k]) for k in range(len(files))]

    cap = cv2.VideoCapture(0)

    known_images = load_vector_database("P:/coding_weeks/machine_learning/repo/database/training_database.vdb")

    known_labels = []

    for label in known_images:
        known_labels.append(label)

    while True:
        # Capture image par image
        ret, frame = cap.read()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes, faces = face_detection(rgb, xml_path)

        names = []

        for face in faces:
            face = cv2.resize(face, (128, 128))
            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            vector_list = hog(face, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))

            vector = numpy.zeros((len(vector_list), 1))

            for k in range(len(vector_list)):
                vector[k, 0] = vector_list[k]

            # guess = network.forward_propagation(vector)
            #
            # max_index = 0
            # max_value = guess[0, 0]
            #
            # for k in range(len(known_labels)):
            #     if guess[k, 0] > max_value:
            #         max_index = k
            #         max_value = guess[k, 0]
            #
            # if max_value < 0.3:
            #     names.append("UNKNOWN" + str(max_value))
            #
            # else:
            #     names.append(known_labels[max_index] + str(max_value))
            #
            #     print("GUESS {} | TRUSTED {}".format(known_labels[max_index], str(100.0 * max_value)[:5]))

            labels = []

            for network in networks:
                guess = network.forward_propagation(vector)

                max_index = 0
                max_value = guess[0, 0]

                for k in range(len(known_labels)):
                    if guess[k, 0] > max_value:
                        max_index = k
                        max_value = guess[k, 0]

                labels.append(known_labels[max_index])

            labels.sort()

            d = {}

            for label in labels:
                if label not in d:
                    d[label] = 1
                else:
                    d[label] += 1

            max = 0
            label = ""

            for l in d:
                if d[l] > max:
                    max = d[l]
                    label = l

            if max >= 0.8 * len(files):
                names.append(label)
            else:
                names.append("UNKNOWN")

        for ((x_beginning, y_beginning, face_width, face_height), name) in zip(boxes, names):
            cv2.rectangle(frame, (x_beginning, y_beginning), (x_beginning + face_width, y_beginning + face_height), (0, 255, 0), 2)

            cv2.putText(frame, name, (x_beginning, y_beginning), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


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

    return faces, images


detection_cam("../database/multi_net/", "../database/xml/haarcascade_frontalface_default.xml")
