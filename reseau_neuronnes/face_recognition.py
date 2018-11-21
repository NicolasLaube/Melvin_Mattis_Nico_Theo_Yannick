import cv2 as cv


def face_delimitation(face_path):
    """
    Shows the faces on an image and recognize the eyes and the nose
    :param face_path: path to the image to scale and work with
    :return: Noting to return
    """
    face_cascade = cv.CascadeClassifier('../database/haarcascade_frontalface_default.xml')
    img = cv.imread(face_path)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    images = []

    for face in faces:
        (x_beginning, y_beginning, face_width, face_height) = face

        roi_img = img[y_beginning:y_beginning+face_height, x_beginning:x_beginning+face_width]

        images.append(roi_img)

    return images


face_delimitation('../database/images/MERKEL_003.png')
