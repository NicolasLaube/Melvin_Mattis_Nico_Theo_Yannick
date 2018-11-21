import cv2 as cv


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
        (x, y, w, h) = face
        x_rescaled = x - padding
        y_rescaled = y - 2 * padding
        w_rescaled = w + 2 * padding
        h_rescaled = h + 4 * padding
        if x_rescaled >= 0 and y_rescaled >= 0 and w_rescaled <= len(img) and h_rescaled <= len(img[0]):
            (x_beginning, y_beginning, face_width, face_height) = (x_rescaled, y_rescaled, w_rescaled, h_rescaled)
        else:
            (x_beginning, y_beginning, face_width, face_height) = (0, 0, len(img), len(img[0]))
        roi_img = img[y_beginning:y_beginning+face_height, x_beginning:x_beginning+face_width]
        eyes = eye_cascade.detectMultiScale(roi_img)
        noses = nose_cascade.detectMultiScale(roi_img)
        for (ex_beginning, ey_beginning, eye_width, eye_height) in eyes:
            cv.rectangle(roi_img, (ex_beginning, ey_beginning), (ex_beginning+eye_width, ey_beginning+eye_height), (0, 255, 0), 2)
        for (nx_beginning, ny_beginning, nose_width, nose_height) in noses:
            cv.rectangle(roi_img, (nx_beginning, ny_beginning), (nx_beginning+nose_width, ny_beginning+nose_height), (0, 0, 255), 2)
        cv.imshow('img', roi_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
face_delimitation('../database/images/POUTINE_004.png', 50)
