import cv2 as cv
import numpy as np
import dlib
import math

GREEN = (0, 255, 0)

fonts = cv.FONT_HERSHEY_COMPLEX

# dlib 人脸检测器，识别人脸
detectFace = dlib.get_frontal_face_detector()


def faceDetector(image, gray, Draw = True):
    cordFace1 = (0, 0)
    cordFace2 = (0, 0)

    faces = detectFace(gray)
    face = None

    for face in faces:

        cordFace1 = (face.left(), face.top())
        cordFace2 = (face.right(), face.bottom())

        if Draw:
            cv.rectangle(image, cordFace1, cordFace2, GREEN, 2)

    return image, face
