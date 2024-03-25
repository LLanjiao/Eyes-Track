import cv2
import numpy as np
import dlib
import math

GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 247, 255)
PURPLE = (128, 0, 128)
RED = (0, 0, 255)

fonts = cv2.FONT_HERSHEY_COMPLEX

# dlib 人脸检测器，识别人脸
detectFace = dlib.get_frontal_face_detector()
# dlib 68点人脸识别特征点数据集
predictor = dlib.shape_predictor(
    "predictor/shape_predictor_68_face_landmarks.dat")


class eyes_tracking:
    def __init__(self):
        self.pointList = None
        self.eyesPoints = None
        self.rightEyesPoints = None
        self.leftEyesPoints = None

    def track(self, frame):
        landmarkFrame = frame.copy()
        self.eyesLandmarkPointsExtract(landmarkFrame)
        eyeImage = self.eyesFrameSplit(frame)
        return landmarkFrame, eyeImage

    def eyesLandmarkPointsExtract(self, frame, isDrawRange=True, isDrawPoints=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detectFace(gray)
        # 人脸框对角点坐标
        face = None
        # 图像中可能有多张人脸
        for face in faces:
            # 人脸框对角点坐标赋值
            cord_face1 = (face.left(), face.top())
            cord_face2 = (face.right(), face.bottom())
            # 绘制框
            if isDrawRange:
                # cv.rectangle 在image 上通过对角点绘制矩形，设置颜色，框粗细
                cv2.rectangle(frame, cord_face1, cord_face2, GREEN, 2)

        landmarks = predictor(gray, face)
        pointList = []
        # 遍历得到的特征点集，存放在pointList 中，并在image 上绘制
        for n in range(0, 68):
            point = (landmarks.part(n).x, landmarks.part(n).y)
            pointList.append(point)
            if isDrawPoints:
                # cv.circle在image 上以给定圆心和半径作圆
                cv2.circle(frame, point, 1, ORANGE, 1)
        self.pointList = pointList

    def eyesFrameSplit(self, frame):
        self.eyesPoints = self.pointList[36:48]
        self.rightEyesPoints = self.pointList[36:42]
        self.leftEyesPoints = self.pointList[42:48]

        maxX = (sorted(self.eyesPoints, key=lambda item: item[0], reverse=True))[0][0]
        minX = (sorted(self.eyesPoints, key=lambda item: item[0]))[0][0]
        maxY = (sorted(self.eyesPoints, key=lambda item: item[1], reverse=True))[0][1]
        minY = (sorted(self.eyesPoints, key=lambda item: item[1]))[0][1]

        x = (maxX - minX) % 2
        y = (maxY - minY) % 2
        eyesImage = frame[minY - y:maxY + y, minX - x:maxX + x]
        return eyesImage

    def eyesFrameProcessing(self, frame):
        pass
