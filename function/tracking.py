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
        self.height = None

    def track(self, frame, thresh):
        landmarkFrame = frame.copy()
        self.eyesLandmarkPointsExtract(landmarkFrame)
        eyeImage = self.eyesFrameSplit(frame)
        redWeight, histogram, binary = self.eyesFrameProcessing(eyeImage, thresh)
        rightEye = self.irisDetection(binary, eyeImage)
        return landmarkFrame, eyeImage, redWeight, histogram, binary, rightEye

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

        maxX = (sorted(self.leftEyesPoints, key=lambda item: item[0], reverse=True))[0][0]
        minX = (sorted(self.leftEyesPoints, key=lambda item: item[0]))[0][0]
        maxY = (sorted(self.leftEyesPoints, key=lambda item: item[1], reverse=True))[0][1]
        minY = (sorted(self.leftEyesPoints, key=lambda item: item[1]))[0][1]

        x = (maxX - minX) % 2
        y = (maxY - minY) % 2

        eyesImage = frame[minY - y:maxY + y, minX - x:maxX + x]
        eyesImage = cv2.resize(eyesImage, None, fx=3, fy=3, interpolation=cv2.INTER_AREA)
        return eyesImage

    def eyesFrameProcessing(self, eyesImage, thresh):
        # 高斯模糊处理
        eyesImage_blur = cv2.GaussianBlur(eyesImage, (5, 5), 0)
        # 提取红色分量
        blue, green, eyesImage_blur_redWeight = cv2.split(eyesImage_blur, mv=None)
        # 直方图均衡化
        eyesImage_blur_redWeight_histogram = cv2.equalizeHist(eyesImage_blur_redWeight)

        self.height, width = eyesImage_blur_redWeight_histogram.shape[0:2]
        eyesImage_blur_redWeight_histogram_binary = eyesImage_blur_redWeight_histogram.copy()
        # 遍历每一个像素点
        for row in range(self.height):
            for col in range(width):
                # 获取到灰度值
                gray = eyesImage_blur_redWeight_histogram[row, col]
                # 如果灰度值高于阈值 就等于255最大值
                if gray > thresh:
                    eyesImage_blur_redWeight_histogram_binary[row, col] = 255
                # 如果小于阈值，就直接改为0
                elif gray < thresh:
                    eyesImage_blur_redWeight_histogram_binary[row, col] = 0

        return eyesImage_blur_redWeight, eyesImage_blur_redWeight_histogram, eyesImage_blur_redWeight_histogram_binary

    def irisDetection(self, binary, eyeImage):
        height, width = binary.shape[0:2]
        minRadius = int(height / 10)
        maxRadius = int(height / 10)

        canny = 100
        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 500, param1=canny, param2=5, minRadius=minRadius * 0, maxRadius=maxRadius * 8)
        eyesImagePainted = eyeImage.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(eyesImagePainted, (i[0], i[1]), i[2], RED, 1)  # 在原图上画圆，圆心，半径，颜色，线框
                cv2.circle(eyesImagePainted, (i[0], i[1]), 2, RED, 1)
        return eyesImagePainted
