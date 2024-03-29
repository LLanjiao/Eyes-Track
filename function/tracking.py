import cv2
import numpy as np
import dlib

from settings import settings

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
predictor = dlib.shape_predictor(str(settings.PREDICTOR_PATH))


class eyes_tracking:
    def __init__(self):
        self.pointList = None       # 68特征点
        self.eyesPoints = None      # 眼睛坐标集
        self.rightEyesPoints = None # 右眼坐标集
        self.leftEyesPoints = None  # 左眼坐标集

    def track(self, frame, thresh):
        """
        输入图像、二值化图像阈值并进行虹膜定位处理

        :param frame: 待处理的图像
        :param thresh: 二值化的阈值
        :return:
        haveFace：是否检测到人脸，如False，则不进行图像变换处理，直接进行返回，其他值的返回均为None
        faceLandmarkFrame：人脸框及特征点绘制图像
        eyeImage_left：左眼图像
        redWeight_left：左眼图像红色分量-灰度图像
        histogram_left：左眼灰度图像直方图均衡化处理
        binary_left：左眼二值化图像
        trackingEye_left：左眼虹膜定位图像
        _right为对应右眼图像
        """
        # 赋值原图像，并在复制图像上进行框、点的绘制
        faceLandmarkFrame = frame.copy()
        # 人脸检测
        haveFace = self.eyesLandmarkPointsExtract(faceLandmarkFrame)
        # 检测到人脸即进入人眼提取和处理
        if haveFace:
            # 左右眼提取
            eyeImage_left, eyeImage_right = self.eyesFrameSplit(frame)

            # 左眼处理
            redWeight_left, histogram_left, binary_left = self.eyesFrameProcessing(eyeImage_left, thresh)
            trackingEye_left = self.irisDetection(binary_left, eyeImage_left)
            # 右眼处理
            redWeight_right, histogram_right, binary_right = self.eyesFrameProcessing(eyeImage_right, thresh)
            trackingEye_right = self.irisDetection(binary_right, eyeImage_right)

            return haveFace, faceLandmarkFrame, \
                   eyeImage_left, redWeight_left, histogram_left, binary_left, trackingEye_left, \
                   eyeImage_right, redWeight_right, histogram_right, binary_right, trackingEye_right
        # 未检测到人脸，返回haceFace=False，其他值为None
        else:
            return haveFace, frame, None, None, None, None, None, None, None, None, None, None

    def eyesLandmarkPointsExtract(self, frame, isDrawRange=True, isDrawPoints=True):
        """
        检测输入中图像的人脸部分，检测68特征点
        并通过输入isDrawRange和isDrawPoints参数选择是否在图像上绘出
        返回haveFace判断是否检测到人脸
        如人脸检测阶段没有检测到人脸，不再进行68点检测，直接返回haveFace=False
        否则返回haveFace=True

        :param frame: 待处理图像
        :param isDrawRange: 是否绘制人脸框
        :param isDrawPoints: 是否绘制68特征点
        :return: haveFace是否检测到人脸
        """
        # 使用dlib人脸识别器检测人脸
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
        # 未检测到人脸则直接返回False
        if face is None:
            return False

        # 使用68特征点检测
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
        return True

    def eyesFrameSplit(self, frame):
        """
        左右眼图像切割提取

        :param frame: 原图像
        :return:
        eyesImage_left：左眼图像
        eyesImage_tight：右眼图像
        """
        self.eyesPoints = self.pointList[36:48]
        self.leftEyesPoints = self.pointList[36:42]
        self.rightEyesPoints = self.pointList[42:48]

        # 左眼
        lmaxX = (sorted(self.leftEyesPoints, key=lambda item: item[0], reverse=True))[0][0]
        lminX = (sorted(self.leftEyesPoints, key=lambda item: item[0]))[0][0]
        lmaxY = (sorted(self.leftEyesPoints, key=lambda item: item[1], reverse=True))[0][1]
        lminY = (sorted(self.leftEyesPoints, key=lambda item: item[1]))[0][1]
        lx = (lmaxX - lminX) % 2
        ly = (lmaxY - lminY) % 2
        eyesImage_left = frame[lminY - ly:lmaxY + ly, lminX - lx:lmaxX + lx]
        eyesImage_left = cv2.resize(eyesImage_left, None, fx=3, fy=3, interpolation=cv2.INTER_AREA)

        # 右眼
        rmaxX = (sorted(self.rightEyesPoints, key=lambda item: item[0], reverse=True))[0][0]
        rminX = (sorted(self.rightEyesPoints, key=lambda item: item[0]))[0][0]
        rmaxY = (sorted(self.rightEyesPoints, key=lambda item: item[1], reverse=True))[0][1]
        rminY = (sorted(self.rightEyesPoints, key=lambda item: item[1]))[0][1]
        rx = (rmaxX - rminX) % 2
        ry = (rmaxY - rminY) % 2
        eyesImage_right = frame[rminY - ry:rmaxY + ry, rminX - rx:rmaxX + rx]
        eyesImage_right = cv2.resize(eyesImage_right, None, fx=3, fy=3, interpolation=cv2.INTER_AREA)

        return eyesImage_left, eyesImage_right

    @staticmethod
    def eyesFrameProcessing(eyesImage, thresh):
        """
        图像处理：
        1.高斯模糊处理
        2.提取红色分量
        3.直方图均衡化
        4.二值化处理

        :param eyesImage: 眼睛图像（左/右眼）
        :param thresh: 二值化阈值
        :return:
        eyesImage_blur_redWeight：红色分量图像
        eyesImage_blur_redWeight_histogram：直方图均衡化后图像
        eyesImage_blur_redWeight_histogram_binary：二值化图像
        """
        # 高斯模糊处理
        eyesImage_blur = cv2.GaussianBlur(eyesImage, (5, 5), 0)
        # 提取红色分量
        blue, green, eyesImage_blur_redWeight = cv2.split(eyesImage_blur, mv=None)
        # 直方图均衡化
        eyesImage_blur_redWeight_histogram = cv2.equalizeHist(eyesImage_blur_redWeight)

        # 二值化处理
        height, width = eyesImage_blur_redWeight_histogram.shape[0:2]
        # 复制一张图片用以遍历二值化
        eyesImage_blur_redWeight_histogram_binary = eyesImage_blur_redWeight_histogram.copy()
        # 遍历每一个像素点
        for row in range(height):
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

    @staticmethod
    def irisDetection(binary, eyeImage):
        """
        虹膜定位功能，目前使用霍夫圆直接检测
        待升级算法

        :param binary: 眼部二值化图像
        :param eyeImage: 眼部原图像
        :return: eyesImagePainted绘制虹膜后的眼部图像
        """
        height, width = binary.shape[0:2]
        minRadius = int(height / 10)
        maxRadius = int(height / 10)

        canny = 100
        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 500, param1=canny, param2=5, minRadius=minRadius * 0,
                                   maxRadius=maxRadius * 8)
        eyesImagePainted = eyeImage.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(eyesImagePainted, (i[0], i[1]), i[2], RED, 1)  # 在原图上画圆，圆心，半径，颜色，线框
                cv2.circle(eyesImagePainted, (i[0], i[1]), 2, RED, 1)
        return eyesImagePainted
