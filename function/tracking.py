import math

import cv2
import numpy as np
import dlib
import sys

from settings import settings
sys.setrecursionlimit(3000)

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
        self.pointList = None  # 68特征点
        self.eyesPoints = None  # 眼睛坐标集
        self.rightEyePoints = None  # 右眼坐标集
        self.leftEyePoints = None  # 左眼坐标集

        self.haveFace = False
        self.leftX = None
        self.leftY = None
        self.rightX = None
        self.rightY = None

        self.binaryStandard = None

    def irisTrack(self, frame, binaryMethod, irisDetectionMethod, thresh):
        """
        输入图像、二值化图像阈值并进行虹膜定位处理

        :param frame: 待处理的图像
        :param binaryMethod: 二值化处理使用的方法
        :param irisDetectionMethod: 虹膜定位使用的方法
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
        self.haveFace = haveFace
        # 检测到人脸即进入人眼提取和处理
        if haveFace:
            # 左右眼提取
            eyeImage_left, eyeImage_right = self.eyesFrameSplit(frame)

            # 左眼处理
            leftBlink = self.blinkDetection(self.leftEyePoints)
            redWeight_left, histogram_left, binary_left = self.eyesFrameProcessing(eyeImage_left, binaryMethod, thresh)
            trackingEye_left, self.leftX, self.leftY = self.irisDetection(binary_left, irisDetectionMethod, eyeImage_left)
            # 右眼处理
            rightBlink = self.blinkDetection(self.rightEyePoints)
            redWeight_right, histogram_right, binary_right = self.eyesFrameProcessing(eyeImage_right, binaryMethod, thresh)
            trackingEye_right, self.rightX, self.rightY = self.irisDetection(binary_right, irisDetectionMethod, eyeImage_right)

            return haveFace, faceLandmarkFrame, \
                eyeImage_left, redWeight_left, histogram_left, binary_left, trackingEye_left, leftBlink, \
                eyeImage_right, redWeight_right, histogram_right, binary_right, trackingEye_right, rightBlink,
        # 未检测到人脸，返回haveFace=False，其他值为None
        else:
            return haveFace, frame, \
                   None, None, None, None, None, None, \
                   None, None, None, None, None, None

    def eyesLandmarkPointsExtract(self, frame, isDrawRange=False, isDrawPoints=True):
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
                cv2.rectangle(frame, cord_face1, cord_face2, YELLOW, 2)
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
                cv2.circle(frame, point, 3, BLACK, 1)
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
        self.leftEyePoints = self.pointList[36:42]
        self.rightEyePoints = self.pointList[42:48]

        # 左眼
        lmaxX = (sorted(self.leftEyePoints, key=lambda item: item[0], reverse=True))[0][0]
        lminX = (sorted(self.leftEyePoints, key=lambda item: item[0]))[0][0]
        lmaxY = (sorted(self.leftEyePoints, key=lambda item: item[1], reverse=True))[0][1]
        lminY = (sorted(self.leftEyePoints, key=lambda item: item[1]))[0][1]
        ly = (lmaxY - lminY) % 2
        eyesImage_left = frame[lminY - ly:lmaxY + ly, lminX:lmaxX]
        eyesImage_left = cv2.resize(eyesImage_left, None, fx=3, fy=3, interpolation=cv2.INTER_AREA)

        # 右眼
        rmaxX = (sorted(self.rightEyePoints, key=lambda item: item[0], reverse=True))[0][0]
        rminX = (sorted(self.rightEyePoints, key=lambda item: item[0]))[0][0]
        rmaxY = (sorted(self.rightEyePoints, key=lambda item: item[1], reverse=True))[0][1]
        rminY = (sorted(self.rightEyePoints, key=lambda item: item[1]))[0][1]
        ry = (rmaxY - rminY) % 2
        eyesImage_right = frame[rminY - ry:rmaxY + ry, rminX:rmaxX]
        eyesImage_right = cv2.resize(eyesImage_right, None, fx=3, fy=3, interpolation=cv2.INTER_AREA)

        return eyesImage_left, eyesImage_right

    def blinkDetection(self, eyePoints):
        """
        眨眼检测
        求出眼上下眼睑特征点（左眼38-39和41-42，右眼44-45和48-47）中心点，求其距离
        眨眼标准距离为上眼睑特征点的距离的一半
        上下距离与标准距离对比，小于标准距离则判为眨眼
        :param eyePoints: 眼睛特征点集
        :return: 是否眨眼，True为眨眼，False为睁眼
        """
        isBlink = False
        top = self.midPoint(eyePoints[1], eyePoints[2])
        bottom = self.midPoint(eyePoints[4], eyePoints[5])
        dis = self.pointDistance(top, bottom)
        standard = self.pointDistance(eyePoints[1], eyePoints[2]) / 2
        if dis < standard:
            isBlink = True
        return isBlink

    @staticmethod
    def midPoint(point1, point2):
        """
        寻找平面两点连线的中心点
        :param point1: 第一点
        :param point2: 第二点
        :return: pointOut输出中心点
        """
        x1, y1 = point1
        x2, y2 = point2
        x_out = int((x1 + x2) / 2)
        y_out = int((y1 + y2) / 2)
        point_out = (x_out, y_out)
        return point_out

    @staticmethod
    def pointDistance(point1, point2):
        """
        使用勾股定理计算平面两点距离
        :param point1: 第一点
        :param point2: 第二点
        :return: distance距离
        """
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distance

    def eyesFrameProcessing(self, eyesImage, binaryMethod, thresh):
        """
        图像处理：
        1.高斯模糊处理
        2.提取红色分量
        3.直方图均衡化
        4.二值化处理

        :param eyesImage: 眼睛图像（左/右眼）
        :param binaryMethod: 二值化方法
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
        eyesImage_blur_redWeight_histogram_binary = None
        if binaryMethod == "useDirectCompare":
            eyesImage_blur_redWeight_histogram_binary = self.binarybyDirectCompare(eyesImage_blur_redWeight_histogram, thresh)
        elif binaryMethod == "useCoarsePositioning":
            eyesImage_blur_redWeight_histogram_binary = self.binarybyCoarsePositioning(eyesImage_blur_redWeight_histogram, thresh)

        return eyesImage_blur_redWeight, eyesImage_blur_redWeight_histogram, eyesImage_blur_redWeight_histogram_binary

    @staticmethod
    def binarybyDirectCompare(gray, thresh):
        """
        直接计算对比灰度值进行二值化处理
        搜寻整张灰度图像，高于阈值的像素设为白色，低于阈值的设为黑色
        :param gray: 灰度图像
        :param thresh: 二值化阈值
        :return: binary二值化图像
        """
        height, width = gray.shape[0:2]
        # 复制一张图片用以遍历二值化
        binary = gray.copy()

        # 遍历每一个像素点
        for row in range(height):
            for col in range(width):
                # 获取到灰度值
                grayscale = gray[row, col]
                # 如果灰度值高于阈值 就等于255,白色
                if grayscale > thresh:
                    binary[row, col] = 255
                # 如果小于阈值，就直接改为0，黑色
                elif grayscale < thresh:
                    binary[row, col] = 0
        return binary

    def binarybyCoarsePositioning(self, gray, thresh):
        """
        粗定位与递归扩散法二值化图像
        先使用圆算子粗定位虹膜中心，再通过递归扩散算法于粗定位中心向四周扩散进行二值化
        :param gray: 灰度图像
        :param thresh: 二值化阈值
        :return: binary二值化图像
        """
        height, width = gray.shape[0:2]
        binary = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8) + 255
        diameter = height
        coarseX, coarseY = self.circleScan(gray, diameter)
        # numpy数组中对应的x、y应为array[y, x]
        self.binaryStandard = gray[coarseY, coarseX]
        self.binaryRecursion(gray, coarseX, coarseY, binary, thresh)
        return binary

    def binaryRecursion(self, gray, x, y, binary, thresh):
        """
        递归扩散算法
        向粗定位点的右、下、左、上搜寻，
        灰度值与粗定位点灰度值之差小于阈值的像素坐标在复制的纯白画布上设为黑色
        :param gray: 灰度图像
        :param x: 横坐标-array[y, x]
        :param y: 纵坐标-array[y, x]
        :param binary: 二值化图像
        :param thresh: 阈值
        :return:
        """
        if int(gray[y, x]) - int(self.binaryStandard) < thresh:
            binary[y, x] = 0
        else:
            return
        if x + 1 < binary.shape[1]:
            if binary[y, x + 1] != 0:
                self.binaryRecursion(gray, x + 1, y, binary, thresh)
        if y + 1 < binary.shape[0]:
            if binary[y + 1, x] != 0:
                self.binaryRecursion(gray, x, y + 1, binary, thresh)
        if x - 1 > 0:
            if binary[y, x - 1] != 0:
                self.binaryRecursion(gray, x - 1, y, binary, thresh)
        if y - 1 > 0:
            if binary[y - 1, x] != 0:
                self.binaryRecursion(gray, x, y - 1, binary, thresh)

    def irisDetection(self, binary, irisDetectionMethod, eyeImage):
        eyesImagePainted = None
        x = None
        y = None
        if irisDetectionMethod == "useHoughCircles":
            eyesImagePainted, x, y, radius = self.irisDetectionByHoughCircles(binary, eyeImage)
        elif irisDetectionMethod == "useOperator":
            eyesImagePainted, x, y, radius = self.irisDetectionByOperator(binary, eyeImage)

        return eyesImagePainted, x, y

    @staticmethod
    def irisDetectionByHoughCircles(binary, eyeImage):
        """
        虹膜定位功能，使用霍夫圆直接检测

        :param binary: 眼部二值化图像
        :param eyeImage: 眼部原图像
        :return: eyesImagePainted绘制虹膜后的眼部图像
        x虹膜中心点x坐标
        y虹膜中心点y坐标
        """
        height, width = binary.shape[0:2]
        minRadius = int(height / 10)
        maxRadius = int(height / 10)
        x = None
        y = None
        radius = None

        canny = 100
        circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 500, param1=canny, param2=5, minRadius=minRadius * 0,
                                   maxRadius=maxRadius * 8)
        eyesImagePainted = eyeImage.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(eyesImagePainted, (i[0], i[1]), i[2], RED, 1)  # 在原图上画圆，圆心，半径，颜色，线框
                cv2.circle(eyesImagePainted, (i[0], i[1]), 2, RED, 1)
                x = i[0]
                y = i[1]
                radius = i[2]
        return eyesImagePainted, x, y, radius

    def irisDetectionByOperator(self, binary, eyeImage):
        """
        使用圆算子算法检测虹膜
        需要定位的黑色部分为0，不需要的白色部分为255，因为圆算子数值为1，只需求出两数组的哈达玛积，最小值即为黑色部分最多的位置，即为要定位的虹膜位置

        :param binary: 二值化图像
        :param eyeImage: 眼睛的原图像
        :return:
        eyesImagePainted绘制出检测的虹膜位置的眼睛图像
        centerX中心点x坐标
        centerY中心点y坐标
        """
        height, width = binary.shape[0:2]
        left = -1
        right = -1
        # 先从左至右再从右至左，检测首先出现黑色的列，两列横坐标差为检测的虹膜直径，中心为圆心横坐标
        for col in range(width):
            for row in range(height):
                if binary[row, col] == 0:
                    left = col
                    break
            if left != -1:
                break
        for col in range(width - 1, -1, -1):
            for row in range(height):
                if binary[row, col] == 0:
                    right = col
                    break
            if right != -1:
                break

        # 直径
        diameter = right - left
        # 半径
        radius = diameter // 2
        # 圆算子扫描图像
        centerX, centerY = self.circleScan(binary, diameter)

        # 复制眼睛原图像以绘制圆
        eyesImagePainted = eyeImage.copy()
        cv2.circle(eyesImagePainted, (centerX, centerY), radius, GREEN, 1)
        return eyesImagePainted, centerX, centerY, radius

    @staticmethod
    def circleScan(image, diameter=None):
        """
        圆算子检测灰度值最低的区域
        需要将图片上下扩大以使圆算子能正确遍历，最后定位XY坐标时需进行相应处理以得到原图像的正确坐标

        :param image: 待检测图像
        :param diameter: 圆算子的直径
        :return: 灰度值最低的圆的XY坐标
        """
        height, width = image.shape[0:2]
        # 半径
        radius = diameter // 2

        # 创建圆算子，背景为0，圆为1
        circleOperator = np.zeros((diameter, diameter), dtype=np.uint8)
        cv2.circle(circleOperator, (radius, radius), radius, (1,), -1)

        # 原图像高度可能小于圆算子高度，对扩大图像以进行圆算子计算
        higherImg = np.zeros((height + diameter + diameter, width), dtype=np.uint8) + 255
        higherImg[diameter:height + diameter, 0:width] = image

        # 在检测到的直径范围内，使用圆算子与二值图像进行乘运算
        # 需要定位的黑色部分为0，不需要的白色部分为255，因为圆算子数值为1，只需求出两数组的哈达玛积，最小值即为黑色部分最多的位置，即为要定位的虹膜位置
        higherHeight, higherWidth = higherImg.shape[0:2]
        minSum = sys.maxsize
        centerX = 0
        centerY = 0
        for top in range(higherHeight - diameter):
            for left in range(higherWidth - diameter):
                mulArr = circleOperator * higherImg[top:top + diameter, left:left + diameter]
                index = np.sum(mulArr)
                if minSum > index:
                    minSum = index
                    centerX = left + radius
                    centerY = top + radius - diameter
        return centerX, centerY

    def gazeAreaTrack(self, topLeft, topRight, bottomLeft, bottomRight, gazePoint):
        length = topRight[0] - topLeft[0]
        midX = topLeft[0] + length
        if gazePoint[0] < midX:
            return "left"
        elif gazePoint[0] >= midX:
            return "right"
