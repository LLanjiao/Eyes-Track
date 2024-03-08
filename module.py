import cv2 as cv
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

fonts = cv.FONT_HERSHEY_COMPLEX

# dlib 人脸检测器，识别人脸
detectFace = dlib.get_frontal_face_detector()
# dlib 68点人脸识别特征点数据集
predictor = dlib.shape_predictor(
    "Predictor/shape_predictor_68_face_landmarks.dat")


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


def pointDistance(point1, point2):
    """
    使用勾股定理计算平面两点距离
    :param point1: 第一点
    :param point2: 第二点
    :return: distance距离
    """
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
    return distance


def faceDetector(image, gray, draw=True):
    """
    输入原图像与对应的灰度图像，识别并用绿色方框绘制出人脸
    :param image: 图像
    :param gray: 灰度图像
    :param draw: 判断是否绘制，默认为是
    :return: 绘制人脸框后的图像，检测器输出的矩形
    """
    # 对灰度图像进行人脸检测处理
    faces = detectFace(gray)

    cord_face1 = (0, 0)
    cord_face2 = (0, 0)
    face = None

    for face in faces:
        # 人脸框对角点坐标赋值
        cord_face1 = (face.left(), face.top())
        cord_face2 = (face.right(), face.bottom())
        if draw:
            # cv.rectangle 在image 上通过对角点绘制矩形，设置颜色，框粗细
            cv.rectangle(image, cord_face1, cord_face2, GREEN, 2)

    return image, face


def faceLandmarkDetector(image, gray, face, draw=True):
    """
    调用68点人脸特征点数据集计算特征点，并在图像上绘出
    :param image: 原图像
    :param gray: 灰度图像
    :param face: 人脸检测器输出的矩形
    :param draw: 判断是否绘制，默认为是
    :return: 绘制后的图像，特征点数组
    """

    # 调用68点特征点数据集，输出特征点集
    landmarks = predictor(gray, face)
    point_list = []

    # 遍历得到的特征点集，存放在pointList 中，并在image 上绘制
    for n in range(0, 68):
        point = (landmarks.part(n).x, landmarks.part(n).y)
        point_list.append(point)
        if draw:
            # cv.circle在image 上以给定圆心和半径作圆
            cv.circle(image, point, 3, ORANGE, 1)

    return image, point_list


def eyesLandmarkPoints(image, point_list, draw):
    """
    提取68点特征点中的左右眼点集，选择绘制，输出
    :param image: 原图像
    :param point_list: 68点特征点集
    :param draw: 判断是否绘制，默认为是
    :return: 右眼点集，左眼点集
    """
    right_eyes_points = point_list[36:42]
    left_eyes_points = point_list[42:48]
    if draw:
        for right in right_eyes_points:
            cv.circle(image, right, 2, ORANGE, 1)
        for left in left_eyes_points:
            cv.circle(image, left, 2, ORANGE, 1)
    return right_eyes_points, left_eyes_points


def eyesTracking(image, gray, eye_points):
    """
    通过单眼特征点坐标与灰度图像的二值化处理提取眼睛黑色部分
    :param image: 原图片
    :param gray: 灰度图片
    :param eye_points: 单眼特征点坐标
    :return:
    """
    # .shape获取原图像大小
    dim = gray.shape
    # np.zeros创建dim大小的多维数组，默认值为0，表现为纯黑色的灰度图像
    mask = np.zeros(dim, dtype=np.uint8)

    # 使用眼睛特征点坐标集eyes_points 创建np数组
    PollyPoints = np.array(eye_points, dtype=np.int32)
    # cv.fillPoly 在图像mask 以数组PollyPoints 组成的闭合曲线绘制封闭图形，颜色设置为255，纯白色的灰度图像
    cv.fillPoly(mask, [PollyPoints], 255)

    # 特征点绘制的图形与灰度图像进行与运算分离出眼睛的灰度图像
    eye_image = cv.bitwise_and(gray, gray, mask=mask)
    eye_image_blur = cv.GaussianBlur(eye_image, (5, 5), 0)
    # 计算眼睛特征点坐标的最大最小横纵坐标值
    maxX = (sorted(eye_points, key=lambda item: item[0], reverse=True))[0][0]
    SecondMaxX = (sorted(eye_points, key=lambda item: item[0], reverse=True))[1][0]
    minX = (sorted(eye_points, key=lambda item: item[0]))[0][0]
    SecondMinX = (sorted(eye_points, key=lambda item: item[0]))[1][0]
    maxY = (sorted(eye_points, key=lambda item: item[1], reverse=True))[0][1]
    minY = (sorted(eye_points, key=lambda item: item[1]))[0][1]

    # 将眼睛中黑色部分转为白色
    eye_image_blur[mask == 0] = 255



    # 高斯滤波
    image = cv.GaussianBlur(image, (7, 7), 0)
    # 红色分量
    blue, green, red = cv.split(image, mv=None)
    cv.imshow('red', red)

    # 剪裁下眼睛大小的画布
    x = (maxX - minX) % 2
    y = (maxY - minY) % 2
    image_cropped_eye = image[minY - y:maxY + y, minX - x:maxX + x]
    image_cropped_eye = cv.resize(image_cropped_eye, None, fx=3, fy=3, interpolation=cv.INTER_AREA)
    #cv.imshow('image_cropped_eye', image_cropped_eye)

    gray_image_cropped_eye = cv.cvtColor(image_cropped_eye, cv.COLOR_BGR2GRAY)
    cv.imshow('gray1', gray_image_cropped_eye)
    equ = cv.equalizeHist(gray_image_cropped_eye)
    cv.imshow('equ1', equ)






    cropped_eye = eye_image_blur[minY:maxY, minX:maxX]
    height, width = cropped_eye.shape

    # cv.resize 对眼睛图像进行放大以便于观察
    # resize_cropped_eye = cv.resize(cropped_eye, (width * 5, height * 5))
    # cv.threshold 将眼睛的灰度图像转化为二值图像，阈值为100-->后续可添加调试阈值功能以适应不同光照环境
    ret, thresholdEye = cv.threshold(eye_image_blur, 50, 255, cv.THRESH_BINARY)
    ret1, red = cv.threshold(red, 70, 255, cv.THRESH_BINARY)
    cv.imshow('red1', red)

    """
    thresholdEyefix = thresholdEye.copy()
    for col in range(thresholdEyefix.shape[1]):
        if minX < col < maxX:
            white_run_length = 0
            start_index = 0
            end_index = 0
            for row in range(thresholdEyefix.shape[0]):
                if minY < row < maxY:
                    if thresholdEyefix[row, col] == 255:  # 白色像素
                        if white_run_length == 0:
                            start_index = row
                        white_run_length += 1
                        end_index = row
                    else:  # 黑色像素
                        if start_index == 0:
                            white_run_length = 0
                            continue
                        if white_run_length > 0 and white_run_length < height / 2 and (
                                start_index == 0 or thresholdEyefix[start_index - 1, col] == 0) and (
                                end_index == height - 1 or thresholdEyefix[end_index + 1, col] == 0):
                            thresholdEyefix[start_index:end_index + 1, col] = 0  # 将连续的白色像素改为黑色像素
                        white_run_length = 0
    """



    canny = 100


    height, width = equ.shape[0:2]#eye_image_blur
    # 设置阈值
    thresh = 30
    # 遍历每一个像素点
    for row in range(height):
        for col in range(width):
            # 获取到灰度值
            gray = equ[row, col]
            # 如果灰度值高于阈值 就等于255最大值
            if gray > thresh:
                equ[row, col] = 255
            # 如果小于阈值，就直接改为0
            elif gray < thresh:
                equ[row, col] = 0



    # eye_image_blur[1,1] = 1

    minRadius = int(height / 10)
    maxRadius = int(height / 10)


    circles = cv.HoughCircles(equ, cv.HOUGH_GRADIENT, 1, 500, param1=canny, param2=5, minRadius=minRadius*0, maxRadius=maxRadius*8)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(image_cropped_eye, (i[0], i[1]), i[2], RED, 1)  # 在原图上画圆，圆心，半径，颜色，线框
            cv.circle(image_cropped_eye, (i[0], i[1]), 2, RED, 1)

            cv.circle(gray_image_cropped_eye, (i[0], i[1]), i[2], 255, 1)

    cv.imshow('image_cropped_eye', image_cropped_eye)


    """
    thresholdEyefix = thresholdEye.copy()
    for col in range(thresholdEyefix.shape[1]):
        white_run_length = 0
        start_index = 0
        end_index = 0
        for row in range(height):
            if thresholdEyefix[row, col] == 255:  # 白色像素
                if white_run_length == 0:
                    start_index = row
                white_run_length += 1
                end_index = row
            else:  # 黑色像素
                if start_index == 0:
                    white_run_length = 0
                    continue
                if white_run_length > 0 and white_run_length < height / 2 and (
                        start_index == 0 or thresholdEyefix[start_index - 1, col] == 0) and (
                        end_index == height - 1 or thresholdEyefix[end_index + 1, col] == 0):
                    thresholdEyefix[start_index:end_index + 1, col] = 0  # 将连续的白色像素改为黑色像素
                white_run_length = 0
    """

    # 将剪切的二值图像分为三分，根据哪部分的黑色部分更多作为判断眼睛位置的依据
    div_part = int(width / 3)
    right_part = thresholdEye[0:height, 0:div_part]
    center_part = thresholdEye[0:height, div_part:div_part + div_part]
    left_part = thresholdEye[0:height, div_part + div_part:width]
    # 各个部分的黑色点数量
    right_black_px = np.sum(right_part == 0)
    center_black_px = np.sum(center_part == 0)
    left_black_px = np.sum(left_part == 0)
    pos, color = position([right_black_px, center_black_px, left_black_px])

    # 二值图像的轮廓
    edges = cv.Canny(thresholdEye, threshold1=canny/2, threshold2=canny)

    # 提取的眼睛灰度图像
    cv.imshow("cropEye", gray_image_cropped_eye)#eye_image_blur
    # 二值化处理后的眼睛黑色部分
    cv.imshow("thresholdEye", gray_image_cropped_eye)
    # cv.imshow('test', thresholdEyefix)
    # 二值化图像的轮廓
    # cv.imshow("edge", edges)
    return pos, color


def position(point_list):
    """
    通过二值化图像眼睛黑色部分占比判断眼睛位置
    :param point_list: 黑色点数量
    :return: 眼睛方向
    """
    max_index = point_list.index(max(point_list))
    pos_eye = ''
    color = [WHITE, BLACK]
    if max_index == 0:
        pos_eye = "Right"
        color = [YELLOW, BLACK]
    elif max_index == 1:
        pos_eye = "Center"
        color = [BLACK, PURPLE]
    elif max_index == 2:
        pos_eye = "Left"
        color = [RED, BLACK]
    else:
        pos_eye = "Eye Closed"
        color = [BLACK, WHITE]
    return pos_eye, color
