import cv2 as cv
import numpy as np
import dlib
import math

GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)

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


def eyesLandmarkPoints(image, point_list, draw=True):
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


def blindDetector(eyes_points):
    top = eyes_points[1:3]
    bottom = eyes_points[4:6]
    top_mid = midPoint(top[0], top[1])
    bottom_mid = midPoint(bottom[0], bottom[1])
