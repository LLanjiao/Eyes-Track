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


def faceDetector(image, gray, draw=True):
    """
    输入原图像与对应的灰度图像，识别并用绿色方框绘制出人脸
    :param image: 图像
    :param gray: 灰度图像
    :param draw: 判断是否绘制
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
    :param draw: 判断是否绘制
    :return: 绘制后的图像，特征点数组
    """

    # 调用68点特征点数据集，输出特征点集
    landmarks = predictor(gray, face)
    pointList = []

    # 遍历得到的特征点集，存放在pointList 中，并在image 上绘制
    for n in range(0, 68):
        point = (landmarks.part(n).x, landmarks.part(n).y)
        pointList.append(point)
        if draw:
            cv.circle(image, point, 3, ORANGE, 2)

    return image, pointList
