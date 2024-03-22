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

    def faceDetector(self, image, gray, draw=True):
        """
        输入原图像与对应的灰度图像，识别并用绿色方框绘制出人脸
        :param image: 图像
        :param gray: 灰度图像
        :param draw: 判断是否绘制，默认为是
        :return: 绘制人脸框后的图像，检测器输出的矩形
        """
        # 对灰度图像进行人脸检测处理
        faces = detectFace(gray)
        # 人脸框对角点坐标
        cord_face1 = (0, 0)
        cord_face2 = (0, 0)
        face = None
        for face in faces:
            # 人脸框对角点坐标赋值
            cord_face1 = (face.left(), face.top())
            cord_face2 = (face.right(), face.bottom())
            # 绘制框
            if draw:
                image_painted = image.copy()
                # cv.rectangle 在image 上通过对角点绘制矩形，设置颜色，框粗细
                cv2.rectangle(image_painted, cord_face1, cord_face2, GREEN, 2)

        return image_painted, face