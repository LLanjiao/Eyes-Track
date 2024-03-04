import cv2 as cv
import numpy as np
import module

# VideoCapture 调用摄像头，参数0为电脑默认摄像头
cameraId = 0
camera = cv.VideoCapture(cameraId)

while True:
    # 读取一帧视频，ret 检测是否读取，False 即为最后一帧图像，frame 为图片
    ret, frame = camera.read()

    Draw = True
    # 将取得的一帧图像转换为灰度图像
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 调用faceDetector 逐帧检测人脸并在图片上画出人脸范围，输出到image
    image, face = module.faceDetector(frame, grayFrame)

    # imshow 使用窗口显示图像，若没有窗口则创建
    cv.imshow('Frame', image)

    # 等待键入 “q” 以退出循环
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# 释放并关闭窗口
camera.release()
cv.destroyAllWindows()
