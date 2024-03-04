import cv2 as cv
import numpy as np

# VideoCapture调用摄像头，参数0为电脑默认摄像头
cameraId = 0
camera = cv.VideoCapture(cameraId)

while True:
    # 读取一帧视频，ret检测是否读取，Flase即为最后一帧图像，frame为图片
    ret, frame = camera.read()

    # imshow使用窗口显示图像，若没有窗口则创建
    cv.imshow('Frame', frame)

    key = cv.waitKey(1)

    if key == ord('q'):
        break

camera.release()
cv.destroyAllWindows()

