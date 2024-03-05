import cv2 as cv
import module

# VideoCapture 调用摄像头，参数0为电脑默认摄像头
cameraId = 0
camera = cv.VideoCapture(cameraId)

while True:
    # 读取一帧视频，ret 检测是否读取，False 即为最后一帧图像，frame 为图片
    # ret, frame = camera.read()

    frame = cv.imread('../face.png')

    Draw = True
    # 将取得的一帧图像转换为灰度图像
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 调用faceDetector 逐帧检测人脸并在图片上画出人脸范围，输出到image
    image, face = module.faceDetector(frame, grayFrame)

    # 如果没有检测到面部就调用faceLandmarkDetector 会令数据类型不符而使程序崩溃
    if face is not None:
        image, pointList = module.faceLandmarkDetector(frame, grayFrame, face, False)
        rightpoints, leftpoints = module.eyesLandmarkPoints(image, pointList)
        module.eyesTracking(image, grayFrame, rightpoints)

    # imshow 使用窗口显示图像，若没有窗口则创建
    cv.namedWindow('main')
    cv.imshow('main', grayFrame)

    # 等待键入 “q” 以退出循环
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# 释放并关闭窗口
camera.release()
cv.destroyAllWindows()
