import cv2 as cv
import module

# VideoCapture 调用摄像头，参数0为电脑默认摄像头
cameraId = 0
camera = cv.VideoCapture(0)
# camera.open('Facetracking.mov')
# camera.open('VideoFile.mp4')
# camera.open('http://admin:admin@10.31.151.122:8081/video')

while True:
    # 读取一帧视频，ret 检测是否读取，False 即为最后一帧图像，frame 为图片
    ret, frame = camera.read()

    # frame = cv.imread('face4.png')

    # frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

    Draw = True
    # 将取得的一帧图像转换为灰度图像
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 调用faceDetector 逐帧检测人脸并在图片上画出人脸范围，输出到image
    image, face = module.faceDetector(frame, grayFrame)

    # 如果没有检测到面部就调用faceLandmarkDetector 会令数据类型不符而使程序崩溃
    if face is not None:
        framec = frame.copy()
        image1, pointList = module.faceLandmarkDetector(framec, grayFrame, face, True)
        cv.namedWindow('main')
        cv.imshow('main', image1)
        image, pointList = module.faceLandmarkDetector(frame, grayFrame, face, False)
        rightpoints, leftpoints = module.eyesLandmarkPoints(image, pointList, False)
        pos, color = module.eyesTracking(image, grayFrame, rightpoints)
        # cv.line(image, (30, 90), (100, 90), color[0], 30)
        # cv.putText(image, f'{pos}', (35, 95), module.fonts, 0.6, color[1], 2)

    # imshow 使用窗口显示图像，若没有窗口则创建



    # 等待键入 “q” 以退出循环
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# 释放并关闭窗口
camera.release()
cv.destroyAllWindows()
