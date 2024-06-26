import cv2

from frame_sources import frameSources


class camera(frameSources):
    def __init__(self, cameraType):
        self.cameraType = cameraType
        self.cameraID = 0
        self.phoneCameraURL = "http://admin:admin@10.31.162.180:8081/video"
        self.getCamera = None

    def start(self):
        if self.cameraType == "使用电脑摄像头":
            self.getCamera = cv2.VideoCapture(0)
        elif self.cameraType == "使用网络摄像头":
            self.getCamera = cv2.VideoCapture()
            self.getCamera.open(self.phoneCameraURL)

    def next_frame(self):
        ret, frame = self.getCamera.read()
        # 摄像头输入的为镜像图片，对其做水平反转处理
        frame = cv2.flip(frame, 1)
        return ret, frame

    def stop(self):
        self.getCamera.release()
