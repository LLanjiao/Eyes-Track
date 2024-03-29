import warnings
import warnings

import cv2
import numpy
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QFileDialog, QSlider, QComboBox
from PyQt6.uic import loadUi

from frame_sources.camera import camera
from frame_sources.image import image
from frame_sources.video import video
from function.tracking import eyes_tracking
from settings import settings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Window(QWidget):
    camera: QPushButton
    openFile: QPushButton
    stop: QPushButton
    threshSlider: QSlider
    cameraChoose: QComboBox

    def __init__(self):
        super(Window, self).__init__()
        loadUi(settings.GUI_FILE_PATH, self)

        self.filePath = None
        self.fileType = None
        self.cameraType = None
        self.frame = None
        self.timer = None
        self.frameSources = None
        self.isCameraOpening = False
        self.isPlaying = False
        self.thresh = 50
        self.tracking = eyes_tracking()

        self.camera.clicked.connect(self.cameraController)
        self.openFile.clicked.connect(self.fileChoose)
        self.stop.clicked.connect(self.stopPlay)
        self.threshSlider.valueChanged.connect(self.threshChange)

    def cameraController(self):
        if self.isCameraOpening:
            self.isCameraOpening = False
            self.cameraType = None
            self.camera.setText("openCamera")
            self.stopPlay()
        else:
            self.isCameraOpening = True
            self.fileType = "camera"
            self.cameraType = self.cameraChoose.currentText()
            self.camera.setText("closeCamera")
            self.play()

    def fileChoose(self):
        """
        打开文件浏览器，选择视频或图片文件，设置filePath、fileType
        """
        fd = QFileDialog()
        # 设置过滤器
        fileFilters = "image(*.png *.xpm *.jpg);; video(*.mov *.mp4)"
        # 静态方法打开文件浏览器
        response = fd.getOpenFileName(
            parent=self,
            caption="openFile",
            directory=str(settings.CASES_FILE_PATH),
            filter=fileFilters,
        )
        self.filePath = response[0]
        if response[1] == "image(*.png *.xpm *.jpg)":
            self.fileType = "image"
        elif response[1] == "video(*.mov *.mp4)":
            self.fileType = "video"
        if self.fileType is not None:
            self.play()

    def play(self):
        if self.fileType == "image":
            self.frameSources = image(self.filePath)
        elif self.fileType == "video":
            self.frameSources = video(self.filePath)
        elif self.fileType == "camera":
            self.frameSources = camera(self.cameraType)
        self.frameSources.start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrame)
        self.timer.start(settings.REFRESH_PERIOD)
        self.isPlaying = True
        # self.nextFrame()

    def nextFrame(self):
        ret, self.frame = self.frameSources.next_frame()
        if not ret:
            self.timer.stop()
        else:
            haveFace, frame, eyeImage_left, redWeight_left, histogram_left, binary_left, trackingEye_left, \
            eyeImage_right, redWeight_right, histogram_right, binary_right, trackingEye_right \
                = self.tracking.track(self.frame, self.thresh)
            if haveFace:
                self.display_image(self.opencv_to_qt(frame))

                self.display_image(self.opencv_to_qt(eyeImage_left), window="eyeImage_left")
                self.display_image(self.opencv_to_qt(redWeight_left), window="redWeight_left")
                self.display_image(self.opencv_to_qt(histogram_left), window="histogram_left")
                self.display_image(self.opencv_to_qt(binary_left), window="binary_left")
                self.display_image(self.opencv_to_qt(trackingEye_left), window="tracking_left")

                self.display_image(self.opencv_to_qt(eyeImage_right), window="eyeImage_right")
                self.display_image(self.opencv_to_qt(redWeight_right), window="redWeight_right")
                self.display_image(self.opencv_to_qt(histogram_right), window="histogram_right")
                self.display_image(self.opencv_to_qt(binary_right), window="binary_right")
                self.display_image(self.opencv_to_qt(trackingEye_right), window="tracking_right")
            else:
                self.display_image(self.opencv_to_qt(frame))

    def stopPlay(self):
        if self.isPlaying:
            self.frameSources.stop()
            self.timer.stop()
            self.findLabelbyName("faceFrame").setText("faceFrame")
            self.findLabelbyName("eyeImage_left").setText("eyeImage_left")
            self.findLabelbyName("redWeight_left").setText("redWeight_left")
            self.findLabelbyName("histogram_left").setText("histogram_left")
            self.findLabelbyName("binary_left").setText("binary_left")
            self.findLabelbyName("tracking_left").setText("tracking_left")

            self.findLabelbyName("eyeImage_right").setText("eyesImage_right")
            self.findLabelbyName("redWeight_right").setText("redWeight_right")
            self.findLabelbyName("histogram_right").setText("histogram_right")
            self.findLabelbyName("binary_right").setText("binary_right")
            self.findLabelbyName("tracking_right").setText("tracking_right")

            self.filePath = None
            self.fileType = None
            self.frame = None
            self.frameSources = None
            self.isPlaying = False

    @staticmethod
    def opencv_to_qt(img) -> QImage:
        """
        OpenCV图像格式转换为Qt格式，以在gui上显示
        """
        qformat = QImage.Format.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                qformat = QImage.Format.Format_RGBA8888
            else:  # RGB
                qformat = QImage.Format.Format_RGB888

        img = numpy.require(img, numpy.uint8, "C")
        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)  # BGR to RGB
        out_image = out_image.rgbSwapped()
        return out_image

    def display_image(self, img: QImage, window="faceFrame"):
        """
        显示图片
        """
        display_label = self.findLabelbyName(window)
        display_label.setPixmap(QPixmap.fromImage(img))
        display_label.setScaledContents(True)

    def findLabelbyName(self, window):
        resLabel: QLabel = getattr(self, window, None)
        if resLabel is None:
            raise ValueError(f"No such display window in gui: {window}")
        return resLabel

    def threshChange(self):
        self.thresh = self.threshSlider.value()
