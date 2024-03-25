import warnings
import warnings

import numpy
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QFileDialog, QSlider
from PyQt6.uic import loadUi

from frame_sources.image import image
from frame_sources.video import video
from function.tracking import eyes_tracking
from settings import settings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Window(QWidget):
    startCamera: QPushButton
    openFile: QPushButton
    stop: QPushButton
    threshSlider: QSlider

    def __init__(self):
        super(Window, self).__init__()
        loadUi(settings.GUI_FILE_PATH, self)

        self.filePath = None
        self.fileType = None
        self.frame = None
        self.timer = None
        self.frameSources = None
        self.thresh = 50
        self.tracking = eyes_tracking()

        self.startCamera.clicked.connect(self.play)
        self.openFile.clicked.connect(self.fileChoose)
        self.stop.clicked.connect(self.stopPlay)
        self.threshSlider.valueChanged.connect(self.threshChange)

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

    def play(self):
        if self.fileType == "image":
            self.frameSources = image(self.filePath)
        elif self.fileType == "video":
            self.frameSources = video(self.filePath)
        self.frameSources.start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrame)
        self.timer.start(settings.REFRESH_PERIOD)
        # self.nextFrame()

    def nextFrame(self):
        ret, self.frame = self.frameSources.next_frame()
        if not ret:
            self.timer.stop()
        else:
            frame, eyesImage, redWeight, histogram, binary, rightEye = self.tracking.track(self.frame, self.thresh)
            self.display_image(self.opencv_to_qt(frame))
            self.display_image(self.opencv_to_qt(eyesImage), window="eyesImage")
            self.display_image(self.opencv_to_qt(redWeight), window="redWeight")
            self.display_image(self.opencv_to_qt(histogram), window="histogram")
            self.display_image(self.opencv_to_qt(binary), window="binary")
            self.display_image(self.opencv_to_qt(rightEye), window="rightEye")

    def stopPlay(self):
        self.frameSources.stop()
        self.timer.stop()
        self.findLabelbyName("faceFrame").setText("faceFrame")
        self.findLabelbyName("eyesImage").setText("eyesImage")
        self.findLabelbyName("redWeight").setText("redWeight")
        self.findLabelbyName("histogram").setText("histogram")
        self.findLabelbyName("binary").setText("binary")
        self.findLabelbyName("rightEye").setText("rightEye")

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
        print(self.thresh)
