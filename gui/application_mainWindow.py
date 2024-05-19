import warnings

import numpy
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QFileDialog, QSlider, QComboBox, QRadioButton
from PyQt6.uic import loadUi

from frame_sources.camera import camera
from frame_sources.image import image
from frame_sources.video import video
from function.tracking import eyes_tracking
from gui.application_locateWindow import LocateWindow
from settings import settings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MainWindow(QWidget):
    camera: QPushButton
    openFile: QPushButton
    stop: QPushButton
    threshSlider: QSlider
    cameraChoose: QComboBox
    locate: QPushButton
    useDirectCompare: QRadioButton
    useCoarsePositioning: QRadioButton
    useHoughCircles: QRadioButton
    useOperator: QRadioButton

    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi(settings.MAINWINDOW_GUI_FILE_PATH, self)

        self.filePath = None
        self.fileType = None
        self.cameraType = None
        self.frame = None
        self.timer = None
        self.frameSources = None
        self.isCameraOpening = False
        self.isPlaying = False
        self.thresh = 50
        self.binaryMethod = "useDirectCompare"
        self.irisDetectionMethod = "useHoughCircles"
        self.tracking = eyes_tracking()

        self.locateWindow = None

        self.camera.clicked.connect(self.cameraController)
        self.openFile.clicked.connect(self.fileChoose)
        self.stop.clicked.connect(self.stopPlay)
        self.threshSlider.valueChanged.connect(self.threshChange)
        self.locate.clicked.connect(self.openLocate)

        self.useDirectCompare.toggled.connect(lambda: self.binaryMethodToggled(self.useDirectCompare))
        self.useCoarsePositioning.toggled.connect(lambda: self.binaryMethodToggled(self.useCoarsePositioning))
        self.useHoughCircles.toggled.connect(lambda: self.irisDetectionMethodToggled(self.useHoughCircles))
        self.useOperator.toggled.connect(lambda: self.irisDetectionMethodToggled(self.useOperator))

    def cameraController(self):
        if self.isCameraOpening:
            self.isCameraOpening = False
            self.cameraType = None
            self.camera.setText("打开摄像头")
            self.stopPlay()
        else:
            self.isCameraOpening = True
            self.fileType = "camera"
            self.cameraType = self.cameraChoose.currentText()
            self.camera.setText("关闭摄像头")
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

    def nextFrame(self):
        ret, self.frame = self.frameSources.next_frame()
        if not ret:
            self.timer.stop()
        else:
            haveFace, frame, \
                eyeImage_left, redWeight_left, histogram_left, binary_left, trackingEye_left, leftBlink, \
                eyeImage_right, redWeight_right, histogram_right, binary_right, trackingEye_right, rightBlink \
                = self.tracking.irisTrack(self.frame, self.binaryMethod, self.irisDetectionMethod, self.thresh)
            if haveFace:
                self.display_image(self.opencv_to_qt(frame))

                if leftBlink:
                    self.findLabelByName("leftBlink").setText("Blink")
                else:
                    self.findLabelByName("leftBlink").setText("Open")
                self.display_image(self.opencv_to_qt(eyeImage_left), window="eyeImage_left")
                self.display_image(self.opencv_to_qt(redWeight_left), window="redWeight_left")
                self.display_image(self.opencv_to_qt(histogram_left), window="histogram_left")
                self.display_image(self.opencv_to_qt(binary_left), window="binary_left")
                self.display_image(self.opencv_to_qt(trackingEye_left), window="irisDetection_left")

                if rightBlink:
                    self.findLabelByName("rightBlink").setText("Blink")
                else:
                    self.findLabelByName("rightBlink").setText("Open")
                self.display_image(self.opencv_to_qt(eyeImage_right), window="eyeImage_right")
                self.display_image(self.opencv_to_qt(redWeight_right), window="redWeight_right")
                self.display_image(self.opencv_to_qt(histogram_right), window="histogram_right")
                self.display_image(self.opencv_to_qt(binary_right), window="binary_right")
                self.display_image(self.opencv_to_qt(trackingEye_right), window="irisDetection_right")
            else:
                self.display_image(self.opencv_to_qt(frame))

    def stopPlay(self):
        if self.isPlaying:
            self.frameSources.stop()
            self.timer.stop()
            self.findLabelByName("faceFrame").setText("原图像")
            self.findLabelByName("eyeImage_left").setText("眼部图像-左眼")
            self.findLabelByName("redWeight_left").setText("红色分量-左眼")
            self.findLabelByName("histogram_left").setText("降噪处理的灰度图像-左眼")
            self.findLabelByName("binary_left").setText("二值化图像-左眼")
            self.findLabelByName("irisDetection_left").setText("虹膜定位-左眼")

            self.findLabelByName("eyeImage_right").setText("眼部图像-右眼")
            self.findLabelByName("redWeight_right").setText("红色分量-右眼")
            self.findLabelByName("histogram_right").setText("降噪处理的灰度图像-右眼")
            self.findLabelByName("binary_right").setText("二值化图像-右眼")
            self.findLabelByName("irisDetection_right").setText("虹膜定位-右眼")

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
        display_label = self.findLabelByName(window)
        display_label.setPixmap(QPixmap.fromImage(img))
        display_label.setScaledContents(True)

    def findLabelByName(self, window):
        resLabel: QLabel = getattr(self, window, None)
        if resLabel is None:
            raise ValueError(f"No such display window in gui: {window}")
        return resLabel

    def threshChange(self):
        self.thresh = self.threshSlider.value()

    def openLocate(self):
        self.locateWindow = LocateWindow(self.tracking)
        self.locateWindow.setWindowTitle("locate")
        self.locateWindow.showFullScreen()

    def binaryMethodToggled(self, choice):
        if choice.isChecked():
            self.binaryMethod = choice.objectName()

    def irisDetectionMethodToggled(self, choice):
        if choice.isChecked():
            self.irisDetectionMethod = choice.objectName()
