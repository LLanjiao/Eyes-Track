from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton
from PyQt6.uic import loadUi

from gui.application_trackWindow import trackWindow
from settings import settings


class LocateWindow(QWidget):
    startLocate: QPushButton
    toTrackWindow: QPushButton
    exitButton: QPushButton

    def __init__(self, tracking):
        super(LocateWindow, self).__init__()
        loadUi(settings.LOCATEWINDOW_GUI_FILE_PATH, self)

        self.tracking = tracking
        self.timer = None
        self.timeCnt = 3

        self.topLeft_left = None
        self.topRight_left = None
        self.bottomLeft_left = None
        self.bottomRight_left = None

        self.topLeft_right = None
        self.topRight_right = None
        self.bottomLeft_right = None
        self.bottomRight_right = None
        self.isLocated = False

        self.trackWindow = None

        self.startLocate.clicked.connect(self.start)
        self.toTrackWindow.clicked.connect(self.openTrackWindow)
        self.exitButton.clicked.connect(self.exit)

    def openTrackWindow(self):
        if self.isLocated:
            self.timer.stop()
            self.trackWindow = trackWindow(self)
            self.trackWindow.setWindowTitle("track")
            self.trackWindow.showFullScreen()
        else:
            self.findLabelByName("message").setText("还未成功进行定位！\n请点击下方按钮进行定位")

    def start(self):
        if self.isLocated:
            self.topLeft_left = None
            self.topRight_left = None
            self.bottomLeft_left = None
            self.bottomRight_left = None

            self.topLeft_right = None
            self.topRight_right = None
            self.bottomLeft_right = None
            self.bottomRight_right = None
            self.isLocated = False
            self.refreshText()
        self.startLocate.setText(f"正在定位")
        self.startLocate.setEnabled(False)
        self.toTrackWindow.setEnabled(False)
        self.timer = QTimer()
        self.timer.timeout.connect(self.getPosition)
        self.timer.start(settings.REFRESH_PERIOD_1S)

    def refreshText(self):
        self.findLabelByName("topLeftLabel").setText(f"左上")
        self.findLabelByName("topRightLabel").setText(f"右上")
        self.findLabelByName("bottomLeftLabel").setText(f"左下")
        self.findLabelByName("bottomRightLabel").setText(f"右下")

    def getPosition(self):
        if self.topLeft_left is None and self.topLeft_right is None:
            if self.timeCnt != 0:
                self.findLabelByName("message").setText(f"请注视左上角！\n{self.timeCnt}")
                self.timeCnt -= 1
            else:
                self.topLeft_left = (self.tracking.leftX, self.tracking.leftY)
                self.topLeft_right = (self.tracking.rightX, self.tracking.rightY)
                self.findLabelByName("message").setText(f"左上已定位"
                                                        f"\n左眼：（{self.topLeft_left[0]}, {self.topLeft_left[1]}）"
                                                        f"\n右眼：（{self.topLeft_right[0]}, {self.topLeft_right[1]}）")
                self.findLabelByName("topLeftLabel").setText(f"左上"
                                                             f"\n左眼定位：（{self.topLeft_left[0]}, {self.topLeft_left[1]}）"
                                                             f"\n右眼定位：（{self.topLeft_right[0]}, {self.topLeft_right[1]}）")
                self.timeCnt = 3
        elif self.topRight_left is None and self.topRight_right is None:
            if self.timeCnt != 0:
                self.findLabelByName("message").setText(f"请注视右上角！\n{self.timeCnt}")
                self.timeCnt -= 1
            else:
                self.topRight_left = (self.tracking.leftX, self.tracking.leftY)
                self.topRight_right = (self.tracking.rightX, self.tracking.rightY)
                self.findLabelByName("message").setText(f"右上已定位"
                                                        f"\n左眼：（{self.topRight_left[0]}, {self.topRight_left[1]}）"
                                                        f"\n右眼：（{self.topRight_right[0]}, {self.topRight_right[1]}）")
                self.findLabelByName("topRightLabel").setText(f"右上"
                                                              f"\n左眼定位：（{self.topRight_left[0]}, {self.topRight_left[1]}）"
                                                              f"\n右眼定位：（{self.topRight_right[0]}, {self.topRight_right[1]}）")
                self.timeCnt = 3
        elif self.bottomLeft_left is None and self.bottomLeft_right is None:
            if self.timeCnt != 0:
                self.findLabelByName("message").setText(f"请注视左下角！\n{self.timeCnt}")
                self.timeCnt -= 1
            else:
                self.bottomLeft_left = (self.tracking.leftX, self.tracking.leftY)
                self.bottomLeft_right = (self.tracking.rightX, self.tracking.rightY)
                self.findLabelByName("message").setText(f"左下已定位"
                                                        f"\n左眼：（{self.bottomLeft_left[0]}, {self.bottomLeft_left[1]}）"
                                                        f"\n右眼：（{self.bottomLeft_right[0]}, {self.bottomLeft_right[1]}）")
                self.findLabelByName("bottomLeftLabel").setText(f"左下"
                                                                f"\n左眼定位：（{self.bottomLeft_left[0]}, {self.bottomLeft_left[1]}）"
                                                                f"\n右眼定位：（{self.bottomLeft_right[0]}, {self.bottomLeft_right[1]}）")
                self.timeCnt = 3
        elif self.bottomRight_left is None and self.bottomRight_right is None:
            if self.timeCnt != 0:
                self.findLabelByName("message").setText(f"请注视右下角！\n{self.timeCnt}")
                self.timeCnt -= 1
            else:
                self.bottomRight_left = (self.tracking.leftX, self.tracking.leftY)
                self.bottomRight_right = (self.tracking.rightX, self.tracking.rightY)
                self.findLabelByName("message").setText(f"右下已定位"
                                                        f"\n左眼：（{self.bottomRight_left[0]}, {self.bottomRight_left[1]}）"
                                                        f"\n右眼：（{self.bottomRight_right[0]}, {self.bottomRight_right[1]}）")
                self.findLabelByName("bottomRightLabel").setText(f"右下"
                                                                 f"\n左眼定位：（{self.bottomRight_left[0]}, {self.bottomRight_left[1]}）"
                                                                 f"\n右眼定位：（{self.bottomRight_right[0]}, {self.bottomRight_right[1]}）")
                self.timeCnt = 3
        else:
            self.timer.stop()
            self.findLabelByName("message").setText(f"定位完成！")
            self.startLocate.setEnabled(True)
            self.toTrackWindow.setEnabled(True)
            self.startLocate.setText(f"重新定位")
            self.isLocated = True

    def exit(self):
        self.close()

    def findLabelByName(self, window):
        resLabel: QLabel = getattr(self, window, None)
        if resLabel is None:
            raise ValueError(f"No such display window in gui: {window}")
        return resLabel
