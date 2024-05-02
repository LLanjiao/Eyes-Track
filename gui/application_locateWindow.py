from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton
from PyQt6.uic import loadUi

from settings import settings


class LocateWindow(QWidget):
    pushButton: QPushButton

    def __init__(self, tracking):
        super(LocateWindow, self).__init__()
        loadUi(settings.LOCATEWINDOW_GUI_FILE_PATH, self)

        self.tracking = tracking
        self.timer = None
        self.timeCnt = 3

        self.topLeft = None
        self.topRight = None
        self.bottomLeft = None
        self.bottomRight = None

        self.pushButton.clicked.connect(self.timeController)

    def timeController(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.getPosition)
        self.timer.start(settings.REFRESH_PERIOD_1S)

    def getPosition(self):
        if self.topLeft is None:
            if self.timeCnt != 0:
                self.findLabelbyName("midLabel").setText(f"请注视左上角！\n{self.timeCnt}")
                self.timeCnt -= 1
            else:
                self.topLeft = (self.tracking.leftX, self.tracking.leftY)
                self.findLabelbyName("midLabel").setText(f"左上已定位\n坐标：（{self.topLeft[0]}, {self.topLeft[1]}）")
                self.findLabelbyName("topLeftLabel").setText(f"左上\n左眼定位：（{self.topLeft[0]}, {self.topLeft[1]}）")
                self.timeCnt = 3
        elif self.topRight is None:
            if self.timeCnt != 0:
                self.findLabelbyName("midLabel").setText(f"请注视右上角！\n{self.timeCnt}")
                self.timeCnt -= 1
            else:
                self.topRight = (self.tracking.leftX, self.tracking.leftY)
                self.findLabelbyName("midLabel").setText(f"右上已定位\n坐标：（{self.topRight[0]}, {self.topRight[1]}）")
                self.findLabelbyName("topRightLabel").setText(f"右上\n左眼定位：（{self.topRight[0]}, {self.topRight[1]}）")
                self.timeCnt = 3
        elif self.bottomLeft is None:
            if self.timeCnt != 0:
                self.findLabelbyName("midLabel").setText(f"请注视左下角！\n{self.timeCnt}")
                self.timeCnt -= 1
            else:
                self.bottomLeft = (self.tracking.leftX, self.tracking.leftY)
                self.findLabelbyName("midLabel").setText(f"左下已定位\n坐标：（{self.bottomLeft[0]}, {self.bottomLeft[1]}）")
                self.findLabelbyName("bottomLeftLabel").setText(f"左下\n左眼定位：（{self.bottomLeft[0]}, {self.bottomLeft[1]}）")
                self.timeCnt = 3
        elif self.bottomRight is None:
            if self.timeCnt != 0:
                self.findLabelbyName("midLabel").setText(f"请注视右下角！\n{self.timeCnt}")
                self.timeCnt -= 1
            else:
                self.bottomRight = (self.tracking.leftX, self.tracking.leftY)
                self.findLabelbyName("midLabel").setText(f"右下已定位\n坐标：（{self.bottomRight[0]}, {self.bottomRight[1]}）")
                self.findLabelbyName("bottomRightLabel").setText(f"右下\n左眼定位：（{self.bottomRight[0]}, {self.bottomRight[1]}）")
                self.timeCnt = 3
        else:
            self.findLabelbyName("midLabel").setText(f"定位完成！")

    def findLabelbyName(self, window):
        resLabel: QLabel = getattr(self, window, None)
        if resLabel is None:
            raise ValueError(f"No such display window in gui: {window}")
        return resLabel