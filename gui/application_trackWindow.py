from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QPushButton, QLabel
from PyQt6.uic import loadUi

from settings import settings


class trackWindow(QWidget):

    startTrack: QPushButton
    return2locate: QPushButton
    exitButton: QPushButton

    def __init__(self, parentWindow):
        super(trackWindow, self).__init__()
        loadUi(settings.TRACKWINDOW_GUI_FILE_PATH, self)

        self.timer = None

        self.topLeft = parentWindow.topLeft_left
        self.topRight = parentWindow.topRight_left
        self.bottomLeft = parentWindow.bottomLeft_left
        self.bottomRight = parentWindow.bottomRight_left

        self.parentWindow = parentWindow

        self.startTrack.clicked.connect(self.start)
        self.return2locate.clicked.connect(self.return2locateWindow)
        self.exitButton.clicked.connect(self.closeWindow)



    def start(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.gazeAreaTrack)
        self.timer.start(settings.REFRESH_PERIOD)

    def gazeAreaTrack(self):
        if self.parentWindow.tracking.haveFace:
            gazePoint = (self.parentWindow.tracking.leftX, self.parentWindow.tracking.leftY)
            area = self.parentWindow.tracking.gazeAreaTrack(self.topLeft, self.topRight, self.bottomLeft, self.bottomRight, gazePoint)
            print(gazePoint)
            self.changeColor(area)
            self.findLabelByName("TopLeft").setText(f"{area}")
        else:
            self.findLabelByName("TopLeft").setText("have no face")

    def changeColor(self, area):
        if area == "left":
            self.findLabelByName("TopLeft").setStyleSheet("background-color: red")
            self.findLabelByName("TopRight").setStyleSheet("")
        elif area == "right":
            self.findLabelByName("TopRight").setStyleSheet("background-color: red")
            self.findLabelByName("TopLeft").setStyleSheet("")

    def findLabelByName(self, window):
        resLabel: QLabel = getattr(self, window, None)
        if resLabel is None:
            raise ValueError(f"No such display window in gui: {window}")
        return resLabel

    def return2locateWindow(self):
        if self.timer is not None:
            self.timer.stop()
        self.close()

    def closeWindow(self):
        self.parentWindow.close()
        if self.timer is not None:
            self.timer.stop()
        self.close()


