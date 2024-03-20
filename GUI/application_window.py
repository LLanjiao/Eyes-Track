import sys
import warnings
import numpy
import cv2

from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QApplication, QPushButton, QLabel
from PyQt6.uic import loadUi

import input

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Window(QWidget):

    startCamera: QPushButton

    def __init__(self, frame):
        super(Window, self).__init__()
        loadUi('./assets/Eyes-Track.ui', self)

        self.frame = frame

        self.startCamera.clicked.connect(self.start)


    def start(self):
        self.display_image(self.opencv_to_qt(self.frame))


    @staticmethod
    def opencv_to_qt(img) -> QImage:
        """
        Convert OpenCV image to PyQT image
        by changing format to RGB/RGBA from BGR
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
        Display the image on a window - which is a label specified in the GUI .ui file
        """

        display_label: QLabel = getattr(self, window, None)
        if display_label is None:
            raise ValueError(f"No such display window in GUI: {window}")
        display_label.setPixmap(QPixmap.fromImage(img))
        display_label.setScaledContents(True)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    frame = cv2.imread('../Resources/face.png')
    window = Window(frame)
    window.setWindowTitle("Eye Tracking")
    window.show()
    sys.exit(app.exec())
