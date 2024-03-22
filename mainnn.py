import sys
import cv2

from PyQt6.QtWidgets import QApplication

from frame_sources.pic import pic
from gui.application_window import Window


if __name__ == "__main__":
    app = QApplication(sys.argv)
    p = pic('resources/face.png')
    frame = p.start()
    # frame = cv2.imread('../resources/face.png')
    window = Window(frame)
    window.setWindowTitle("Eye Tracking")
    window.show()
    sys.exit(app.exec())