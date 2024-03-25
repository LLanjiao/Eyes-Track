import sys
import cv2

from PyQt6.QtWidgets import QApplication

from frame_sources.image import image
from gui.application_window import Window

from settings import settings


if __name__ == "__main__":
    print(settings.CASES_FILE_PATH)
    app = QApplication(sys.argv)
    # p = image('resources/face.png')
    # frame = p.start()
    # frame = cv2.imread('../resources/face.png')
    window = Window()
    window.setWindowTitle("Eye Tracking")
    window.show()
    sys.exit(app.exec())