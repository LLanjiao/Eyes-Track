import sys

from PyQt6.QtWidgets import QApplication

from gui.application_window import Window

from settings import settings


if __name__ == "__main__":
    print(settings.CASES_FILE_PATH)
    app = QApplication(sys.argv)
    window = Window()
    window.setWindowTitle("Eye Tracking")
    window.show()
    sys.exit(app.exec())