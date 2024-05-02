import sys

from PyQt6.QtWidgets import QApplication

from gui.application_mainWindow import MainWindow

from settings import settings


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Eye Tracking")
    window.show()
    sys.exit(app.exec())
