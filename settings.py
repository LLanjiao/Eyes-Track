import os
from dataclasses import dataclass
from os import environ
from pathlib import Path

e = environ.get


@dataclass
class Settings:
    BASE_DIR = Path(os.path.split(os.path.abspath(__file__))[0])
    ASSETS: Path = BASE_DIR / "gui" / "assets"

    MAINWINDOW_GUI_FILE_PATH: Path = e("MAINWINDOW_GUI_FILE_PATH", ASSETS / "Eyes-Track.ui")
    LOCATEWINDOW_GUI_FILE_PATH: Path = e("LOCATEWINDOW_GUI_FILE_PATH", ASSETS / "locate.ui")
    TRACKWINDOW_GUI_FILE_PATH: Path = e("TRACKWINDOW_GUI_FILE_PATH", ASSETS / "track.ui")

    CASES_FILE_PATH = e("CASES_FILE_PATH", BASE_DIR / "resources")
    PREDICTOR_PATH = e("PREDICTOR_PATH", BASE_DIR / "predictor" / "shape_predictor_68_face_landmarks.dat")

    REFRESH_PERIOD: int = e("CAMERA_REFRESH_PERIOD", 33)
    REFRESH_PERIOD_1S: int = e("REFRESH_PERIOD_1S", 1000)


settings = Settings()
