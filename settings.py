import os
from dataclasses import dataclass
from os import environ
from pathlib import Path

e = environ.get


@dataclass
class Settings:

    BASE_DIR = Path(os.path.split(os.path.abspath(__file__))[0])
    ASSETS: Path = BASE_DIR / "gui" / "assets"

    GUI_FILE_PATH: Path = e("GUI_FILE_PATH", ASSETS / "Eyes-Track.ui")

settings = Settings()