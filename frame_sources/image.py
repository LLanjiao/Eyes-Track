import cv2

from frame_sources import frameSources


class image(frameSources):
    def __init__(self, filePath):
        self.filePath = filePath
        self.frame = None

    def start(self):
        self.frame = cv2.imread(self.filePath)

    def next_frame(self):
        return True, self.frame

    def stop(self):
        self.frame = None
