import cv2

from frame_sources import frameSources


class video(frameSources):
    def __init__(self, filePath):
        self.videoSources = None
        self.filePath = filePath

    def start(self):
        self.videoSources = cv2.VideoCapture()
        self.videoSources.open(self.filePath)

    def next_frame(self):
        ret, frame = self.videoSources.read()
        return frame

    def stop(self):
        self.videoSources.release()
