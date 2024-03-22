import cv2


class pic:
    def __init__(self, file):
        self.file = file

    def start(self):
        res = cv2.imread(self.file)
        return res
