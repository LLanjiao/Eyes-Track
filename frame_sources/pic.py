import cv2


class pic:
    def __init__(self):
        self.file = None

    def setLocation(self, file):
        self.file = file

    def getLocation(self):
        return self.file

    def start(self):
        res = cv2.imread(self.file)
        return res
