from abc import ABC, abstractmethod


class frameSources(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def next_frame(self):
        pass

    @abstractmethod
    def stop(self):
        pass
