from abc import *
from tensorflow.keras import Sequential


class Model(Sequential, metaclass=ABCMeta):
    def __init__(
            self,
            epoch: int, batch_size: int
    ):
        super().__init__()

        self.epoch: int = epoch
        self.batch_size: int = batch_size

        self.save_path: str = 'model'

    @abstractmethod
    def build_model(self):
        pass

    def save_model(self):
        pass
