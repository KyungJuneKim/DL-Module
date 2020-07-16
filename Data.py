from abc import *


class DataSet(metaclass=ABCMeta):
    def __init__(self):
        self.input_size = 100
        self.output_size = 5

    @abstractmethod
    def make_data(self):
        pass

    @abstractmethod
    def print_data(self):
        pass


class PWM(DataSet):
    def make_data(self):
        pass

    def print_data(self):
        pass
