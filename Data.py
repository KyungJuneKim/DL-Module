from abc import *
from random import randrange
from typing import List


class DataSet(metaclass=ABCMeta):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.Y_test = []

    def split_data(self, ratio: List[float] = None) -> List:
        pass

    def make_data(self):
        pass

    @abstractmethod
    def single_data(self, factor) -> List:
        pass

    @abstractmethod
    def print_data(self):
        pass


class PWM(DataSet):
    def __init__(self, ratio: List[float] = None, period: int = 20, cycle: int = 5):
        self.ratio = ratio
        if self.ratio is None:
            self.ratio = [0, 0.25, 0.5, 0.75, 1]
        super().__init__(period * cycle, len(self.ratio))
        self.period = period
        self.cycle = cycle

    def single_data(self, factor) -> List:
        data = []
        scaling = 1000
        for i in range(self.period):
            if i < factor * self.period:
                data.append((800+randrange(200))/scaling)
            else:
                data.append(randrange(200)/scaling)

        return data
