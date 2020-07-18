import numpy as np
from abc import *
from random import randrange
from typing import List

from util import split_data_set


class DataSet(metaclass=ABCMeta):
    def __init__(self, factors: List, input_size, output_size):
        self.factors = factors
        self.input_size = input_size
        self.output_size = output_size

        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []

    def make_data(self, num: int = 100, ratio: List[float] = None):
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        x_test = []
        y_test = []

        for factor in self.factors:
            x = []
            y = []
            for i in range(num):
                x.append(self.single_x(factor))
                y.append(self.single_y(factor))

            train_ds_tmp, val_ds_tmp, test_ds_tmp = split_data_set(x, y, ratio)
            x_train += train_ds_tmp[0]
            y_train += train_ds_tmp[1]
            x_val += val_ds_tmp[0]
            y_val += val_ds_tmp[1]
            x_test += test_ds_tmp[0]
            y_test += test_ds_tmp[1]

        self.x_train = self.reshape_x(x_train)
        self.y_train = self.reshape_y(y_train)
        self.x_val = self.reshape_x(x_val)
        self.y_val = self.reshape_y(y_val)
        self.x_test = self.reshape_x(x_test)
        self.y_test = self.reshape_y(y_test)

        return self

    @abstractmethod
    def single_x(self, factor):
        pass

    @abstractmethod
    def single_y(self, factor):
        pass

    @abstractmethod
    def reshape_x(self, x) -> np.ndarray:
        pass

    @abstractmethod
    def reshape_y(self, y) -> np.ndarray:
        pass


class CategoricalPWM(DataSet):
    def __init__(self, ratio: List[float] = None, period: int = 20, cycle: int = 5):
        self.period = period
        self.cycle = cycle
        if ratio is None:
            ratio = [0, 0.25, 0.5, 0.75, 1]
        super().__init__(ratio, period * cycle, len(ratio))

    def single_x(self, factor):
        data = []
        scaling = 1000
        for i in range(self.period):
            if i < factor * self.period:
                data.append((800+randrange(200))/scaling)
            else:
                data.append(randrange(200)/scaling)

        return data * self.cycle

    def single_y(self, factor):
        return self.factors.index(factor)

    def reshape_x(self, x) -> np.ndarray:
        return np.array(x, dtype=np.float32).reshape((-1, self.input_size, 1))

    def reshape_y(self, y) -> np.ndarray:
        return np.eye(len(self.factors), dtype=np.float32)[y]


class RegressionPWM(DataSet):
    def __init__(self, ratio: List[float] = None, period: int = 20, cycle: int = 5):
        self.period = period
        self.cycle = cycle
        if ratio is None:
            ratio = np.arange(0., 1., 0.01).tolist()
        super().__init__(ratio, period * cycle, len(ratio))

    def single_x(self, factor):
        data = []
        scaling = 1000
        for i in range(self.period):
            if i < factor * self.period:
                data.append((800+randrange(200))/scaling)
            else:
                data.append(randrange(200)/scaling)

        return data * self.cycle

    def single_y(self, factor):
        return self.factors

    def reshape_x(self, x) -> np.ndarray:
        return np.array(x, dtype=np.float32).reshape((-1, self.input_size, 1))

    def reshape_y(self, y) -> np.ndarray:
        return np.array(y, dtype=np.float32)
